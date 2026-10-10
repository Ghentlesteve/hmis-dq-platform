"""Command-line entry point: ``hmis-dq <command>``."""

import logging
import subprocess
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import httpx
import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from hmis_dq.config import StoreKind, get_settings
from hmis_dq.dhis2 import DHIS2Client
from hmis_dq.extract.catalog import DEFAULT_DATASETS, resolve_data_set
from hmis_dq.extract.job import ChunkResult, Extractor, new_run_id
from hmis_dq.extract.periods import Month, month_range
from hmis_dq.extract.store import (
    LakeUnavailableError,
    LocalRawStore,
    RawStore,
    S3RawStore,
    open_store,
)
from hmis_dq.extract.sync import sync_stores

if TYPE_CHECKING:
    from hmis_dq.nifi.deploy import Deployer

app = typer.Typer(help="HMIS data quality and early-warning platform.", no_args_is_help=True)
lake_app = typer.Typer(help="Manage the S3 data lake.", no_args_is_help=True)
app.add_typer(lake_app, name="lake")
spark_app = typer.Typer(help="Spark jobs (run inside the spark container).", no_args_is_help=True)
app.add_typer(spark_app, name="spark")
ml_app = typer.Typer(help="Forecasting and anomaly detection (runs locally).", no_args_is_help=True)
app.add_typer(ml_app, name="ml")
nifi_app = typer.Typer(help="The scheduled NiFi ingestion flow.", no_args_is_help=True)
app.add_typer(nifi_app, name="nifi")
console = Console()

StoreOption = Annotated[
    StoreKind | None,
    typer.Option("--store", help="Where to write the bronze layer. Default: HMIS_STORE or lake."),
]


def _open_store(kind: StoreKind | None) -> RawStore:
    try:
        return open_store(get_settings(), kind)
    except LakeUnavailableError as exc:
        console.print(f"[red]Lake unavailable:[/] {exc}")
        raise typer.Exit(code=2) from exc


DISTRICT_LEVEL = 2


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v")] = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )


@app.command()
def ping() -> None:
    """Check the DHIS2 connection and credentials."""
    settings = get_settings()
    with DHIS2Client.from_settings(settings) as client:
        info = client.system_info()
    console.print(f"[green]Connected[/] to {info.system_name} (DHIS2 {info.version})")


def _default_end() -> str:
    return str(Month.from_date(date.today()).shift(-1))


@app.command()
def extract(
    start: Annotated[str, typer.Option(help="First month, YYYY-MM.")] = "2023-01",
    end: Annotated[
        str | None, typer.Option(help="Last month, YYYY-MM. Default: last month.")
    ] = None,
    dataset: Annotated[
        list[str] | None,
        typer.Option("--dataset", "-d", help="Dataset name or UID. Repeatable."),
    ] = None,
    workers: Annotated[int, typer.Option(min=1, max=16, help="Parallel requests.")] = 4,
    refresh_recent: Annotated[
        int,
        typer.Option(min=0, help="Always re-fetch the last N complete months (late reports)."),
    ] = 3,
    force: Annotated[bool, typer.Option(help="Re-fetch chunks that already exist.")] = False,
    skip_metadata: Annotated[bool, typer.Option(help="Don't snapshot metadata.")] = False,
    store: StoreOption = None,
) -> None:
    """Pull DHIS2 metadata and data values into the bronze layer."""
    settings = get_settings()
    months = month_range(Month.parse(start), Month.parse(end or _default_end()))
    data_sets = [resolve_data_set(d) for d in (dataset or DEFAULT_DATASETS)]
    raw_store = _open_store(store)
    run_id = new_run_id()
    console.print(f"Writing to [bold]{_describe(raw_store)}[/]")

    with DHIS2Client.from_settings(settings) as client:
        extractor = Extractor(
            client,
            raw_store,
            source_url=str(settings.dhis2_base_url),
            workers=workers,
            refresh_recent_months=refresh_recent,
        )

        if not skip_metadata:
            with console.status("Snapshotting metadata..."):
                counts = extractor.extract_metadata(run_id)
            console.print(
                "Metadata: " + ", ".join(f"{name} [bold]{n}[/]" for name, n in counts.items())
            )

        districts = [
            ou.id
            for ou in client.organisation_units(level=DISTRICT_LEVEL, fields="id,name,level,path")
        ]
        total = len(data_sets) * len(districts) * len(months)
        console.print(
            f"Run [bold]{run_id}[/]: {len(data_sets)} datasets x {len(districts)} districts "
            f"x {len(months)} months ({months[0]} to {months[-1]}) = {total} chunks"
        )

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Extracting", total=total)

            def on_result(result: ChunkResult) -> None:
                progress.advance(task)

            summary = extractor.extract_data_values(
                data_sets, districts, months, run_id=run_id, force=force, on_result=on_result
            )

    table = Table(title=f"Run {summary.run_id} (DHIS2 {summary.server_version})")
    for column in ("fetched", "skipped", "failed", "records", "MB written"):
        table.add_column(column, justify="right")
    table.add_row(
        str(summary.fetched),
        str(summary.skipped),
        f"[red]{summary.failed}[/]" if summary.failed else "0",
        f"{summary.records:,}",
        f"{summary.bytes_written / 1e6:.1f}",
    )
    console.print(table)

    if not summary.ok:
        for failure in summary.failures[:10]:
            console.print(f"[red]x[/] {failure['key']}: {failure['error']}")
        console.print("Re-run the same command to retry only the failed chunks.")
        raise typer.Exit(code=1)


def _describe(store: RawStore) -> str:
    if isinstance(store, S3RawStore):
        return f"s3://{store.bucket} ({store.client.meta.endpoint_url})"
    if isinstance(store, LocalRawStore):
        return str(store.root)
    return type(store).__name__


@lake_app.command("upload")
def lake_upload(
    prefix: Annotated[str, typer.Option(help="Only copy keys starting with this.")] = "",
    overwrite: Annotated[bool, typer.Option(help="Copy even if the key exists.")] = False,
    workers: Annotated[int, typer.Option(min=1, max=32)] = 8,
) -> None:
    """Copy the local bronze folder (data/raw) into the lake's bronze bucket."""
    settings = get_settings()
    source = LocalRawStore(settings.raw_dir)
    destination = _open_store(StoreKind.LAKE)
    total = sum(1 for _ in source.list_keys(prefix))
    console.print(f"{total} files in {source.root} -> {_describe(destination)}")

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Uploading", total=None)
        result = sync_stores(
            source,
            destination,
            prefix=prefix,
            overwrite=overwrite,
            workers=workers,
            on_planned=lambda n: progress.update(task, total=n),
            on_copied=lambda _: progress.advance(task),
        )

    console.print(
        f"[green]Done:[/] {result.copied} copied, {result.skipped} already in the lake, "
        f"{result.bytes_written / 1e6:.1f} MB written"
    )


@lake_app.command("status")
def lake_status() -> None:
    """Show what is stored in the lake's bronze bucket."""
    store = _open_store(StoreKind.LAKE)
    areas: Counter[str] = Counter()
    datasets: Counter[str] = Counter()
    for key in store.list_keys():
        parts = key.split("/")
        areas["/".join(parts[:2])] += 1
        if parts[1:2] == ["data_value_sets"] and len(parts) > 2:  # noqa: PLR2004
            datasets[parts[2]] += 1

    table = Table(title=f"Lake: {_describe(store)}")
    table.add_column("area")
    table.add_column("objects", justify="right")
    for area, count in sorted(areas.items()):
        table.add_row(area, f"{count:,}")
    for dataset, count in sorted(datasets.items()):
        table.add_row(f"  {dataset}", f"{count:,}")
    console.print(table if areas else "[yellow]The bronze bucket is empty.[/]")


@spark_app.command("smoke")
def spark_smoke() -> None:
    """Read the bronze layer from the lake with Spark and count values per dataset/year."""
    # imported here so the rest of the CLI works on machines without pyspark
    from hmis_dq.spark.smoke import bronze_value_counts  # noqa: PLC0415

    settings = get_settings()
    with console.status("Starting Spark and scanning the bronze bucket..."):
        rows = bronze_value_counts(settings).collect()

    table = Table(title=f"Bronze layer via Spark (s3a://{settings.bronze_bucket})")
    for column in ("dataset", "year", "values", "facilities"):
        table.add_column(column, justify="left" if column == "dataset" else "right")
    for row in rows:
        table.add_row(row.dataset, row.year, f"{row['values']:,}", f"{row.facilities:,}")
    table.add_row("[bold]total[/]", "", f"[bold]{sum(r['values'] for r in rows):,}[/]", "")
    console.print(table)


@spark_app.command("silver")
def spark_silver() -> None:
    """Build the silver layer: bronze JSON -> clean Parquet tables in the silver bucket."""
    from hmis_dq.spark.silver import run_silver  # noqa: PLC0415  (pyspark only needed here)

    settings = get_settings()
    with console.status("Building silver tables (this takes a few minutes)..."):
        result = run_silver(settings)

    table = Table(title=f"Silver layer: s3a://{settings.silver_bucket}/dhis2/")
    table.add_column("table")
    table.add_column("rows", justify="right")
    for name, rows in result.row_counts.items():
        table.add_row(name, f"{rows:,}")
    console.print(table)

    statuses = Table(title="data_values by value_status")
    statuses.add_column("status")
    statuses.add_column("rows", justify="right")
    for status, rows in sorted(result.value_status_counts.items(), key=lambda kv: -kv[1]):
        statuses.add_row(status, f"{rows:,}")
    console.print(statuses)
    console.print(f"Duplicates removed: {result.duplicates_removed:,}")


@spark_app.command("gold")
def spark_gold() -> None:
    """Build the gold layer from silver: facility_month, reporting, district_month."""
    from hmis_dq.spark.gold import run_gold  # noqa: PLC0415  (pyspark only needed here)

    settings = get_settings()
    with console.status("Building gold tables..."):
        result = run_gold(settings)

    table = Table(title=f"Gold layer: s3a://{settings.gold_bucket}/dhis2/")
    table.add_column("table")
    table.add_column("rows", justify="right")
    for name, rows in result.row_counts.items():
        table.add_row(name, f"{rows:,}")
    console.print(table)

    completeness = Table(title="Reporting completeness (whole window)")
    for column in ("dataset", "expected", "received", "completeness"):
        completeness.add_column(column, justify="left" if column == "dataset" else "right")
    for dataset, expected, received, rate in result.completeness:
        completeness.add_row(dataset, f"{expected:,}", f"{received:,}", f"{rate:.1%}")
    console.print(completeness)


@spark_app.command("dq")
def spark_dq() -> None:
    """Run the data quality checks and scores over the gold layer."""
    from hmis_dq.spark.dq.job import run_dq  # noqa: PLC0415  (pyspark only needed here)

    settings = get_settings()
    with console.status("Running data quality checks and scores..."):
        result = run_dq(settings)

    rates = Table(title="Reporting (whole window)")
    for column in ("dataset", "completeness", "timeliness"):
        rates.add_column(column, justify="left" if column == "dataset" else "right")
    for dataset, completeness, timeliness in result.completeness:
        shown = f"{timeliness:.1%}" if timeliness is not None else "[yellow]unknown[/]"
        rates.add_row(dataset, f"{completeness:.1%}", shown)
    console.print(rates)

    severities = ("high", "medium", "low")
    checks = Table(title="Findings by check (every check, including those with none)")
    checks.add_column("check")
    for severity in severities:
        checks.add_column(severity, justify="right")
    for check, counts in result.findings_by_check.items():
        cells = [f"{counts[s]:,}" if s in counts else "[dim]0[/]" for s in severities]
        checks.add_row(check, *cells)
    console.print(checks)

    dimensions = ("completeness", "accuracy", "consistency", "integrity", "overall")

    def score_table(title: str, rows: list[dict[str, object]], label: str) -> Table:
        table = Table(title=title)
        table.add_column(label)
        if label != "dataset_id":
            table.add_column("dataset")
        for dimension in dimensions:
            table.add_column(dimension, justify="right")
        table.add_column("grade", justify="center")
        for row in rows:
            cells = ["-" if row[d] is None else f"{row[d]:.1f}" for d in dimensions]
            first = [str(row[label])] + ([] if label == "dataset_id" else [str(row["dataset_id"])])
            table.add_row(*first, *cells, str(row["grade"]))
        return table

    console.print(score_table("National scores (0-100)", result.national, "dataset_id"))
    console.print(score_table("Lowest-scoring districts", result.worst_districts, "district"))


@spark_app.command("build")
def spark_build() -> None:
    """Run the whole Spark pipeline: silver, gold, then the data quality checks."""
    spark_silver()
    spark_gold()
    spark_dq()


@ml_app.command("backtest")
def ml_backtest(
    test_months: Annotated[int, typer.Option(min=3, max=24, help="Months to replay.")] = 12,
) -> None:
    """Backtest the forecasting models against the seasonal-naive benchmark."""
    import pandas as pd  # noqa: PLC0415  (needs the ml extra)

    from hmis_dq.ml.job import run_forecast_backtest  # noqa: PLC0415

    settings = get_settings()
    with console.status("Backtesting forecasting models..."):
        result = run_forecast_backtest(settings, test_months)

    console.print(
        f"{result.series} district x indicator series, last {test_months} months replayed"
    )
    for dataset_id, rows in result.summary.groupby("dataset_id"):
        table = Table(title=f"{dataset_id}: one-month-ahead forecasts vs seasonal naive")
        for column in ("model", "series", "median skill", "beats benchmark", "median sMAPE"):
            table.add_column(column, justify="left" if column == "model" else "right")
        for row in rows.to_dict("records"):
            skill = row["median_skill"]
            table.add_row(
                str(row["model"]),
                str(row["series"]),
                "-" if pd.isna(skill) else f"{skill:+.2f}",
                f"{row['beats_benchmark']:.0%}",
                f"{row['median_smape']:.1f}%",
            )
        console.print(table)
    console.print(
        "skill = 1 - MAE(model) / MAE(same month last year); above 0 beats the benchmark."
    )


@ml_app.command("anomalies")
def ml_anomalies(
    show: Annotated[int, typer.Option(min=1, max=50, help="How many anomalies to list.")] = 10,
) -> None:
    """Find unusual facility-months across all antigens (Isolation Forest)."""
    from hmis_dq.ml.job import run_anomaly_detection  # noqa: PLC0415  (needs the ml extra)

    with console.status("Scoring facility-months..."):
        result = run_anomaly_detection(get_settings())

    found = result.anomalies
    console.print(
        f"{result.profiles:,} facility-months scored; [bold]{len(found)}[/] flagged as anomalies. "
        f"{result.caught_by_rules:.0%} also have a rule-based outlier that month "
        f"({result.caught_strongly:.0%} a high/medium one)."
    )
    table = Table(title="Most unusual facility-months")
    for column in ("facility", "district", "period", "score", "rules", "why"):
        table.add_column(column, overflow="fold")
    for row in found.head(show).to_dict("records"):
        rules = row["rule_severity"] if isinstance(row["rule_severity"], str) else "[yellow]new[/]"
        table.add_row(
            str(row["facility"]),
            str(row["district"]),
            str(row["period"]),
            f"{row['anomaly_score']:.2f}",
            rules,
            str(row["explanation"]),
        )
    console.print(table)


@ml_app.command("early-warning")
def ml_early_warning() -> None:
    """List districts below their expected range, and facilities with recent drops."""
    from hmis_dq.ml.job import run_early_warning  # noqa: PLC0415  (needs the ml extra)

    with console.status("Checking the latest month against expected ranges..."):
        result = run_early_warning(get_settings())

    months = ", ".join(f"{d} {m}" for d, m in result.latest_month.items())
    console.print(f"Latest month checked: {months}")
    if result.districts.empty:
        console.print("[green]No district is below its expected range.[/]")
    else:
        table = Table(title=f"District warnings ({len(result.districts)})")
        table.add_column("severity")
        table.add_column("warning", overflow="fold")
        for row in result.districts.to_dict("records"):
            colour = "red" if row["severity"] == "high" else "yellow"
            table.add_row(f"[{colour}]{row['severity']}[/]", str(row["message"]))
        console.print(table)

    if not result.history.empty:
        history = Table(title="Warnings over the last 12 months (each month replayed as latest)")
        for column in ("month", "warnings", "service decline", "reporting drop", "high"):
            history.add_column(column, justify="left" if column == "month" else "right")
        by_month = result.history.groupby(result.history["period_start"].dt.strftime("%Y-%m"))
        for month, rows in by_month:
            causes = rows["likely_cause"].value_counts()
            history.add_row(
                str(month),
                str(len(rows)),
                str(causes.get("service decline", 0)),
                str(causes.get("reporting drop", 0)),
                str(int((rows["severity"] == "high").sum())),
            )
        console.print(history)

    if result.facilities.empty:
        console.print("No facility had a multi-antigen drop in the last 3 months.")
    else:
        table = Table(
            title=f"Facilities with recent multi-antigen drops ({len(result.facilities)})"
        )
        for column in ("facility", "district", "period", "why"):
            table.add_column(column, overflow="fold")
        for row in result.facilities.head(15).to_dict("records"):
            table.add_row(
                str(row["facility"]),
                str(row["district"]),
                str(row["period"]),
                str(row["explanation"]),
            )
        console.print(table)


@app.command()
def dashboard(
    port: Annotated[int, typer.Option(help="Port to serve the dashboard on.")] = 8501,
) -> None:
    """Open the data quality dashboard (Streamlit) in the browser."""
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    raise typer.Exit(subprocess.call(command))


# ------------------------------------------------------------------- NiFi


def _nifi() -> "Deployer":
    from hmis_dq.nifi.client import NiFiClient, NiFiError  # noqa: PLC0415
    from hmis_dq.nifi.deploy import Deployer  # noqa: PLC0415

    try:
        client = NiFiClient.from_settings(get_settings())
        client.login()
    except (NiFiError, httpx.TransportError) as exc:
        console.print(f"[red]Can't reach NiFi:[/] {exc}")
        console.print("Is it running?  docker compose --profile nifi up -d")
        raise typer.Exit(code=2) from exc
    return Deployer(client)


def _deployed_group(deployer: "Deployer") -> str:
    from hmis_dq.nifi.flow import FLOW_NAME  # noqa: PLC0415

    group = deployer.find_group(FLOW_NAME)
    if group is None:
        console.print(f"[red]{FLOW_NAME!r} isn't deployed yet:[/] run hmis-dq nifi deploy")
        raise typer.Exit(code=1)
    return str(group["id"])


@nifi_app.command("deploy")
def nifi_deploy(
    replace: Annotated[
        bool, typer.Option(help="Rebuild the flow if it is already deployed.")
    ] = False,
    start: Annotated[bool, typer.Option(help="Start it once it is built.")] = False,
) -> None:
    """Build the DHIS2 -> bronze flow in NiFi, from the code in hmis_dq/nifi/flow.py."""
    from hmis_dq.nifi.client import NiFiError  # noqa: PLC0415
    from hmis_dq.nifi.flow import dhis2_to_bronze  # noqa: PLC0415

    settings = get_settings()
    deployer = _nifi()
    flow = dhis2_to_bronze(settings)
    try:
        with console.status(f"Building {flow.name!r} in NiFi..."):
            deployment = deployer.deploy(flow, replace=replace)
    except NiFiError as exc:
        console.print(f"[red]Deploy failed:[/] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"Built {flow.name!r}: {len(deployment.processors)} processors, "
        f"{len(flow.connections)} connections. Open {settings.nifi_url}nifi to see it."
    )
    if deployment.problems:
        table = Table(title="NiFi says these need fixing before the flow can start")
        table.add_column("component")
        table.add_column("problem", overflow="fold")
        for name, errors in deployment.problems.items():
            for error in errors:
                table.add_row(name, error)
        console.print(table)
        raise typer.Exit(code=1)
    console.print("[green]Every processor and service is valid.[/]")
    if start:
        deployer.start(deployment)
        console.print(f"Started. It runs on its schedule ({settings.nifi_schedule}, Quartz cron);")
        console.print("to fetch now: hmis-dq nifi run")


@nifi_app.command("run")
def nifi_run() -> None:
    """Fetch the latest months now, instead of waiting for the schedule."""
    deployer = _nifi()
    group_id = _deployed_group(deployer)
    with console.status("Asking DHIS2 for the district list..."):
        deployer.run_once(group_id, "List districts")
    console.print("Triggered. The chunks are now flowing; watch them with: hmis-dq nifi status")


@nifi_app.command("status")
def nifi_status() -> None:
    """Show what each step of the flow did in the last 5 minutes."""
    deployer = _nifi()
    rows, failed = deployer.status(_deployed_group(deployer))
    table = Table(title="DHIS2 to bronze (FlowFiles in the last 5 minutes)")
    for column in ("step", "state", "in", "out"):
        table.add_column(column, justify="left" if column in {"step", "state"} else "right")
    for row in rows:
        colour = {"RUNNING": "green", "STOPPED": "yellow"}.get(row["state"], "red")
        table.add_row(row["name"], f"[{colour}]{row['state']}[/]", str(row["in"]), str(row["out"]))
    console.print(table)
    if failed:
        console.print(
            f"[red]{failed} FlowFile(s) parked in 'Failed'.[/] Open NiFi, right-click the "
            "queue into the funnel, and choose List queue to see why."
        )
    else:
        console.print("Nothing has failed.")


if __name__ == "__main__":
    app()
