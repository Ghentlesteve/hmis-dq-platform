"""Command-line entry point: ``hmis-dq <command>``."""

import logging
from collections import Counter
from datetime import date
from typing import Annotated

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

app = typer.Typer(help="HMIS data quality and early-warning platform.", no_args_is_help=True)
lake_app = typer.Typer(help="Manage the S3 data lake.", no_args_is_help=True)
app.add_typer(lake_app, name="lake")
spark_app = typer.Typer(help="Spark jobs (run inside the spark container).", no_args_is_help=True)
app.add_typer(spark_app, name="spark")
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


if __name__ == "__main__":
    app()
