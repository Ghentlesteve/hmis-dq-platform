"""Command-line entry point: ``hmis-dq <command>``."""

import logging
from datetime import date
from typing import Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from hmis_dq.config import get_settings
from hmis_dq.dhis2 import DHIS2Client
from hmis_dq.extract.catalog import DEFAULT_DATASETS, resolve_data_set
from hmis_dq.extract.job import ChunkResult, Extractor, new_run_id
from hmis_dq.extract.periods import Month, month_range
from hmis_dq.extract.store import LocalRawStore

app = typer.Typer(help="HMIS data quality and early-warning platform.", no_args_is_help=True)
console = Console()

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
        int, typer.Option(min=0, help="Always re-fetch this many most recent months.")
    ] = 3,
    force: Annotated[bool, typer.Option(help="Re-fetch chunks that already exist.")] = False,
    skip_metadata: Annotated[bool, typer.Option(help="Don't snapshot metadata.")] = False,
) -> None:
    """Pull DHIS2 metadata and data values into the bronze layer."""
    settings = get_settings()
    months = month_range(Month.parse(start), Month.parse(end or _default_end()))
    data_sets = [resolve_data_set(d) for d in (dataset or DEFAULT_DATASETS)]
    store = LocalRawStore(settings.raw_dir)
    run_id = new_run_id()

    with DHIS2Client.from_settings(settings) as client:
        extractor = Extractor(
            client,
            store,
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


if __name__ == "__main__":
    app()
