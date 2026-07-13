"""Console script for spectoprep."""

import typer
from rich.console import Console

from spectoprep import __version__
from spectoprep.logging import configure_logging

app = typer.Typer(help="SpectoPrep: Bayesian optimization of spectral preprocessing.")
console = Console()


@app.callback()
def main(
    log_level: str = typer.Option("INFO", "--log-level", help="Logging verbosity."),
    json_logs: bool = typer.Option(False, "--json-logs", help="Emit JSON logs."),
) -> None:
    """Configure logging for all subcommands."""
    configure_logging(level=log_level, json_logs=json_logs, force=True)


@app.command()
def version() -> None:
    """Print the installed SpectoPrep version."""
    console.print(f"spectoprep {__version__}")


@app.command()
def info() -> None:
    """Show a short summary of available preprocessing steps."""
    from spectoprep.pipeline.config import AVAILABLE_STEPS

    console.print(f"[bold]SpectoPrep[/bold] {__version__}")
    console.print(f"{len(AVAILABLE_STEPS)} preprocessing transforms available:")
    for key, description in AVAILABLE_STEPS.items():
        console.print(f"  [cyan]{key}[/cyan]: {description}")


if __name__ == "__main__":
    app()
