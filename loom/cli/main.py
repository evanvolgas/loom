"""Command-line interface for Loom."""

import asyncio
import sys

import click
from rich.console import Console

from loom import __version__
from loom.core.exceptions import ConfigurationError, PipelineError, ValidationError
from loom.parsers import parse_pipeline, validate_pipeline
from loom.runner import PipelineRunner

console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="loom")
def cli() -> None:
    """Loom - Declarative orchestration for AI pipelines."""
    pass


@cli.command()
@click.argument("pipeline_path", type=click.Path(exists=True))
def run(pipeline_path: str) -> None:
    """Run a pipeline from YAML definition.

    Example:
        loom run pipelines/customer_sentiment.yaml
    """
    try:
        # Parse pipeline configuration
        console.print(f"[cyan]Loading pipeline: {pipeline_path}[/cyan]")
        config = parse_pipeline(pipeline_path)

        console.print(
            f"[green]Pipeline loaded: {config.name} v{config.version}[/green]\n"
        )

        # Create and run pipeline
        runner = PipelineRunner(config)
        run_result = asyncio.run(runner.run())

        # Exit with appropriate code
        if run_result.status.value == "completed":
            sys.exit(0)
        elif run_result.status.value == "partial":
            console.print("[yellow]Warning: Pipeline completed with some failures[/yellow]")
            sys.exit(0)
        else:
            console.print(f"[red]Pipeline failed: {run_result.error_message}[/red]")
            sys.exit(1)

    except (ConfigurationError, ValidationError) as e:
        console.print(f"[red]Configuration error: {e}[/red]")
        sys.exit(1)
    except PipelineError as e:
        console.print(f"[red]Pipeline error: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("pipeline_path", type=click.Path(exists=True))
def validate(pipeline_path: str) -> None:
    """Validate a pipeline YAML definition.

    Example:
        loom validate pipelines/customer_sentiment.yaml
    """
    try:
        console.print(f"[cyan]Validating pipeline: {pipeline_path}[/cyan]")

        if validate_pipeline(pipeline_path):
            config = parse_pipeline(pipeline_path)
            console.print("[green]✓ Pipeline is valid[/green]")
            console.print(f"  Name: {config.name}")
            console.print(f"  Version: {config.version}")
            console.print(f"  Evaluators: {len(config.evaluate.evaluators)}")
            console.print(f"  Quality Gate: {config.evaluate.quality_gate.value}")
            sys.exit(0)
        else:
            console.print("[red]✗ Pipeline is invalid[/red]")
            sys.exit(1)

    except (ConfigurationError, ValidationError) as e:
        console.print(f"[red]✗ Validation failed: {e}[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Unexpected error: {e}[/red]")
        sys.exit(1)


@cli.command()
def version() -> None:
    """Show Loom version."""
    console.print(f"Loom version {__version__}")


if __name__ == "__main__":
    cli()
