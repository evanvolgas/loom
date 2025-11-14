"""Pipeline runner for orchestrating all stages."""

import time
import uuid
from datetime import datetime
from typing import List

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from loom.core.exceptions import PipelineError
from loom.core.models import PipelineConfig, PipelineRun, Record
from loom.core.types import PipelineStatus
from loom.engines import EvaluateEngine, ExtractEngine, LoadEngine, TransformEngine


class PipelineRunner:
    """Orchestrate complete pipeline execution."""

    def __init__(self, config: PipelineConfig):
        """Initialize pipeline runner.

        Args:
            config: Complete pipeline configuration
        """
        self.config = config
        self.console = Console()

        # Initialize engines
        self.extract_engine = ExtractEngine(config.extract)
        self.transform_engine = TransformEngine(config.transform)
        self.evaluate_engine = EvaluateEngine(config.evaluate)
        self.load_engine = LoadEngine(config.load)

    async def run(self) -> PipelineRun:
        """Execute complete pipeline.

        Returns:
            PipelineRun with execution results and metrics

        Raises:
            PipelineError: If pipeline execution fails
        """
        # Initialize run metadata
        run = PipelineRun(
            run_id=str(uuid.uuid4()),
            pipeline_name=self.config.name,
            pipeline_version=self.config.version,
            status=PipelineStatus.RUNNING,
            started_at=datetime.now(),
        )

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=self.console,
            ) as progress:
                # Stage 1: Extract
                task_extract = progress.add_task("[cyan]Extracting...", total=None)
                records, extract_time = await self._extract_stage(progress, task_extract)
                run.metrics.total_records = len(records)
                run.metrics.extracted_records = len(records)
                run.metrics.extract_time = extract_time

                if not records:
                    run.status = PipelineStatus.COMPLETED
                    run.completed_at = datetime.now()
                    run.metrics.total_time = extract_time
                    return run

                # Stage 2: Transform
                task_transform = progress.add_task(
                    f"[yellow]Transforming {len(records)} records...", total=len(records)
                )
                records, transform_time = await self._transform_stage(
                    records, progress, task_transform
                )
                run.metrics.transformed_records = len(
                    [r for r in records if r.transformed_data]
                )
                run.metrics.transform_time = transform_time

                # Stage 3: Evaluate
                task_evaluate = progress.add_task(
                    f"[green]Evaluating {len(records)} records...", total=len(records)
                )
                records, evaluate_time = await self._evaluate_stage(
                    records, progress, task_evaluate
                )
                run.metrics.evaluated_records = len(
                    [r for r in records if r.quality_gate_passed is not None]
                )
                run.metrics.passed_records = len(
                    [r for r in records if r.quality_gate_passed is True]
                )
                run.metrics.failed_records = len(
                    [r for r in records if r.quality_gate_passed is False]
                )
                run.metrics.evaluate_time = evaluate_time

                # Check batch threshold
                if not self.evaluate_engine.check_batch_threshold(records):
                    run.status = PipelineStatus.FAILED
                    run.error_message = (
                        f"Batch quality gate failed: "
                        f"{run.metrics.passed_records}/{run.metrics.total_records} passed "
                        f"(required: {self.config.evaluate.batch_threshold})"
                    )
                    run.completed_at = datetime.now()
                    run.metrics.total_time = (
                        extract_time + transform_time + evaluate_time
                    )
                    return run

                # Stage 4: Load
                task_load = progress.add_task(
                    f"[blue]Loading {run.metrics.passed_records} passed records...",
                    total=run.metrics.passed_records,
                )
                loaded_count, load_time = await self._load_stage(
                    records, progress, task_load
                )
                run.metrics.loaded_records = loaded_count
                run.metrics.load_time = load_time

            # Complete
            run.status = (
                PipelineStatus.COMPLETED
                if run.metrics.failed_records == 0
                else PipelineStatus.PARTIAL
            )
            run.completed_at = datetime.now()
            run.metrics.total_time = (
                extract_time + transform_time + evaluate_time + load_time
            )

            self._print_summary(run)
            return run

        except Exception as e:
            run.status = PipelineStatus.FAILED
            run.error_message = str(e)
            run.completed_at = datetime.now()
            self.console.print(f"[red]Pipeline failed: {e}[/red]")
            raise PipelineError(f"Pipeline execution failed: {e}")

        finally:
            # Clean up resources
            await self.transform_engine.close()

    async def _extract_stage(
        self, progress: Progress, task_id
    ) -> tuple[List[Record], float]:
        """Execute extract stage.

        Returns:
            Tuple of (records, execution_time)
        """
        start = time.time()
        records = await self.extract_engine.extract()
        elapsed = time.time() - start
        progress.update(task_id, completed=True)
        return records, elapsed

    async def _transform_stage(
        self, records: List[Record], progress: Progress, task_id
    ) -> tuple[List[Record], float]:
        """Execute transform stage.

        Returns:
            Tuple of (records, execution_time)
        """
        start = time.time()
        transformed = await self.transform_engine.transform_batch(records)
        elapsed = time.time() - start
        progress.update(task_id, completed=len(records))
        return transformed, elapsed

    async def _evaluate_stage(
        self, records: List[Record], progress: Progress, task_id
    ) -> tuple[List[Record], float]:
        """Execute evaluate stage.

        Returns:
            Tuple of (records, execution_time)
        """
        start = time.time()
        evaluated = await self.evaluate_engine.evaluate_batch(records)
        elapsed = time.time() - start
        progress.update(task_id, completed=len(records))
        return evaluated, elapsed

    async def _load_stage(
        self, records: List[Record], progress: Progress, task_id
    ) -> tuple[int, float]:
        """Execute load stage.

        Returns:
            Tuple of (loaded_count, execution_time)
        """
        start = time.time()
        loaded_count = await self.load_engine.load(records)
        elapsed = time.time() - start
        progress.update(task_id, completed=loaded_count)
        return loaded_count, elapsed

    def _print_summary(self, run: PipelineRun) -> None:
        """Print execution summary."""
        self.console.print("\n[bold]Pipeline Execution Summary[/bold]")
        self.console.print(f"Run ID: {run.run_id}")
        self.console.print(f"Pipeline: {run.pipeline_name} v{run.pipeline_version}")
        self.console.print(f"Status: {run.status.value}")
        self.console.print("\n[bold]Metrics:[/bold]")
        self.console.print(f"  Total Records: {run.metrics.total_records}")
        self.console.print(f"  Extracted: {run.metrics.extracted_records}")
        self.console.print(f"  Transformed: {run.metrics.transformed_records}")
        self.console.print(f"  Evaluated: {run.metrics.evaluated_records}")
        self.console.print(f"  Passed QG: {run.metrics.passed_records}")
        self.console.print(f"  Failed QG: {run.metrics.failed_records}")
        self.console.print(f"  Loaded: {run.metrics.loaded_records}")
        self.console.print("\n[bold]Timing:[/bold]")
        self.console.print(f"  Extract: {run.metrics.extract_time:.2f}s")
        self.console.print(f"  Transform: {run.metrics.transform_time:.2f}s")
        self.console.print(f"  Evaluate: {run.metrics.evaluate_time:.2f}s")
        self.console.print(f"  Load: {run.metrics.load_time:.2f}s")
        self.console.print(f"  Total: {run.metrics.total_time:.2f}s")
