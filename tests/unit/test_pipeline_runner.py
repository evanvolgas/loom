"""Unit tests for PipelineRunner."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

from loom.core.models import (
    PipelineConfig,
    ExtractConfig,
    TransformConfig,
    EvaluateConfig,
    LoadConfig,
    Record,
)
from loom.core.types import PipelineStatus, RecordStatus, SourceType, DestinationType
from loom.core.exceptions import PipelineError
from loom.runner.pipeline_runner import PipelineRunner


@pytest.fixture
def pipeline_config():
    """Create a basic pipeline configuration."""
    return PipelineConfig(
        name="test_pipeline",
        version="1.0.0",
        extract=ExtractConfig(
            source="data/test.csv",
            source_type=SourceType.CSV,
        ),
        transform=TransformConfig(
            prompt="prompts/test.txt",
            model="gpt-4o-mini",
        ),
        evaluate=EvaluateConfig(
            evaluators=[
                {"name": "semantic_eval", "type": "semantic", "threshold": 0.8},
            ],
            quality_gate="all_pass",
            batch_threshold=0.8,
        ),
        load=LoadConfig(
            destination="output/test.csv",
            destination_type=DestinationType.CSV,
        ),
    )


@pytest.fixture
def sample_records():
    """Create sample records for testing."""
    return [
        Record(id="rec_1", data={"text": "Sample 1"}),
        Record(id="rec_2", data={"text": "Sample 2"}),
        Record(id="rec_3", data={"text": "Sample 3"}),
    ]


@pytest.fixture
def transformed_records():
    """Create transformed records."""
    return [
        Record(
            id="rec_1",
            data={"text": "Sample 1"},
            transformed_data="Transformed 1",
            status=RecordStatus.TRANSFORMED,
        ),
        Record(
            id="rec_2",
            data={"text": "Sample 2"},
            transformed_data="Transformed 2",
            status=RecordStatus.TRANSFORMED,
        ),
        Record(
            id="rec_3",
            data={"text": "Sample 3"},
            transformed_data="Transformed 3",
            status=RecordStatus.TRANSFORMED,
        ),
    ]


@pytest.fixture
def evaluated_records():
    """Create evaluated records with quality gates."""
    return [
        Record(
            id="rec_1",
            data={"text": "Sample 1"},
            transformed_data="Transformed 1",
            status=RecordStatus.EVALUATED,
            quality_gate_passed=True,
        ),
        Record(
            id="rec_2",
            data={"text": "Sample 2"},
            transformed_data="Transformed 2",
            status=RecordStatus.EVALUATED,
            quality_gate_passed=True,
        ),
        Record(
            id="rec_3",
            data={"text": "Sample 3"},
            transformed_data="Transformed 3",
            status=RecordStatus.EVALUATED,
            quality_gate_passed=False,
        ),
    ]


class TestPipelineRunner:
    """Test PipelineRunner functionality."""

    def test_init(self, pipeline_config):
        """Test runner initialization."""
        runner = PipelineRunner(pipeline_config)

        assert runner.config == pipeline_config
        assert runner.extract_engine is not None
        assert runner.transform_engine is not None
        assert runner.evaluate_engine is not None
        assert runner.load_engine is not None
        assert runner.console is not None

    @pytest.mark.asyncio
    async def test_run_success_all_pass(
        self, pipeline_config, sample_records, transformed_records, evaluated_records, mocker
    ):
        """Test successful pipeline run with all records passing."""
        runner = PipelineRunner(pipeline_config)

        # Mock all engines
        mocker.patch.object(
            runner.extract_engine, "extract", return_value=sample_records
        )
        mocker.patch.object(
            runner.transform_engine, "transform_batch", return_value=transformed_records
        )
        mocker.patch.object(
            runner.evaluate_engine, "evaluate_batch", return_value=evaluated_records
        )
        mocker.patch.object(
            runner.evaluate_engine, "check_batch_threshold", return_value=True
        )
        mocker.patch.object(
            runner.load_engine, "load", return_value=2  # 2 passed records
        )
        mocker.patch.object(
            runner.transform_engine, "close", return_value=None
        )

        # Mock console to suppress output
        mocker.patch.object(runner, "console")

        # Run pipeline
        result = await runner.run()

        # Verify result
        assert result.status == PipelineStatus.PARTIAL  # 1 failed record
        assert result.pipeline_name == "test_pipeline"
        assert result.pipeline_version == "1.0.0"
        assert result.metrics.total_records == 3
        assert result.metrics.extracted_records == 3
        assert result.metrics.transformed_records == 3
        assert result.metrics.evaluated_records == 3
        assert result.metrics.passed_records == 2
        assert result.metrics.failed_records == 1
        assert result.metrics.loaded_records == 2
        assert result.completed_at is not None
        assert result.metrics.total_time > 0

        # Verify engines were called
        runner.extract_engine.extract.assert_called_once()
        runner.transform_engine.transform_batch.assert_called_once_with(sample_records)
        runner.evaluate_engine.evaluate_batch.assert_called_once_with(transformed_records)
        runner.load_engine.load.assert_called_once_with(evaluated_records)
        runner.transform_engine.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_empty_extraction(self, pipeline_config, mocker):
        """Test pipeline run with empty extraction."""
        runner = PipelineRunner(pipeline_config)

        # Mock extract to return empty list
        mocker.patch.object(runner.extract_engine, "extract", return_value=[])
        mocker.patch.object(runner.transform_engine, "close", return_value=None)
        mocker.patch.object(runner, "console")

        result = await runner.run()

        # Verify early termination
        assert result.status == PipelineStatus.COMPLETED
        assert result.metrics.total_records == 0
        assert result.metrics.extracted_records == 0
        assert result.completed_at is not None

        # Verify transform/evaluate/load were NOT called
        runner.extract_engine.extract.assert_called_once()
        assert not hasattr(runner.transform_engine.transform_batch, "called")

    @pytest.mark.asyncio
    async def test_run_batch_quality_gate_failure(
        self, pipeline_config, sample_records, transformed_records, evaluated_records, mocker
    ):
        """Test pipeline run where batch quality gate fails."""
        runner = PipelineRunner(pipeline_config)

        # Mock engines - batch threshold fails
        mocker.patch.object(runner.extract_engine, "extract", return_value=sample_records)
        mocker.patch.object(
            runner.transform_engine, "transform_batch", return_value=transformed_records
        )
        mocker.patch.object(
            runner.evaluate_engine, "evaluate_batch", return_value=evaluated_records
        )
        mocker.patch.object(
            runner.evaluate_engine, "check_batch_threshold", return_value=False
        )
        mocker.patch.object(runner.transform_engine, "close", return_value=None)
        mocker.patch.object(runner, "console")

        result = await runner.run()

        # Verify failure
        assert result.status == PipelineStatus.FAILED
        assert "Batch quality gate failed" in result.error_message
        assert result.metrics.passed_records == 2
        assert result.metrics.total_records == 3
        assert result.completed_at is not None

        # Verify load was NOT called
        assert not hasattr(runner.load_engine.load, "called")

    @pytest.mark.asyncio
    async def test_run_extract_exception(self, pipeline_config, mocker):
        """Test pipeline run when extract stage fails."""
        runner = PipelineRunner(pipeline_config)

        # Mock extract to raise exception
        mocker.patch.object(
            runner.extract_engine, "extract", side_effect=Exception("Extract failed")
        )
        mocker.patch.object(runner.transform_engine, "close", return_value=None)
        mocker.patch.object(runner, "console")

        # Run should raise PipelineError
        with pytest.raises(PipelineError) as exc_info:
            await runner.run()

        assert "Pipeline execution failed" in str(exc_info.value)

        # Verify cleanup was called
        runner.transform_engine.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_transform_exception(
        self, pipeline_config, sample_records, mocker
    ):
        """Test pipeline run when transform stage fails."""
        runner = PipelineRunner(pipeline_config)

        mocker.patch.object(runner.extract_engine, "extract", return_value=sample_records)
        mocker.patch.object(
            runner.transform_engine,
            "transform_batch",
            side_effect=Exception("Transform failed"),
        )
        mocker.patch.object(runner.transform_engine, "close", return_value=None)
        mocker.patch.object(runner, "console")

        with pytest.raises(PipelineError) as exc_info:
            await runner.run()

        assert "Pipeline execution failed" in str(exc_info.value)
        runner.transform_engine.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_stage(self, pipeline_config, sample_records, mocker):
        """Test extract stage execution."""
        runner = PipelineRunner(pipeline_config)

        # Mock extract engine
        mocker.patch.object(runner.extract_engine, "extract", return_value=sample_records)

        # Mock progress
        progress = MagicMock()
        task_id = "task_1"

        records, elapsed = await runner._extract_stage(progress, task_id)

        assert records == sample_records
        assert elapsed >= 0
        progress.update.assert_called_once_with(task_id, completed=True)

    @pytest.mark.asyncio
    async def test_transform_stage(
        self, pipeline_config, sample_records, transformed_records, mocker
    ):
        """Test transform stage execution."""
        runner = PipelineRunner(pipeline_config)

        mocker.patch.object(
            runner.transform_engine, "transform_batch", return_value=transformed_records
        )

        progress = MagicMock()
        task_id = "task_1"

        records, elapsed = await runner._transform_stage(sample_records, progress, task_id)

        assert records == transformed_records
        assert elapsed >= 0
        progress.update.assert_called_once_with(task_id, completed=len(sample_records))

    @pytest.mark.asyncio
    async def test_evaluate_stage(
        self, pipeline_config, transformed_records, evaluated_records, mocker
    ):
        """Test evaluate stage execution."""
        runner = PipelineRunner(pipeline_config)

        mocker.patch.object(
            runner.evaluate_engine, "evaluate_batch", return_value=evaluated_records
        )

        progress = MagicMock()
        task_id = "task_1"

        records, elapsed = await runner._evaluate_stage(
            transformed_records, progress, task_id
        )

        assert records == evaluated_records
        assert elapsed >= 0
        progress.update.assert_called_once_with(task_id, completed=len(transformed_records))

    @pytest.mark.asyncio
    async def test_load_stage(self, pipeline_config, evaluated_records, mocker):
        """Test load stage execution."""
        runner = PipelineRunner(pipeline_config)

        mocker.patch.object(runner.load_engine, "load", return_value=2)

        progress = MagicMock()
        task_id = "task_1"

        loaded_count, elapsed = await runner._load_stage(evaluated_records, progress, task_id)

        assert loaded_count == 2
        assert elapsed >= 0
        progress.update.assert_called_once_with(task_id, completed=2)

    def test_print_summary(self, pipeline_config, mocker):
        """Test summary printing."""
        runner = PipelineRunner(pipeline_config)

        # Mock console
        mock_console = mocker.patch.object(runner, "console")

        # Create a sample run
        from loom.core.models import PipelineRun, PipelineRunMetrics

        run = PipelineRun(
            run_id="test-123",
            pipeline_name="test_pipeline",
            pipeline_version="1.0.0",
            status=PipelineStatus.COMPLETED,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            metrics=PipelineRunMetrics(
                total_records=10,
                extracted_records=10,
                transformed_records=10,
                evaluated_records=10,
                passed_records=8,
                failed_records=2,
                loaded_records=8,
                extract_time=1.0,
                transform_time=2.0,
                evaluate_time=1.5,
                load_time=0.5,
                total_time=5.0,
            ),
        )

        runner._print_summary(run)

        # Verify console.print was called multiple times
        assert mock_console.print.call_count > 10  # Multiple lines printed

    @pytest.mark.asyncio
    async def test_run_all_records_pass(
        self, pipeline_config, sample_records, mocker
    ):
        """Test pipeline run where all records pass quality gates."""
        runner = PipelineRunner(pipeline_config)

        # Create records where all pass
        all_pass_records = [
            Record(
                id=f"rec_{i}",
                data={"text": f"Sample {i}"},
                transformed_data=f"Transformed {i}",
                status=RecordStatus.EVALUATED,
                quality_gate_passed=True,
            )
            for i in range(3)
        ]

        mocker.patch.object(runner.extract_engine, "extract", return_value=sample_records)
        mocker.patch.object(
            runner.transform_engine, "transform_batch", return_value=all_pass_records
        )
        mocker.patch.object(
            runner.evaluate_engine, "evaluate_batch", return_value=all_pass_records
        )
        mocker.patch.object(runner.evaluate_engine, "check_batch_threshold", return_value=True)
        mocker.patch.object(runner.load_engine, "load", return_value=3)
        mocker.patch.object(runner.transform_engine, "close", return_value=None)
        mocker.patch.object(runner, "console")

        result = await runner.run()

        # Should be COMPLETED since all passed
        assert result.status == PipelineStatus.COMPLETED
        assert result.metrics.failed_records == 0
        assert result.metrics.passed_records == 3
        assert result.metrics.loaded_records == 3
