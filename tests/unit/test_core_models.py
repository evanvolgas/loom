"""Unit tests for core Pydantic models."""

import pytest
from datetime import datetime
from pydantic import ValidationError

from loom.core.models import (
    Record,
    EvaluatorConfig,
    ExtractConfig,
    TransformConfig,
    EvaluateConfig,
    LoadConfig,
    PipelineConfig,
    PipelineRunMetrics,
    PipelineRun,
)
from loom.core.types import (
    RecordStatus,
    QualityGateType,
    SourceType,
    DestinationType,
    PipelineStatus,
)


class TestRecord:
    """Test Record model."""

    def test_record_creation(self):
        """Test basic record creation."""
        record = Record(
            id="test_1",
            data={"text": "Hello world", "value": 42},
            metadata={"source": "test"},
            status=RecordStatus.PENDING,
        )

        assert record.id == "test_1"
        assert record.data["text"] == "Hello world"
        assert record.data["value"] == 42
        assert record.metadata["source"] == "test"
        assert record.status == RecordStatus.PENDING
        assert record.transformed_data is None
        assert record.evaluation_scores == {}
        assert record.quality_gate_passed is None
        assert record.error is None

    def test_record_defaults(self):
        """Test record with default values."""
        record = Record(id="test_2", data={"key": "value"})

        assert record.metadata == {}
        assert record.status == RecordStatus.PENDING
        assert record.evaluation_scores == {}

    def test_record_with_evaluation(self):
        """Test record with evaluation scores."""
        record = Record(
            id="test_3",
            data={"text": "Test"},
            transformed_data="Transformed test",
            evaluation_scores={"semantic": 0.95, "criteria": 0.88},
            quality_gate_passed=True,
            status=RecordStatus.PASSED,
        )

        assert record.transformed_data == "Transformed test"
        assert record.evaluation_scores["semantic"] == 0.95
        assert record.evaluation_scores["criteria"] == 0.88
        assert record.quality_gate_passed is True
        assert record.status == RecordStatus.PASSED

    def test_record_with_error(self):
        """Test record with error."""
        record = Record(
            id="test_4",
            data={"text": "Test"},
            status=RecordStatus.ERROR,
            error="Processing failed",
        )

        assert record.status == RecordStatus.ERROR
        assert record.error == "Processing failed"

    def test_record_mutability(self):
        """Test that record fields can be updated."""
        record = Record(id="test_5", data={"key": "value"})

        # Update fields
        record.transformed_data = "New data"
        record.evaluation_scores["test"] = 0.9
        record.quality_gate_passed = True
        record.status = RecordStatus.PASSED

        assert record.transformed_data == "New data"
        assert record.evaluation_scores["test"] == 0.9
        assert record.quality_gate_passed is True
        assert record.status == RecordStatus.PASSED


class TestEvaluatorConfig:
    """Test EvaluatorConfig model."""

    def test_evaluator_config_creation(self):
        """Test basic evaluator config creation."""
        config = EvaluatorConfig(
            name="semantic",
            type="semantic",
            threshold=0.8,
        )

        assert config.name == "semantic"
        assert config.type == "semantic"
        assert config.threshold == 0.8
        assert config.weight == 1.0  # default

    def test_evaluator_config_with_weight(self):
        """Test evaluator config with custom weight."""
        config = EvaluatorConfig(
            name="criteria",
            type="custom_criteria",
            threshold=0.75,
            weight=2.0,
            criteria="Is it accurate?",
        )

        assert config.weight == 2.0
        assert config.criteria == "Is it accurate?"

    def test_evaluator_threshold_validation(self):
        """Test threshold must be between 0 and 1."""
        with pytest.raises(ValidationError):
            EvaluatorConfig(name="test", type="semantic", threshold=1.5)

        with pytest.raises(ValidationError):
            EvaluatorConfig(name="test", type="semantic", threshold=-0.1)

    def test_evaluator_weight_validation(self):
        """Test weight must be positive."""
        with pytest.raises(ValidationError):
            EvaluatorConfig(name="test", type="semantic", threshold=0.8, weight=0)

        with pytest.raises(ValidationError):
            EvaluatorConfig(name="test", type="semantic", threshold=0.8, weight=-1.0)

    def test_evaluator_config_frozen(self):
        """Test that evaluator config is immutable."""
        config = EvaluatorConfig(name="test", type="semantic", threshold=0.8)

        with pytest.raises(ValidationError):
            config.threshold = 0.9  # type: ignore


class TestExtractConfig:
    """Test ExtractConfig model."""

    def test_extract_config_minimal(self):
        """Test extract config with minimal fields."""
        config = ExtractConfig(source="data.csv")

        assert config.source == "data.csv"
        assert config.source_type is None  # auto-detect
        assert config.batch_size == 100
        assert config.timeout == 30.0

    def test_extract_config_full(self):
        """Test extract config with all fields."""
        config = ExtractConfig(
            source="data.parquet",
            source_type=SourceType.PARQUET,
            batch_size=50,
            timeout=60.0,
        )

        assert config.source == "data.parquet"
        assert config.source_type == SourceType.PARQUET
        assert config.batch_size == 50
        assert config.timeout == 60.0

    def test_extract_config_validation(self):
        """Test extract config validation."""
        with pytest.raises(ValidationError):
            ExtractConfig(source="data.csv", batch_size=0)

        with pytest.raises(ValidationError):
            ExtractConfig(source="data.csv", timeout=-1.0)


class TestTransformConfig:
    """Test TransformConfig model."""

    def test_transform_config_defaults(self):
        """Test transform config with defaults."""
        config = TransformConfig(prompt="prompts/classify.txt")

        assert config.prompt == "prompts/classify.txt"
        assert config.model == "gpt-4o-mini"
        assert config.provider == "openai"
        assert config.batch_size == 50
        assert config.temperature == 0.7
        assert config.max_tokens is None
        assert config.timeout == 60.0

    def test_transform_config_custom(self):
        """Test transform config with custom values."""
        config = TransformConfig(
            prompt="prompts/summarize.txt",
            model="claude-3-5-sonnet",
            provider="anthropic",
            batch_size=20,
            temperature=0.3,
            max_tokens=500,
            timeout=120.0,
        )

        assert config.model == "claude-3-5-sonnet"
        assert config.provider == "anthropic"
        assert config.batch_size == 20
        assert config.temperature == 0.3
        assert config.max_tokens == 500
        assert config.timeout == 120.0

    def test_transform_config_validation(self):
        """Test transform config validation."""
        with pytest.raises(ValidationError):
            TransformConfig(prompt="test.txt", temperature=2.5)  # too high

        with pytest.raises(ValidationError):
            TransformConfig(prompt="test.txt", temperature=-0.1)  # negative

        with pytest.raises(ValidationError):
            TransformConfig(prompt="test.txt", max_tokens=0)  # must be positive


class TestEvaluateConfig:
    """Test EvaluateConfig model."""

    def test_evaluate_config_minimal(self):
        """Test evaluate config with minimal fields."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="semantic", type="semantic", threshold=0.8)
            ]
        )

        assert len(config.evaluators) == 1
        assert config.quality_gate == QualityGateType.ALL_PASS
        assert config.quality_gate_threshold is None
        assert config.batch_threshold is None
        assert config.timeout == 30.0

    def test_evaluate_config_multiple_evaluators(self):
        """Test evaluate config with multiple evaluators."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="semantic", type="semantic", threshold=0.8, weight=2.0),
                EvaluatorConfig(
                    name="criteria",
                    type="custom_criteria",
                    threshold=0.75,
                    weight=1.0,
                    criteria="Is it good?",
                ),
            ],
            quality_gate=QualityGateType.WEIGHTED,
            quality_gate_threshold=0.75,
            batch_threshold=0.95,
            reference_field="expected",
        )

        assert len(config.evaluators) == 2
        assert config.quality_gate == QualityGateType.WEIGHTED
        assert config.quality_gate_threshold == 0.75
        assert config.batch_threshold == 0.95
        assert config.reference_field == "expected"

    def test_evaluate_config_validation(self):
        """Test evaluate config validation."""
        # Must have at least one evaluator
        with pytest.raises(ValidationError):
            EvaluateConfig(evaluators=[])

        # quality_gate_threshold must be in [0, 1]
        with pytest.raises(ValidationError):
            EvaluateConfig(
                evaluators=[
                    EvaluatorConfig(name="test", type="semantic", threshold=0.8)
                ],
                quality_gate_threshold=1.5,
            )


class TestLoadConfig:
    """Test LoadConfig model."""

    def test_load_config_defaults(self):
        """Test load config with defaults."""
        config = LoadConfig(destination="output.csv")

        assert config.destination == "output.csv"
        assert config.destination_type is None  # auto-detect
        assert config.mode == "append"
        assert config.batch_size == 100
        assert config.timeout == 30.0

    def test_load_config_custom(self):
        """Test load config with custom values."""
        config = LoadConfig(
            destination="output.parquet",
            destination_type=DestinationType.PARQUET,
            mode="overwrite",
            batch_size=200,
            timeout=45.0,
        )

        assert config.destination == "output.parquet"
        assert config.destination_type == DestinationType.PARQUET
        assert config.mode == "overwrite"
        assert config.batch_size == 200
        assert config.timeout == 45.0


class TestPipelineConfig:
    """Test PipelineConfig model."""

    def test_pipeline_config_complete(self):
        """Test complete pipeline configuration."""
        config = PipelineConfig(
            name="test_pipeline",
            version="1.0.0",
            description="Test pipeline",
            extract=ExtractConfig(source="input.csv"),
            transform=TransformConfig(prompt="prompts/test.txt"),
            evaluate=EvaluateConfig(
                evaluators=[
                    EvaluatorConfig(name="semantic", type="semantic", threshold=0.8)
                ]
            ),
            load=LoadConfig(destination="output.csv"),
        )

        assert config.name == "test_pipeline"
        assert config.version == "1.0.0"
        assert config.description == "Test pipeline"
        assert isinstance(config.extract, ExtractConfig)
        assert isinstance(config.transform, TransformConfig)
        assert isinstance(config.evaluate, EvaluateConfig)
        assert isinstance(config.load, LoadConfig)

    def test_pipeline_config_without_description(self):
        """Test pipeline config without optional description."""
        config = PipelineConfig(
            name="test_pipeline",
            version="1.0.0",
            extract=ExtractConfig(source="input.csv"),
            transform=TransformConfig(prompt="prompts/test.txt"),
            evaluate=EvaluateConfig(
                evaluators=[
                    EvaluatorConfig(name="semantic", type="semantic", threshold=0.8)
                ]
            ),
            load=LoadConfig(destination="output.csv"),
        )

        assert config.description is None


class TestPipelineRunMetrics:
    """Test PipelineRunMetrics model."""

    def test_pipeline_run_metrics_defaults(self):
        """Test pipeline run metrics with defaults."""
        metrics = PipelineRunMetrics()

        assert metrics.total_records == 0
        assert metrics.extracted_records == 0
        assert metrics.transformed_records == 0
        assert metrics.evaluated_records == 0
        assert metrics.passed_records == 0
        assert metrics.failed_records == 0
        assert metrics.loaded_records == 0
        assert metrics.error_records == 0
        assert metrics.extract_time == 0.0
        assert metrics.transform_time == 0.0
        assert metrics.evaluate_time == 0.0
        assert metrics.load_time == 0.0
        assert metrics.total_time == 0.0

    def test_pipeline_run_metrics_with_values(self):
        """Test pipeline run metrics with values."""
        metrics = PipelineRunMetrics(
            total_records=100,
            extracted_records=100,
            transformed_records=100,
            evaluated_records=100,
            passed_records=95,
            failed_records=5,
            loaded_records=95,
            error_records=0,
            extract_time=5.2,
            transform_time=45.8,
            evaluate_time=12.3,
            load_time=3.1,
            total_time=66.4,
        )

        assert metrics.total_records == 100
        assert metrics.passed_records == 95
        assert metrics.failed_records == 5
        assert metrics.total_time == 66.4

    def test_pipeline_run_metrics_mutable(self):
        """Test that metrics can be updated."""
        metrics = PipelineRunMetrics()

        metrics.total_records = 50
        metrics.passed_records = 45
        metrics.total_time = 30.5

        assert metrics.total_records == 50
        assert metrics.passed_records == 45
        assert metrics.total_time == 30.5


class TestPipelineRun:
    """Test PipelineRun model."""

    def test_pipeline_run_creation(self):
        """Test pipeline run creation."""
        run = PipelineRun(
            run_id="run_123",
            pipeline_name="test_pipeline",
            pipeline_version="1.0.0",
            status=PipelineStatus.PENDING,
        )

        assert run.run_id == "run_123"
        assert run.pipeline_name == "test_pipeline"
        assert run.pipeline_version == "1.0.0"
        assert run.status == PipelineStatus.PENDING
        assert run.started_at is None
        assert run.completed_at is None
        assert isinstance(run.metrics, PipelineRunMetrics)
        assert run.error_message is None

    def test_pipeline_run_complete_lifecycle(self):
        """Test complete pipeline run lifecycle."""
        started = datetime.now()
        completed = datetime.now()

        run = PipelineRun(
            run_id="run_456",
            pipeline_name="test_pipeline",
            pipeline_version="1.0.0",
            status=PipelineStatus.COMPLETED,
            started_at=started,
            completed_at=completed,
            metrics=PipelineRunMetrics(
                total_records=100,
                passed_records=95,
                failed_records=5,
                total_time=60.0,
            ),
        )

        assert run.status == PipelineStatus.COMPLETED
        assert run.started_at == started
        assert run.completed_at == completed
        assert run.metrics.total_records == 100

    def test_pipeline_run_with_error(self):
        """Test pipeline run with error."""
        run = PipelineRun(
            run_id="run_789",
            pipeline_name="test_pipeline",
            pipeline_version="1.0.0",
            status=PipelineStatus.FAILED,
            error_message="Pipeline failed due to timeout",
        )

        assert run.status == PipelineStatus.FAILED
        assert run.error_message == "Pipeline failed due to timeout"

    def test_pipeline_run_mutable(self):
        """Test that pipeline run can be updated."""
        run = PipelineRun(
            run_id="run_999",
            pipeline_name="test",
            pipeline_version="1.0.0",
        )

        # Update status and timestamps
        run.status = PipelineStatus.RUNNING
        run.started_at = datetime.now()
        run.metrics.total_records = 50

        assert run.status == PipelineStatus.RUNNING
        assert run.started_at is not None
        assert run.metrics.total_records == 50
