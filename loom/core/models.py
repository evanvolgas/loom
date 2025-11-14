"""Core Pydantic data models for Loom."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .types import (
    DestinationType,
    PipelineStatus,
    QualityGateType,
    RecordStatus,
    SourceType,
)


class Record(BaseModel):
    """Single data record flowing through pipeline."""

    model_config = ConfigDict(frozen=False)

    id: str = Field(..., description="Unique record identifier")
    data: Dict[str, Any] = Field(..., description="Record data payload")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Record metadata"
    )
    status: RecordStatus = Field(
        default=RecordStatus.PENDING, description="Processing status"
    )
    transformed_data: Optional[str] = Field(
        default=None, description="Transformed text output"
    )
    evaluation_scores: Dict[str, float] = Field(
        default_factory=dict, description="Evaluation scores by evaluator name"
    )
    quality_gate_passed: Optional[bool] = Field(
        default=None, description="Whether record passed quality gate"
    )
    error: Optional[str] = Field(default=None, description="Error message if failed")


class EvaluatorConfig(BaseModel):
    """Configuration for a single evaluator."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Evaluator identifier")
    type: str = Field(..., description="Evaluator type (semantic, custom_criteria)")
    threshold: float = Field(..., ge=0.0, le=1.0, description="Pass threshold")
    weight: float = Field(default=1.0, gt=0.0, description="Weight for weighted gate")
    criteria: Optional[str] = Field(
        default=None, description="Criteria for custom_criteria evaluator"
    )


class ExtractConfig(BaseModel):
    """Extract stage configuration."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(..., description="Data source URI or path")
    source_type: Optional[SourceType] = Field(
        default=None, description="Source type (auto-detected from path if None)"
    )
    batch_size: int = Field(default=100, gt=0, description="Records per batch")
    timeout: float = Field(
        default=30.0, gt=0.0, description="Timeout in seconds per batch"
    )


class TransformConfig(BaseModel):
    """Transform stage configuration."""

    model_config = ConfigDict(frozen=True)

    prompt: str = Field(..., description="Path to prompt template file")
    model: str = Field(
        default="gpt-4o-mini", description="LLM model identifier"
    )
    provider: str = Field(
        default="openai", description="LLM provider (openai, anthropic, google, groq)"
    )
    batch_size: int = Field(default=50, gt=0, description="Records per batch")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="LLM temperature"
    )
    max_tokens: Optional[int] = Field(
        default=None, gt=0, description="Max tokens per response"
    )
    timeout: float = Field(
        default=60.0, gt=0.0, description="Timeout in seconds per LLM call"
    )


class EvaluateConfig(BaseModel):
    """Evaluate stage configuration."""

    model_config = ConfigDict(frozen=True)

    evaluators: List[EvaluatorConfig] = Field(
        ..., min_length=1, description="List of evaluators to run"
    )
    quality_gate: QualityGateType = Field(
        default=QualityGateType.ALL_PASS, description="Quality gate strategy"
    )
    quality_gate_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Threshold for weighted quality gate",
    )
    batch_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of records that must pass",
    )
    reference_field: Optional[str] = Field(
        default=None, description="Field name for reference text (for semantic eval)"
    )
    timeout: float = Field(
        default=30.0, gt=0.0, description="Timeout in seconds per evaluation"
    )


class LoadConfig(BaseModel):
    """Load stage configuration."""

    model_config = ConfigDict(frozen=True)

    destination: str = Field(..., description="Data destination URI or path")
    destination_type: Optional[DestinationType] = Field(
        default=None, description="Destination type (auto-detected from path if None)"
    )
    mode: str = Field(
        default="append", description="Load mode (append, overwrite, upsert)"
    )
    batch_size: int = Field(default=100, gt=0, description="Records per batch")
    timeout: float = Field(
        default=30.0, gt=0.0, description="Timeout in seconds per batch"
    )


class PipelineConfig(BaseModel):
    """Complete pipeline configuration."""

    model_config = ConfigDict(frozen=True)

    name: str = Field(..., description="Pipeline name")
    version: str = Field(..., description="Pipeline version")
    description: Optional[str] = Field(default=None, description="Pipeline description")
    extract: ExtractConfig = Field(..., description="Extract stage config")
    transform: TransformConfig = Field(..., description="Transform stage config")
    evaluate: EvaluateConfig = Field(..., description="Evaluate stage config")
    load: LoadConfig = Field(..., description="Load stage config")


class PipelineRunMetrics(BaseModel):
    """Metrics for a pipeline run."""

    model_config = ConfigDict(frozen=False)

    total_records: int = Field(default=0, description="Total records processed")
    extracted_records: int = Field(default=0, description="Successfully extracted")
    transformed_records: int = Field(default=0, description="Successfully transformed")
    evaluated_records: int = Field(default=0, description="Successfully evaluated")
    passed_records: int = Field(default=0, description="Passed quality gate")
    failed_records: int = Field(default=0, description="Failed quality gate")
    loaded_records: int = Field(default=0, description="Successfully loaded")
    error_records: int = Field(default=0, description="Errored during processing")

    extract_time: float = Field(default=0.0, description="Extract stage duration (s)")
    transform_time: float = Field(
        default=0.0, description="Transform stage duration (s)"
    )
    evaluate_time: float = Field(default=0.0, description="Evaluate stage duration (s)")
    load_time: float = Field(default=0.0, description="Load stage duration (s)")
    total_time: float = Field(default=0.0, description="Total pipeline duration (s)")


class PipelineRun(BaseModel):
    """Pipeline execution run metadata."""

    model_config = ConfigDict(frozen=False)

    run_id: str = Field(..., description="Unique run identifier")
    pipeline_name: str = Field(..., description="Pipeline name")
    pipeline_version: str = Field(..., description="Pipeline version")
    status: PipelineStatus = Field(
        default=PipelineStatus.PENDING, description="Execution status"
    )
    started_at: Optional[datetime] = Field(default=None, description="Start timestamp")
    completed_at: Optional[datetime] = Field(
        default=None, description="Completion timestamp"
    )
    metrics: PipelineRunMetrics = Field(
        default_factory=PipelineRunMetrics, description="Run metrics"
    )
    error_message: Optional[str] = Field(
        default=None, description="Error message if failed"
    )
