"""Core type definitions and enums for Loom."""

from enum import Enum


class StageType(str, Enum):
    """Pipeline stage types."""

    EXTRACT = "extract"
    TRANSFORM = "transform"
    EVALUATE = "evaluate"
    LOAD = "load"


class QualityGateType(str, Enum):
    """Quality gate evaluation strategies."""

    ALL_PASS = "all_pass"
    MAJORITY_PASS = "majority_pass"
    ANY_PASS = "any_pass"
    WEIGHTED = "weighted"


class SourceType(str, Enum):
    """Data source types (Phase 1: local files only)."""

    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    PARQUET = "parquet"
    # Phase 2+:
    # POSTGRES = "postgres"
    # MYSQL = "mysql"
    # S3 = "s3"
    # HTTP = "http"


class DestinationType(str, Enum):
    """Data destination types (Phase 1: local files only)."""

    CSV = "csv"
    JSON = "json"
    JSONL = "jsonl"
    PARQUET = "parquet"
    # Phase 2+:
    # POSTGRES = "postgres"
    # MYSQL = "mysql"
    # S3 = "s3"


class PipelineStatus(str, Enum):
    """Pipeline execution status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"  # Some records failed quality gates


class RecordStatus(str, Enum):
    """Individual record processing status."""

    PENDING = "pending"
    EXTRACTED = "extracted"
    TRANSFORMED = "transformed"
    EVALUATED = "evaluated"
    PASSED = "passed"  # Passed quality gates
    FAILED = "failed"  # Failed quality gates
    LOADED = "loaded"
    ERROR = "error"  # Processing error
