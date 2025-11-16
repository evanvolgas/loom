# Loom Implementation Specification

**Version:** 0.1.0-alpha
**Status:** Implementation Design
**Created:** 2025-11-14
**Supplements:** DESIGN_SPEC.md

---

## Purpose

This document provides concrete implementation details for Loom's Phase 1 development. It addresses technical design gaps identified in DESIGN_SPEC.md and provides specific patterns, schemas, and code structures.

---

## Table of Contents

1. [Extract/Load Connector Architecture](#extractload-connector-architecture)
2. [Storage Schema Design](#storage-schema-design)
3. [Error Handling & Retry Strategy](#error-handling--retry-strategy)
4. [Credentials Management](#credentials-management)
5. [Transform Engine Design](#transform-engine-design)
6. [Quality Gate Semantics](#quality-gate-semantics)
7. [Airflow Integration](#airflow-integration)
8. [Memory Management](#memory-management)
9. [Configuration System](#configuration-system)
10. [Phase 1 Implementation Checklist](#phase-1-implementation-checklist)

---

## Extract/Load Connector Architecture

### Base Abstraction

```python
# loom/connectors/base.py
from abc import ABC, abstractmethod
from typing import AsyncIterator, List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Record:
    """Single data record flowing through pipeline."""
    id: str
    data: Dict[str, Any]
    metadata: Dict[str, Any]

@dataclass
class ExtractResult:
    """Result of extract operation."""
    records: List[Record]
    total_count: int
    has_more: bool
    cursor: Optional[str] = None  # For pagination

@dataclass
class LoadResult:
    """Result of load operation."""
    success_count: int
    failure_count: int
    errors: List[Dict[str, Any]]

class DataConnector(ABC):
    """Base class for all data connectors."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get("name", self.__class__.__name__)

    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to data source."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to data source."""
        pass

    @abstractmethod
    async def extract(
        self,
        query: str,
        batch_size: int = 100,
        cursor: Optional[str] = None
    ) -> ExtractResult:
        """Extract records in batches."""
        pass

    @abstractmethod
    async def load(
        self,
        records: List[Record],
        destination: str
    ) -> LoadResult:
        """Load records to destination."""
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """Verify connection is valid."""
        pass
```

### Postgres Connector

```python
# loom/connectors/postgres.py
import asyncpg
from typing import AsyncIterator, List, Optional
from .base import DataConnector, Record, ExtractResult, LoadResult

class PostgresConnector(DataConnector):
    """PostgreSQL data connector."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        """Establish connection pool."""
        self.pool = await asyncpg.create_pool(
            host=self.config["host"],
            port=self.config.get("port", 5432),
            database=self.config["database"],
            user=self.config["user"],
            password=self.config["password"],
            min_size=self.config.get("min_pool_size", 1),
            max_size=self.config.get("max_pool_size", 10)
        )

    async def disconnect(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()

    async def extract(
        self,
        query: str,
        batch_size: int = 100,
        cursor: Optional[str] = None
    ) -> ExtractResult:
        """Extract records using SQL query."""
        async with self.pool.acquire() as conn:
            # Add LIMIT/OFFSET for batching
            if cursor:
                offset = int(cursor)
                paginated_query = f"{query} LIMIT {batch_size} OFFSET {offset}"
            else:
                offset = 0
                paginated_query = f"{query} LIMIT {batch_size}"

            rows = await conn.fetch(paginated_query)

            # Convert rows to Record objects
            records = [
                Record(
                    id=str(row["id"]) if "id" in row else f"row_{offset + i}",
                    data=dict(row),
                    metadata={"source": "postgres", "table": self._extract_table(query)}
                )
                for i, row in enumerate(rows)
            ]

            # Check if more records exist
            total_count = await self._get_total_count(conn, query)
            has_more = (offset + len(records)) < total_count
            next_cursor = str(offset + batch_size) if has_more else None

            return ExtractResult(
                records=records,
                total_count=total_count,
                has_more=has_more,
                cursor=next_cursor
            )

    async def load(
        self,
        records: List[Record],
        destination: str
    ) -> LoadResult:
        """Load records using INSERT statements."""
        success_count = 0
        failure_count = 0
        errors = []

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                for record in records:
                    try:
                        # Build INSERT statement
                        columns = list(record.data.keys())
                        values = [record.data[col] for col in columns]
                        placeholders = [f"${i+1}" for i in range(len(columns))]

                        query = f"""
                            INSERT INTO {destination} ({', '.join(columns)})
                            VALUES ({', '.join(placeholders)})
                        """

                        await conn.execute(query, *values)
                        success_count += 1
                    except Exception as e:
                        failure_count += 1
                        errors.append({
                            "record_id": record.id,
                            "error": str(e)
                        })

        return LoadResult(
            success_count=success_count,
            failure_count=failure_count,
            errors=errors
        )

    async def validate_connection(self) -> bool:
        """Verify connection works."""
        try:
            async with self.pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception:
            return False

    def _extract_table(self, query: str) -> str:
        """Extract table name from query (naive implementation)."""
        # Simple regex to find FROM table_name
        import re
        match = re.search(r'FROM\s+(\w+)', query, re.IGNORECASE)
        return match.group(1) if match else "unknown"

    async def _get_total_count(self, conn, query: str) -> int:
        """Get total count for pagination."""
        count_query = f"SELECT COUNT(*) FROM ({query}) as subquery"
        return await conn.fetchval(count_query)
```

### S3 Connector

```python
# loom/connectors/s3.py
import aioboto3
from typing import List, Optional
from .base import DataConnector, Record, ExtractResult, LoadResult

class S3Connector(DataConnector):
    """AWS S3 data connector."""

    async def connect(self) -> None:
        """Initialize S3 client."""
        self.session = aioboto3.Session()

    async def disconnect(self) -> None:
        """Close S3 client."""
        pass  # aioboto3 handles cleanup

    async def extract(
        self,
        query: str,  # S3 path pattern: s3://bucket/prefix/*
        batch_size: int = 100,
        cursor: Optional[str] = None
    ) -> ExtractResult:
        """Extract objects from S3."""
        # Parse S3 URI
        bucket, prefix = self._parse_s3_uri(query)

        async with self.session.client('s3') as s3:
            # List objects
            list_params = {
                'Bucket': bucket,
                'Prefix': prefix,
                'MaxKeys': batch_size
            }
            if cursor:
                list_params['ContinuationToken'] = cursor

            response = await s3.list_objects_v2(**list_params)

            # Download and convert to Records
            records = []
            for obj in response.get('Contents', []):
                content = await self._download_object(s3, bucket, obj['Key'])
                records.append(Record(
                    id=obj['Key'],
                    data={"content": content, "key": obj['Key']},
                    metadata={"bucket": bucket, "size": obj['Size']}
                ))

            return ExtractResult(
                records=records,
                total_count=response.get('KeyCount', 0),
                has_more=response.get('IsTruncated', False),
                cursor=response.get('NextContinuationToken')
            )

    async def load(
        self,
        records: List[Record],
        destination: str  # S3 URI: s3://bucket/prefix/
    ) -> LoadResult:
        """Upload records to S3."""
        bucket, prefix = self._parse_s3_uri(destination)
        success_count = 0
        failure_count = 0
        errors = []

        async with self.session.client('s3') as s3:
            for record in records:
                try:
                    key = f"{prefix}/{record.id}.json"
                    await s3.put_object(
                        Bucket=bucket,
                        Key=key,
                        Body=json.dumps(record.data).encode('utf-8')
                    )
                    success_count += 1
                except Exception as e:
                    failure_count += 1
                    errors.append({
                        "record_id": record.id,
                        "error": str(e)
                    })

        return LoadResult(
            success_count=success_count,
            failure_count=failure_count,
            errors=errors
        )

    async def validate_connection(self) -> bool:
        """Verify S3 access."""
        try:
            async with self.session.client('s3') as s3:
                await s3.list_buckets()
            return True
        except Exception:
            return False

    def _parse_s3_uri(self, uri: str) -> tuple:
        """Parse s3://bucket/prefix into components."""
        parts = uri.replace("s3://", "").split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""
        return bucket, prefix

    async def _download_object(self, s3, bucket: str, key: str) -> str:
        """Download S3 object content."""
        response = await s3.get_object(Bucket=bucket, Key=key)
        async with response['Body'] as stream:
            return await stream.read()
```

### Connector Registry

```python
# loom/connectors/registry.py
from typing import Dict, Type
from .base import DataConnector
from .postgres import PostgresConnector
from .s3 import S3Connector

class ConnectorRegistry:
    """Registry for data connectors."""

    _connectors: Dict[str, Type[DataConnector]] = {
        "postgres": PostgresConnector,
        "postgresql": PostgresConnector,
        "s3": S3Connector,
    }

    @classmethod
    def register(cls, scheme: str, connector_class: Type[DataConnector]):
        """Register custom connector."""
        cls._connectors[scheme] = connector_class

    @classmethod
    def get_connector(cls, uri: str, config: Dict[str, Any]) -> DataConnector:
        """Get connector instance for URI scheme."""
        scheme = uri.split("://")[0]
        connector_class = cls._connectors.get(scheme)

        if not connector_class:
            raise ValueError(f"No connector registered for scheme: {scheme}")

        return connector_class(config)
```

---

## Storage Schema Design

### Complete SQL Schema

```sql
-- loom/storage/schema.sql

-- Pipeline definitions metadata
CREATE TABLE pipelines (
    name VARCHAR(255) PRIMARY KEY,
    version VARCHAR(50) NOT NULL,
    description TEXT,
    definition JSONB NOT NULL,  -- Full YAML as JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Pipeline runs (execution history)
CREATE TABLE pipeline_runs (
    run_id VARCHAR(255) PRIMARY KEY,
    pipeline_name VARCHAR(255) REFERENCES pipelines(name),
    pipeline_version VARCHAR(50) NOT NULL,
    execution_date TIMESTAMP NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    status VARCHAR(50) NOT NULL,  -- running, success, failed, partial
    error_message TEXT,

    -- Configuration snapshot
    prompt_version VARCHAR(50),
    model VARCHAR(100),
    model_version VARCHAR(50),
    context_versions JSONB,  -- {"customer_domain": "v1.5", ...}

    -- Results
    records_processed INTEGER DEFAULT 0,
    records_validated INTEGER DEFAULT 0,
    records_rejected INTEGER DEFAULT 0,
    rejection_rate FLOAT,

    -- Cost tracking
    cost_total DECIMAL(10, 6),
    cost_breakdown JSONB,  -- {"transform": 1.80, "evaluate": 0.54}

    -- Performance
    duration_seconds FLOAT,
    latency_p50 FLOAT,
    latency_p95 FLOAT,
    latency_p99 FLOAT,

    CONSTRAINT valid_status CHECK (status IN ('running', 'success', 'failed', 'partial'))
);

CREATE INDEX idx_pipeline_runs_pipeline ON pipeline_runs(pipeline_name);
CREATE INDEX idx_pipeline_runs_status ON pipeline_runs(status);
CREATE INDEX idx_pipeline_runs_execution_date ON pipeline_runs(execution_date);

-- Evaluation results per run
CREATE TABLE evaluation_results (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) REFERENCES pipeline_runs(run_id),
    evaluator_name VARCHAR(100) NOT NULL,
    evaluator_type VARCHAR(50) NOT NULL,  -- semantic, custom_criteria, etc.

    -- Scores
    mean_score FLOAT,
    std_score FLOAT,
    min_score FLOAT,
    max_score FLOAT,

    -- Configuration
    threshold FLOAT,
    passed BOOLEAN,

    -- Metadata
    sample_size INTEGER,
    metadata JSONB,  -- Evaluator-specific data

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_evaluation_results_run ON evaluation_results(run_id);
CREATE INDEX idx_evaluation_results_evaluator ON evaluation_results(evaluator_name);

-- Individual record evaluations (detailed tracking)
CREATE TABLE record_evaluations (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) REFERENCES pipeline_runs(run_id),
    record_id VARCHAR(255) NOT NULL,

    -- Evaluation
    evaluator_name VARCHAR(100) NOT NULL,
    score FLOAT,
    passed BOOLEAN,
    confidence FLOAT,
    explanation TEXT,

    -- Timing
    evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    latency_ms FLOAT,

    -- Metadata
    metadata JSONB
);

CREATE INDEX idx_record_evaluations_run ON record_evaluations(run_id);
CREATE INDEX idx_record_evaluations_passed ON record_evaluations(passed);

-- Lineage tracking (input/output relationships)
CREATE TABLE pipeline_lineage (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) REFERENCES pipeline_runs(run_id),

    -- Input lineage
    input_source VARCHAR(500),
    input_query TEXT,
    input_record_count INTEGER,

    -- Output lineage
    output_destination VARCHAR(500),
    output_record_count INTEGER,

    -- Dependencies
    upstream_pipelines JSONB,  -- ["pipeline1:v1.0", "pipeline2:v2.1"]
    downstream_pipelines JSONB,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_pipeline_lineage_run ON pipeline_lineage(run_id);

-- Failed/quarantined records
CREATE TABLE quarantined_records (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(255) REFERENCES pipeline_runs(run_id),
    record_id VARCHAR(255) NOT NULL,

    -- Original data
    record_data JSONB NOT NULL,

    -- Failure info
    failure_stage VARCHAR(50),  -- extract, transform, evaluate, load
    failure_reason TEXT,
    error_details JSONB,

    -- Evaluation scores (if available)
    evaluation_scores JSONB,

    -- Metadata
    quarantined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved BOOLEAN DEFAULT FALSE,
    resolved_at TIMESTAMP,
    resolution_action VARCHAR(100)  -- reprocessed, discarded, manual_fix
);

CREATE INDEX idx_quarantined_records_run ON quarantined_records(run_id);
CREATE INDEX idx_quarantined_records_resolved ON quarantined_records(resolved);

-- Cost tracking by pipeline
CREATE TABLE pipeline_costs (
    id SERIAL PRIMARY KEY,
    pipeline_name VARCHAR(255) REFERENCES pipelines(name),
    date DATE NOT NULL,

    -- Aggregated costs
    total_runs INTEGER,
    total_records_processed INTEGER,
    total_cost DECIMAL(10, 6),

    -- Breakdown
    cost_by_stage JSONB,  -- {"transform": 45.20, "evaluate": 12.30}
    cost_by_model JSONB,  -- {"gpt-4o-mini": 40.50, "gpt-4o": 17.00}

    -- Statistics
    avg_cost_per_run DECIMAL(10, 6),
    avg_cost_per_record DECIMAL(10, 6),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(pipeline_name, date)
);

CREATE INDEX idx_pipeline_costs_date ON pipeline_costs(date);

-- Prompt versions
CREATE TABLE prompt_versions (
    id SERIAL PRIMARY KEY,
    prompt_name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,

    -- Content
    content TEXT NOT NULL,
    template_engine VARCHAR(50) DEFAULT 'jinja2',

    -- Context requirements
    required_context JSONB,  -- ["customer_domain", "sentiment_taxonomy"]

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),

    UNIQUE(prompt_name, version)
);

CREATE INDEX idx_prompt_versions_name ON prompt_versions(prompt_name);

-- Context versions
CREATE TABLE context_versions (
    id SERIAL PRIMARY KEY,
    context_name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,

    -- Content
    content TEXT NOT NULL,
    format VARCHAR(50) DEFAULT 'markdown',

    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_by VARCHAR(255),

    UNIQUE(context_name, version)
);

CREATE INDEX idx_context_versions_name ON context_versions(context_name);

-- Monitoring alerts
CREATE TABLE monitoring_alerts (
    id SERIAL PRIMARY KEY,
    pipeline_name VARCHAR(255) REFERENCES pipelines(name),
    run_id VARCHAR(255) REFERENCES pipeline_runs(run_id),

    -- Alert type
    alert_type VARCHAR(100) NOT NULL,  -- score_degradation, cost_spike, latency_sla
    severity VARCHAR(50) NOT NULL,  -- warning, error, critical

    -- Details
    message TEXT NOT NULL,
    details JSONB,
    threshold_value FLOAT,
    actual_value FLOAT,

    -- Status
    triggered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    acknowledged BOOLEAN DEFAULT FALSE,
    acknowledged_at TIMESTAMP,
    acknowledged_by VARCHAR(255),

    -- Notification
    notification_channel VARCHAR(100),  -- slack, pagerduty, email
    notification_sent BOOLEAN DEFAULT FALSE,
    notification_sent_at TIMESTAMP
);

CREATE INDEX idx_monitoring_alerts_pipeline ON monitoring_alerts(pipeline_name);
CREATE INDEX idx_monitoring_alerts_triggered ON monitoring_alerts(triggered_at);
CREATE INDEX idx_monitoring_alerts_acknowledged ON monitoring_alerts(acknowledged);
```

### SQLAlchemy Models

```python
# loom/storage/models.py
from sqlalchemy import Column, String, Integer, Float, Boolean, TIMESTAMP, JSON, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

class Pipeline(Base):
    __tablename__ = 'pipelines'

    name = Column(String(255), primary_key=True)
    version = Column(String(50), nullable=False)
    description = Column(Text)
    definition = Column(JSON, nullable=False)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())

class PipelineRun(Base):
    __tablename__ = 'pipeline_runs'

    run_id = Column(String(255), primary_key=True)
    pipeline_name = Column(String(255), ForeignKey('pipelines.name'))
    pipeline_version = Column(String(50), nullable=False)
    execution_date = Column(TIMESTAMP, nullable=False)
    started_at = Column(TIMESTAMP, server_default=func.now())
    completed_at = Column(TIMESTAMP)
    status = Column(String(50), nullable=False)
    error_message = Column(Text)

    # Configuration
    prompt_version = Column(String(50))
    model = Column(String(100))
    model_version = Column(String(50))
    context_versions = Column(JSON)

    # Results
    records_processed = Column(Integer, default=0)
    records_validated = Column(Integer, default=0)
    records_rejected = Column(Integer, default=0)
    rejection_rate = Column(Float)

    # Cost
    cost_total = Column(Float)
    cost_breakdown = Column(JSON)

    # Performance
    duration_seconds = Column(Float)
    latency_p50 = Column(Float)
    latency_p95 = Column(Float)
    latency_p99 = Column(Float)

# ... similar models for other tables
```

---

## Error Handling & Retry Strategy

### Error Hierarchy

```python
# loom/errors.py
class LoomError(Exception):
    """Base exception for all Loom errors."""
    pass

class ConfigurationError(LoomError):
    """Pipeline configuration is invalid."""
    pass

class ConnectorError(LoomError):
    """Data connector error."""
    pass

class ExtractError(ConnectorError):
    """Error during data extraction."""
    pass

class LoadError(ConnectorError):
    """Error during data loading."""
    pass

class TransformError(LoomError):
    """Error during AI transformation."""
    pass

class EvaluationError(LoomError):
    """Error during evaluation."""
    pass

class QualityGateError(LoomError):
    """Quality gate failed."""
    pass

class RetryableError(LoomError):
    """Error that can be retried."""
    pass
```

### Retry Strategy

```python
# loom/retry.py
import asyncio
from typing import Callable, TypeVar, Optional
from dataclasses import dataclass

T = TypeVar('T')

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    initial_delay: float = 1.0
    exponential_base: float = 2.0
    max_delay: float = 60.0
    retryable_exceptions: tuple = (RetryableError, asyncio.TimeoutError)

async def retry_with_backoff(
    func: Callable[..., T],
    config: RetryConfig,
    *args,
    **kwargs
) -> T:
    """Execute function with exponential backoff retry."""
    last_exception = None

    for attempt in range(config.max_attempts):
        try:
            return await func(*args, **kwargs)
        except config.retryable_exceptions as e:
            last_exception = e

            if attempt < config.max_attempts - 1:
                delay = min(
                    config.initial_delay * (config.exponential_base ** attempt),
                    config.max_delay
                )
                await asyncio.sleep(delay)
            else:
                raise last_exception
```

### Error Recovery Strategy

```python
# loom/recovery.py
from enum import Enum
from typing import List, Optional
from dataclasses import dataclass

class FailureStage(Enum):
    EXTRACT = "extract"
    TRANSFORM = "transform"
    EVALUATE = "evaluate"
    LOAD = "load"

class RecoveryAction(Enum):
    RETRY = "retry"
    SKIP = "skip"
    QUARANTINE = "quarantine"
    FALLBACK = "fallback"
    ABORT = "abort"

@dataclass
class RecoveryConfig:
    """Configuration for error recovery."""
    action: RecoveryAction
    retry_config: Optional[RetryConfig] = None
    fallback_strategy: Optional[str] = None
    max_failures_per_batch: int = 10
    max_failure_rate: float = 0.1  # 10%

class RecoveryManager:
    """Manages error recovery for pipeline stages."""

    def __init__(self, config: RecoveryConfig, storage):
        self.config = config
        self.storage = storage

    async def handle_extract_error(
        self,
        run_id: str,
        record: Record,
        error: Exception
    ) -> RecoveryAction:
        """Handle extraction error."""
        if isinstance(error, RetryableError) and self.config.action == RecoveryAction.RETRY:
            # Retry extraction
            return RecoveryAction.RETRY
        elif self.config.action == RecoveryAction.SKIP:
            # Skip failed record, continue processing
            await self._log_skipped_record(run_id, record, error, FailureStage.EXTRACT)
            return RecoveryAction.SKIP
        else:
            # Abort pipeline
            return RecoveryAction.ABORT

    async def handle_transform_error(
        self,
        run_id: str,
        record: Record,
        error: Exception
    ) -> RecoveryAction:
        """Handle transformation error."""
        if self.config.action == RecoveryAction.QUARANTINE:
            # Quarantine failed record for later inspection
            await self._quarantine_record(run_id, record, error, FailureStage.TRANSFORM)
            return RecoveryAction.QUARANTINE
        elif self.config.action == RecoveryAction.FALLBACK:
            # Use fallback strategy (cheaper model, cached response, etc.)
            return RecoveryAction.FALLBACK
        else:
            return RecoveryAction.ABORT

    async def handle_evaluate_error(
        self,
        run_id: str,
        record: Record,
        error: Exception
    ) -> RecoveryAction:
        """Handle evaluation error."""
        # Evaluation errors are typically non-retryable
        # Quarantine for manual review
        await self._quarantine_record(run_id, record, error, FailureStage.EVALUATE)
        return RecoveryAction.QUARANTINE

    async def handle_load_error(
        self,
        run_id: str,
        record: Record,
        error: Exception
    ) -> RecoveryAction:
        """Handle loading error."""
        if isinstance(error, RetryableError):
            # Retry load operation
            return RecoveryAction.RETRY
        else:
            # Quarantine - data was transformed and evaluated successfully
            await self._quarantine_record(run_id, record, error, FailureStage.LOAD)
            return RecoveryAction.QUARANTINE

    async def check_failure_threshold(
        self,
        run_id: str,
        total_records: int,
        failed_records: int
    ) -> bool:
        """Check if failure rate exceeds threshold."""
        if total_records == 0:
            return False

        failure_rate = failed_records / total_records

        if failed_records > self.config.max_failures_per_batch:
            return True
        if failure_rate > self.config.max_failure_rate:
            return True

        return False

    async def _quarantine_record(
        self,
        run_id: str,
        record: Record,
        error: Exception,
        stage: FailureStage
    ):
        """Store failed record in quarantine table."""
        await self.storage.quarantine_record(
            run_id=run_id,
            record_id=record.id,
            record_data=record.data,
            failure_stage=stage.value,
            failure_reason=str(error),
            error_details={
                "exception_type": type(error).__name__,
                "exception_message": str(error)
            }
        )

    async def _log_skipped_record(
        self,
        run_id: str,
        record: Record,
        error: Exception,
        stage: FailureStage
    ):
        """Log skipped record for monitoring."""
        # Could write to separate skip log table if needed
        pass
```

---

## Credentials Management

### Credential Provider Interface

```python
# loom/credentials.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Credential:
    """Single credential entry."""
    key: str
    value: str
    metadata: Optional[Dict[str, Any]] = None

class CredentialProvider(ABC):
    """Base interface for credential providers."""

    @abstractmethod
    async def get_credential(self, key: str) -> Optional[Credential]:
        """Retrieve credential by key."""
        pass

    @abstractmethod
    async def list_credentials(self, prefix: str = "") -> List[Credential]:
        """List all credentials matching prefix."""
        pass
```

### Environment Variable Provider

```python
# loom/credentials/env.py
import os
from typing import Optional, List
from .base import CredentialProvider, Credential

class EnvCredentialProvider(CredentialProvider):
    """Load credentials from environment variables."""

    def __init__(self, prefix: str = "LOOM_"):
        self.prefix = prefix

    async def get_credential(self, key: str) -> Optional[Credential]:
        """Get credential from environment."""
        env_key = f"{self.prefix}{key.upper()}"
        value = os.getenv(env_key)

        if value:
            return Credential(key=key, value=value)
        return None

    async def list_credentials(self, prefix: str = "") -> List[Credential]:
        """List all credentials with prefix."""
        credentials = []
        search_prefix = f"{self.prefix}{prefix.upper()}"

        for key, value in os.environ.items():
            if key.startswith(search_prefix):
                credential_key = key[len(self.prefix):].lower()
                credentials.append(Credential(key=credential_key, value=value))

        return credentials
```

### AWS Secrets Manager Provider

```python
# loom/credentials/aws_secrets.py
import aioboto3
import json
from typing import Optional, List
from .base import CredentialProvider, Credential

class AWSSecretsProvider(CredentialProvider):
    """Load credentials from AWS Secrets Manager."""

    def __init__(self, region: str = "us-east-1"):
        self.region = region
        self.session = aioboto3.Session()

    async def get_credential(self, key: str) -> Optional[Credential]:
        """Get credential from AWS Secrets Manager."""
        async with self.session.client('secretsmanager', region_name=self.region) as client:
            try:
                response = await client.get_secret_value(SecretId=key)

                # Parse JSON secret
                if 'SecretString' in response:
                    secret = json.loads(response['SecretString'])
                    return Credential(
                        key=key,
                        value=secret,
                        metadata={
                            "version": response.get('VersionId'),
                            "created": response.get('CreatedDate')
                        }
                    )
                return None
            except client.exceptions.ResourceNotFoundException:
                return None

    async def list_credentials(self, prefix: str = "") -> List[Credential]:
        """List all secrets matching prefix."""
        async with self.session.client('secretsmanager', region_name=self.region) as client:
            response = await client.list_secrets()
            credentials = []

            for secret in response.get('SecretList', []):
                name = secret['Name']
                if name.startswith(prefix):
                    cred = await self.get_credential(name)
                    if cred:
                        credentials.append(cred)

            return credentials
```

### Credential Manager

```python
# loom/credentials/manager.py
from typing import Dict, Any, Optional
from .base import CredentialProvider
from .env import EnvCredentialProvider
from .aws_secrets import AWSSecretsProvider

class CredentialManager:
    """Central credential management."""

    def __init__(self, provider_type: str = "env", **kwargs):
        self.provider = self._create_provider(provider_type, **kwargs)

    def _create_provider(self, provider_type: str, **kwargs) -> CredentialProvider:
        """Factory for credential providers."""
        providers = {
            "env": EnvCredentialProvider,
            "aws_secrets": AWSSecretsProvider,
        }

        provider_class = providers.get(provider_type)
        if not provider_class:
            raise ValueError(f"Unknown provider type: {provider_type}")

        return provider_class(**kwargs)

    async def resolve_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve credential placeholders in config."""
        resolved = {}

        for key, value in config.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Extract credential key: ${CREDENTIAL_NAME}
                credential_key = value[2:-1]
                credential = await self.provider.get_credential(credential_key)
                resolved[key] = credential.value if credential else value
            elif isinstance(value, dict):
                # Recursively resolve nested dicts
                resolved[key] = await self.resolve_config(value)
            else:
                resolved[key] = value

        return resolved
```

### Usage in Pipeline Config

```yaml
# pipelines/customer_sentiment.yaml
extract:
  source: postgres://${DB_USER}:${DB_PASSWORD}@localhost/customers

transform:
  model: gpt-4o-mini
  api_key: ${OPENAI_API_KEY}
```

```python
# Resolve credentials at runtime
cred_manager = CredentialManager(provider_type="env")
resolved_config = await cred_manager.resolve_config(pipeline_config)
```

---

## Transform Engine Design

### Transform Engine Architecture

```python
# loom/engines/transform.py
from typing import List, Dict, Any, Optional
import asyncio
from dataclasses import dataclass

# Import Arbiter LLM client
from arbiter.core import LLMManager

@dataclass
class TransformConfig:
    """Configuration for transform engine."""
    prompt_path: str
    prompt_version: str
    model: str
    model_version: str
    temperature: float = 0.0
    context_files: List[str] = None
    batch_size: int = 50
    cache_enabled: bool = True
    cache_ttl_days: int = 7
    max_retries: int = 3

class TransformEngine:
    """AI transformation engine using Arbiter's LLM client."""

    def __init__(
        self,
        config: TransformConfig,
        cache_backend,
        prompt_loader
    ):
        self.config = config
        self.cache = cache_backend
        self.prompt_loader = prompt_loader
        self.llm_client = None

    async def initialize(self):
        """Initialize LLM client from Arbiter."""
        self.llm_client = await LLMManager.get_client(
            model=self.config.model,
            temperature=self.config.temperature
        )

    async def transform_batch(
        self,
        records: List[Record]
    ) -> List[TransformResult]:
        """Transform batch of records using AI."""
        # Load prompt and context
        prompt_template = await self.prompt_loader.load_prompt(
            self.config.prompt_path,
            self.config.prompt_version
        )
        context = await self.prompt_loader.load_context(
            self.config.context_files
        )

        # Process records (potentially in parallel)
        tasks = [
            self._transform_record(record, prompt_template, context)
            for record in records
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        transform_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                transform_results.append(TransformResult(
                    record=records[i],
                    success=False,
                    error=str(result)
                ))
            else:
                transform_results.append(result)

        return transform_results

    async def _transform_record(
        self,
        record: Record,
        prompt_template: str,
        context: Dict[str, str]
    ) -> TransformResult:
        """Transform single record."""
        # Check cache first
        cache_key = self._generate_cache_key(record, prompt_template)

        if self.config.cache_enabled:
            cached_result = await self.cache.get(cache_key)
            if cached_result:
                return TransformResult.from_cache(cached_result)

        # Render prompt with record data and context
        prompt = self._render_prompt(prompt_template, record.data, context)

        # Call LLM using Arbiter's client
        try:
            response = await self.llm_client.generate(
                prompt=prompt,
                max_tokens=1000
            )

            # Parse structured output
            output = self._parse_response(response)

            result = TransformResult(
                record=record,
                success=True,
                output=output,
                tokens_used=response.usage.total_tokens,
                latency=response.latency
            )

            # Cache result
            if self.config.cache_enabled:
                await self.cache.set(
                    cache_key,
                    result,
                    ttl_days=self.config.cache_ttl_days
                )

            return result

        except Exception as e:
            return TransformResult(
                record=record,
                success=False,
                error=str(e)
            )

    def _generate_cache_key(self, record: Record, prompt: str) -> str:
        """Generate cache key from record + prompt + model."""
        import hashlib
        content = f"{record.data}|{prompt}|{self.config.model}|{self.config.prompt_version}"
        return hashlib.sha256(content.encode()).hexdigest()

    def _render_prompt(
        self,
        template: str,
        record_data: Dict[str, Any],
        context: Dict[str, str]
    ) -> str:
        """Render prompt template with data."""
        from jinja2 import Template
        tmpl = Template(template)
        return tmpl.render(input=record_data, context=context)

    def _parse_response(self, response) -> Dict[str, Any]:
        """Parse LLM response into structured output."""
        import json
        # Try to parse JSON from response
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            # Fallback to raw text
            return {"text": response.text}

@dataclass
class TransformResult:
    """Result of transforming a single record."""
    record: Record
    success: bool
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    tokens_used: int = 0
    latency: float = 0.0
    from_cache: bool = False
```

### Cache Backend

```python
# loom/cache/redis.py
import aioredis
import json
from typing import Optional, Any
from datetime import timedelta

class RedisCache:
    """Redis-based caching for transform results."""

    def __init__(self, redis_url: str = "redis://localhost"):
        self.redis_url = redis_url
        self.redis = None

    async def connect(self):
        """Connect to Redis."""
        self.redis = await aioredis.from_url(self.redis_url)

    async def get(self, key: str) -> Optional[Any]:
        """Get cached value."""
        value = await self.redis.get(key)
        if value:
            return json.loads(value)
        return None

    async def set(self, key: str, value: Any, ttl_days: int = 7):
        """Set cached value with TTL."""
        await self.redis.setex(
            key,
            timedelta(days=ttl_days),
            json.dumps(value)
        )

    async def delete(self, key: str):
        """Delete cached value."""
        await self.redis.delete(key)

    async def clear_prefix(self, prefix: str):
        """Clear all keys matching prefix."""
        cursor = b'0'
        while cursor:
            cursor, keys = await self.redis.scan(cursor, match=f"{prefix}*")
            if keys:
                await self.redis.delete(*keys)
```

---

## Quality Gate Semantics

### Quality Gate Implementation

```python
# loom/quality_gates.py
from enum import Enum
from typing import List
from dataclasses import dataclass

class QualityGateType(Enum):
    ALL_PASS = "all_pass"  # All evaluators must pass
    MAJORITY_PASS = "majority_pass"  # >50% evaluators pass
    ANY_PASS = "any_pass"  # At least one evaluator passes
    WEIGHTED = "weighted"  # Weighted average above threshold

@dataclass
class QualityGateConfig:
    """Configuration for quality gate."""
    gate_type: QualityGateType
    threshold: float = 0.7  # For weighted gates
    per_record: bool = True  # Apply per record or per batch

class QualityGate:
    """Quality gate checker."""

    def __init__(self, config: QualityGateConfig):
        self.config = config

    def check_record(
        self,
        evaluation_results: List[EvaluationResult]
    ) -> bool:
        """Check if single record passes quality gate."""
        if not evaluation_results:
            return False

        if self.config.gate_type == QualityGateType.ALL_PASS:
            return all(r.passed for r in evaluation_results)

        elif self.config.gate_type == QualityGateType.MAJORITY_PASS:
            passed_count = sum(1 for r in evaluation_results if r.passed)
            return passed_count > len(evaluation_results) / 2

        elif self.config.gate_type == QualityGateType.ANY_PASS:
            return any(r.passed for r in evaluation_results)

        elif self.config.gate_type == QualityGateType.WEIGHTED:
            avg_score = sum(r.score for r in evaluation_results) / len(evaluation_results)
            return avg_score >= self.config.threshold

        return False

    def check_batch(
        self,
        batch_results: List[List[EvaluationResult]]
    ) -> BatchQualityResult:
        """Check quality gate for entire batch."""
        passed_records = []
        failed_records = []

        for record_results in batch_results:
            if self.check_record(record_results):
                passed_records.append(record_results)
            else:
                failed_records.append(record_results)

        total = len(batch_results)
        passed = len(passed_records)

        return BatchQualityResult(
            total_records=total,
            passed_records=passed,
            failed_records=total - passed,
            pass_rate=passed / total if total > 0 else 0.0
        )

@dataclass
class BatchQualityResult:
    """Result of batch quality gate check."""
    total_records: int
    passed_records: int
    failed_records: int
    pass_rate: float
```

---

## Airflow Integration

### Loom Operator for Airflow

```python
# loom/integrations/airflow.py
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from typing import Optional, Dict, Any

class LoomPipelineOperator(BaseOperator):
    """Airflow operator to run Loom pipelines."""

    @apply_defaults
    def __init__(
        self,
        pipeline_name: str,
        pipeline_version: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        loom_config_path: str = "~/.loom/config.yaml",
        fail_on_quality_gate: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.pipeline_name = pipeline_name
        self.pipeline_version = pipeline_version
        self.params = params or {}
        self.loom_config_path = loom_config_path
        self.fail_on_quality_gate = fail_on_quality_gate

    def execute(self, context):
        """Execute Loom pipeline."""
        from loom.cli import LoomCLI

        # Initialize Loom CLI
        cli = LoomCLI(config_path=self.loom_config_path)

        # Add Airflow context to params
        params = {
            **self.params,
            "execution_date": context["execution_date"],
            "dag_run_id": context["dag_run"].run_id,
        }

        # Run pipeline
        result = cli.run_pipeline(
            name=self.pipeline_name,
            version=self.pipeline_version,
            params=params
        )

        # Check status
        if result.status == "failed":
            raise Exception(f"Pipeline failed: {result.error_message}")

        if result.status == "partial" and self.fail_on_quality_gate:
            raise Exception(f"Pipeline quality gate failed: {result.rejection_rate:.1%} rejected")

        # Return metadata for XCom
        return {
            "run_id": result.run_id,
            "records_processed": result.records_processed,
            "records_validated": result.records_validated,
            "cost_total": result.cost_total
        }
```

### Usage Example

```python
# airflow_dag.py
from airflow import DAG
from loom.integrations.airflow import LoomPipelineOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'data-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 14),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'customer_sentiment_pipeline',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
)

# Run Loom pipeline as Airflow task
sentiment_analysis = LoomPipelineOperator(
    task_id='analyze_sentiment',
    pipeline_name='customer_sentiment',
    params={
        "execution_date": "{{ ds }}",
    },
    dag=dag
)

# Downstream task using XCom
generate_report = PythonOperator(
    task_id='generate_report',
    python_callable=lambda **context: print(
        f"Processed {context['ti'].xcom_pull(task_ids='analyze_sentiment')['records_validated']} records"
    ),
    dag=dag
)

sentiment_analysis >> generate_report
```

---

## Memory Management

### Streaming Architecture

```python
# loom/streaming.py
from typing import AsyncIterator
import asyncio

class StreamProcessor:
    """Process records in streaming fashion to manage memory."""

    def __init__(self, batch_size: int = 100, max_queue_size: int = 1000):
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size

    async def process_stream(
        self,
        extractor,
        transformer,
        evaluator,
        loader
    ) -> PipelineResult:
        """Process pipeline stages as streams with backpressure."""

        # Create async queues for pipeline stages
        extract_queue = asyncio.Queue(maxsize=self.max_queue_size)
        transform_queue = asyncio.Queue(maxsize=self.max_queue_size)
        evaluate_queue = asyncio.Queue(maxsize=self.max_queue_size)

        # Start pipeline stages concurrently
        extract_task = asyncio.create_task(
            self._extract_stage(extractor, extract_queue)
        )
        transform_task = asyncio.create_task(
            self._transform_stage(transformer, extract_queue, transform_queue)
        )
        evaluate_task = asyncio.create_task(
            self._evaluate_stage(evaluator, transform_queue, evaluate_queue)
        )
        load_task = asyncio.create_task(
            self._load_stage(loader, evaluate_queue)
        )

        # Wait for all stages to complete
        results = await asyncio.gather(
            extract_task,
            transform_task,
            evaluate_task,
            load_task
        )

        return self._aggregate_results(results)

    async def _extract_stage(self, extractor, output_queue):
        """Extract stage - read from source and push to queue."""
        cursor = None
        while True:
            result = await extractor.extract(batch_size=self.batch_size, cursor=cursor)
            for record in result.records:
                await output_queue.put(record)

            if not result.has_more:
                break
            cursor = result.cursor

        await output_queue.put(None)  # Signal completion

    async def _transform_stage(self, transformer, input_queue, output_queue):
        """Transform stage - read from input queue, transform, push to output."""
        batch = []

        while True:
            record = await input_queue.get()
            if record is None:  # Completion signal
                if batch:
                    # Process final batch
                    results = await transformer.transform_batch(batch)
                    for result in results:
                        await output_queue.put(result)
                await output_queue.put(None)
                break

            batch.append(record)

            if len(batch) >= self.batch_size:
                # Process full batch
                results = await transformer.transform_batch(batch)
                for result in results:
                    await output_queue.put(result)
                batch = []

    async def _evaluate_stage(self, evaluator, input_queue, output_queue):
        """Evaluate stage - read from input, evaluate, push passed records."""
        while True:
            result = await input_queue.get()
            if result is None:
                await output_queue.put(None)
                break

            if result.success:
                evaluation = await evaluator.evaluate(result)
                if evaluation.passed:
                    await output_queue.put(result)

    async def _load_stage(self, loader, input_queue):
        """Load stage - read from input and write to destination."""
        batch = []

        while True:
            result = await input_queue.get()
            if result is None:
                if batch:
                    await loader.load(batch)
                break

            batch.append(result)

            if len(batch) >= self.batch_size:
                await loader.load(batch)
                batch = []
```

---

## Configuration System

### Configuration Hierarchy

```python
# loom/config.py
from typing import Dict, Any, Optional
import yaml
import os
from pathlib import Path

class LoomConfig:
    """Central configuration management."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._default_config_path()
        self.config = self._load_config()

    def _default_config_path(self) -> str:
        """Get default config path."""
        return os.path.expanduser("~/.loom/config.yaml")

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        if not os.path.exists(self.config_path):
            return self._default_config()

        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _default_config(self) -> Dict[str, Any]:
        """Default configuration."""
        return {
            "storage": {
                "type": "sqlite",
                "path": "~/.loom/loom.db"
            },
            "credentials": {
                "provider": "env",
                "prefix": "LOOM_"
            },
            "cache": {
                "enabled": True,
                "backend": "redis",
                "url": "redis://localhost",
                "ttl_days": 7
            },
            "defaults": {
                "batch_size": 100,
                "max_retries": 3,
                "quality_gate": "all_pass"
            }
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default

        return value if value is not None else default
```

---

## Phase 1 Implementation Checklist

### Core Infrastructure (Week 1-2)

- [ ] **Storage Layer**
  - [ ] Create SQL schema (schema.sql)
  - [ ] Implement SQLAlchemy models
  - [ ] Create database initialization script
  - [ ] Add migration support (Alembic)

- [ ] **Connector System**
  - [ ] Implement base DataConnector abstraction
  - [ ] Implement PostgresConnector
  - [ ] Implement S3Connector
  - [ ] Create ConnectorRegistry
  - [ ] Add connector validation

- [ ] **Configuration & Credentials**
  - [ ] Implement LoomConfig
  - [ ] Implement CredentialManager
  - [ ] Add EnvCredentialProvider
  - [ ] Add config validation

### Pipeline Engine (Week 2-3)

- [ ] **YAML Parser**
  - [ ] Parse pipeline definitions
  - [ ] Validate pipeline schema
  - [ ] Load prompt and context files
  - [ ] Support template variables

- [ ] **Pipeline Executor**
  - [ ] Implement Extract engine
  - [ ] Implement Transform engine (Arbiter integration)
  - [ ] Implement basic Load engine
  - [ ] Add error handling and retry logic

- [ ] **Error Handling**
  - [ ] Define error hierarchy
  - [ ] Implement retry with backoff
  - [ ] Implement RecoveryManager
  - [ ] Add quarantine table support

### CLI (Week 3-4)

- [ ] **Basic Commands**
  - [ ] `loom init` - Initialize loom config
  - [ ] `loom validate <pipeline>` - Validate pipeline definition
  - [ ] `loom run <pipeline>` - Execute pipeline
  - [ ] Add structured logging

- [ ] **Output Formatting**
  - [ ] Progress bars for pipeline stages
  - [ ] Colored output for success/failure
  - [ ] Cost and performance metrics display

### Testing (Week 4)

- [ ] **Unit Tests**
  - [ ] Test connector implementations
  - [ ] Test pipeline parser
  - [ ] Test error handling
  - [ ] Test credential management

- [ ] **Integration Tests**
  - [ ] End-to-end pipeline execution
  - [ ] Test with real Postgres database
  - [ ] Test with S3 (using localstack)

### Documentation

- [ ] **Setup Guide**
  - [ ] Installation instructions
  - [ ] Configuration guide
  - [ ] First pipeline tutorial

- [ ] **API Documentation**
  - [ ] Connector API docs
  - [ ] Pipeline YAML reference
  - [ ] CLI command reference

---

## Next Steps

After completing Phase 1 implementation:

1. **Phase 2: Arbiter Integration**
   - Implement Evaluate engine using Arbiter
   - Add quality gate logic
   - Implement evaluation storage

2. **Phase 3: Observability**
   - Full lineage tracking
   - Cost reporting
   - Performance dashboards

3. **Phase 4: Testing Framework**
   - Unit test runner
   - Distributional tests
   - Integration test support

---

## Conclusion

This implementation specification provides the concrete technical designs needed to build Loom Phase 1. It addresses:

✅ Extract/Load connector architecture with Postgres and S3
✅ Complete storage schema for runs, evaluations, lineage
✅ Error handling strategy with retry and quarantine
✅ Credentials management with pluggable providers
✅ Transform engine using Arbiter's LLM client
✅ Quality gate semantics and implementation
✅ Airflow integration pattern
✅ Memory management with streaming
✅ Configuration system

**Next:** Begin Phase 1 implementation following the checklist.
