# Timeout Specifications

**Version:** 1.0.0
**Status:** Specification
**Created:** 2025-11-14

---

## Purpose

This document defines precise timeout values for all external operations in Loom pipelines. Timeouts prevent cascading failures, resource exhaustion, and unbounded waiting times in production deployments.

---

## Overview

Loom performs multiple types of I/O operations:
1. **LLM API calls** (via Arbiter) - Slowest, most variable
2. **Database queries** (Extract/Load) - Medium latency
3. **Evaluation calls** (via Arbiter) - Variable based on evaluator
4. **Network operations** (S3, external APIs) - Variable network conditions
5. **Cache operations** (Redis) - Should be fast

Each operation type requires appropriate timeout configuration to balance reliability and performance.

---

## Timeout Categories

### 1. LLM API Call Timeouts

**Context:** Generative AI API calls have high variability (100ms - 60s) depending on:
- Model size (gpt-4o vs gpt-4o-mini)
- Prompt length and complexity
- API load and rate limiting
- Network conditions

**Timeout Values:**

```yaml
llm_timeouts:
  # Per-model timeout configuration
  models:
    gpt-4o-mini:
      connect_timeout: 5s      # TCP connection establishment
      read_timeout: 30s        # Time to receive complete response
      total_timeout: 35s       # Connect + read combined

    gpt-4o:
      connect_timeout: 5s
      read_timeout: 60s        # Longer for more complex model
      total_timeout: 65s

    claude-3-5-sonnet:
      connect_timeout: 5s
      read_timeout: 45s
      total_timeout: 50s

  # Default for unknown models
  default:
    connect_timeout: 5s
    read_timeout: 30s
    total_timeout: 35s
```

**Rationale:**
- **Connect timeout (5s)**: Network connection should establish quickly; >5s indicates connectivity issues
- **Read timeout (30-60s)**: Model-dependent; larger models need more time
- **Total timeout**: Connect + read + buffer for retries

**Implementation:**

```python
# loom/engines/transform.py
import asyncio
from typing import Optional

class TransformEngine:
    def __init__(self, config: TransformConfig):
        self.config = config
        self.timeout_config = self._get_timeout_config(config.model)

    def _get_timeout_config(self, model: str) -> Dict[str, float]:
        """Get timeout configuration for model."""
        timeouts = {
            "gpt-4o-mini": {"connect": 5.0, "read": 30.0, "total": 35.0},
            "gpt-4o": {"connect": 5.0, "read": 60.0, "total": 65.0},
            "claude-3-5-sonnet": {"connect": 5.0, "read": 45.0, "total": 50.0},
        }
        return timeouts.get(model, {"connect": 5.0, "read": 30.0, "total": 35.0})

    async def _transform_record(
        self,
        record: Record,
        prompt: str
    ) -> TransformResult:
        """Transform with timeout protection."""
        try:
            # Apply total timeout to entire operation
            async with asyncio.timeout(self.timeout_config["total"]):
                response = await self.llm_client.generate(
                    prompt=prompt,
                    timeout=self.timeout_config["read"]
                )
                return TransformResult.from_response(response)

        except asyncio.TimeoutError:
            raise TransformError(
                f"LLM call exceeded timeout ({self.timeout_config['total']}s) "
                f"for model {self.config.model}"
            )
```

**Monitoring:**

```python
# Track timeout occurrences
llm_timeout_total = Counter(
    'loom_llm_timeout_total',
    'Total LLM API timeouts',
    ['pipeline_name', 'model', 'timeout_type']
)
```

---

### 2. Database Query Timeouts

**Context:** Database operations (PostgreSQL, SQLite) should be fast with proper indexes.

**Timeout Values:**

```yaml
database_timeouts:
  extract:
    connect_timeout: 3s       # Connection pool acquisition
    query_timeout: 10s        # Simple SELECT with WHERE
    complex_query_timeout: 30s # Joins, aggregations

  load:
    connect_timeout: 3s
    insert_timeout: 5s        # Per batch insert
    transaction_timeout: 60s  # Full transaction including retries

  storage:
    connect_timeout: 3s
    simple_query_timeout: 2s  # Metadata queries
    update_timeout: 5s        # Status updates
```

**Rationale:**
- **Connect timeout (3s)**: Connection pool should provide connections quickly
- **Query timeout (10s)**: Well-indexed queries should complete in seconds
- **Complex queries (30s)**: Joins and aggregations may need more time
- **Transaction timeout (60s)**: Allows for retries within transaction

**Implementation:**

```python
# loom/connectors/postgres.py
import asyncpg

class PostgresConnector(DataConnector):
    async def extract(
        self,
        query: str,
        batch_size: int = 100,
        cursor: Optional[str] = None
    ) -> ExtractResult:
        """Extract with timeout protection."""
        try:
            # Set statement timeout at connection level
            async with self.pool.acquire(timeout=3.0) as conn:
                await conn.execute(f"SET statement_timeout = '10s'")

                # Execute query with asyncio timeout as backup
                async with asyncio.timeout(12.0):  # Slightly higher than statement_timeout
                    rows = await conn.fetch(query)

                return self._to_extract_result(rows)

        except asyncio.TimeoutError:
            raise ExtractError(f"Query exceeded timeout (12s): {query[:100]}...")
        except asyncpg.QueryCanceledError:
            raise ExtractError(f"Query canceled by database timeout (10s): {query[:100]}...")
```

---

### 3. Evaluation Call Timeouts

**Context:** Arbiter evaluation calls use LLMs internally, so timeouts similar to transform.

**Timeout Values:**

```yaml
evaluation_timeouts:
  semantic_evaluator:
    timeout: 15s         # Semantic similarity is fast

  custom_criteria_evaluator:
    timeout: 30s         # LLM-based, needs more time

  factuality_evaluator:
    timeout: 45s         # May need retrieval + LLM

  default:
    timeout: 30s         # Conservative default
```

**Implementation:**

```python
# loom/engines/evaluate.py
class EvaluateEngine:
    EVALUATOR_TIMEOUTS = {
        "semantic": 15.0,
        "custom_criteria": 30.0,
        "factuality": 45.0,
    }

    async def evaluate(
        self,
        run_id: str,
        output: str,
        reference: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate with per-evaluator timeouts."""
        from arbiter import evaluate

        # Determine max timeout from configured evaluators
        max_timeout = max(
            self.EVALUATOR_TIMEOUTS.get(ev["type"], 30.0)
            for ev in self.config.evaluators
        )

        try:
            async with asyncio.timeout(max_timeout):
                result = await evaluate(
                    output=output,
                    reference=reference,
                    evaluators=[ev["type"] for ev in self.config.evaluators],
                    model=self.config.evaluation_model or "gpt-4o-mini"
                )
                return result

        except asyncio.TimeoutError:
            raise EvaluationError(
                f"Evaluation exceeded timeout ({max_timeout}s) "
                f"for evaluators: {[ev['type'] for ev in self.config.evaluators]}"
            )
```

---

### 4. Network Operation Timeouts

**Context:** S3, external APIs, and other network operations.

**Timeout Values:**

```yaml
network_timeouts:
  s3:
    connect_timeout: 5s
    read_timeout: 30s      # Per object read
    upload_timeout: 60s    # Per object upload
    list_timeout: 10s      # List objects operation

  http_api:
    connect_timeout: 5s
    read_timeout: 30s
    total_timeout: 35s

  redis_cache:
    connect_timeout: 1s    # Cache should be fast
    operation_timeout: 2s  # Get/set operations
```

**Implementation:**

```python
# loom/connectors/s3.py
import aioboto3
from botocore.config import Config

class S3Connector(DataConnector):
    async def connect(self) -> None:
        """Initialize S3 client with timeouts."""
        config = Config(
            connect_timeout=5,
            read_timeout=30,
            retries={'max_attempts': 3}
        )

        self.session = aioboto3.Session()
        self.s3_client = self.session.client('s3', config=config)

    async def _download_object(
        self,
        bucket: str,
        key: str
    ) -> bytes:
        """Download with timeout protection."""
        try:
            async with asyncio.timeout(35.0):  # Total timeout
                response = await self.s3_client.get_object(
                    Bucket=bucket,
                    Key=key
                )
                async with response['Body'] as stream:
                    return await stream.read()

        except asyncio.TimeoutError:
            raise ConnectorError(f"S3 download timeout (35s): s3://{bucket}/{key}")
```

---

### 5. Cache Operation Timeouts

**Context:** Cache operations should be extremely fast; slow cache defeats purpose.

**Timeout Values:**

```yaml
cache_timeouts:
  redis:
    connect_timeout: 1s     # Connection from pool
    get_timeout: 2s         # Cache read
    set_timeout: 2s         # Cache write
    delete_timeout: 2s      # Cache invalidation

  fallback_behavior:
    on_timeout: skip_cache  # Don't block pipeline on cache issues
```

**Implementation:**

```python
# loom/cache/redis.py
import aioredis
import asyncio

class RedisCache:
    async def get(self, key: str) -> Optional[Any]:
        """Get with timeout and fallback."""
        try:
            async with asyncio.timeout(2.0):
                value = await self.redis.get(key)
                return json.loads(value) if value else None

        except asyncio.TimeoutError:
            logger.warning(f"Cache GET timeout (2s) for key: {key[:50]}...")
            return None  # Cache miss, continue without cache
        except Exception as e:
            logger.error(f"Cache error: {e}")
            return None  # Graceful degradation

    async def set(self, key: str, value: Any, ttl_days: int = 7):
        """Set with timeout and fire-and-forget."""
        try:
            async with asyncio.timeout(2.0):
                await self.redis.setex(
                    key,
                    timedelta(days=ttl_days),
                    json.dumps(value)
                )
        except asyncio.TimeoutError:
            logger.warning(f"Cache SET timeout (2s) for key: {key[:50]}...")
            # Don't raise - cache write failure shouldn't block pipeline
        except Exception as e:
            logger.error(f"Cache write error: {e}")
            # Continue without cache
```

---

### 6. Pipeline-Level Timeouts

**Context:** Total pipeline execution time limits to prevent runaway processes.

**Timeout Values:**

```yaml
pipeline_timeouts:
  default_max_duration: 1h        # Most pipelines complete within 1 hour

  per_stage_timeouts:
    extract_stage: 15m            # Data extraction
    transform_stage: 45m          # Bulk of processing time
    evaluate_stage: 30m           # Evaluation parallelizable
    load_stage: 15m               # Data loading

  record_processing:
    per_record_timeout: 2m        # Single record through all stages
    batch_timeout: 30m            # Batch of records
```

**Implementation:**

```python
# loom/engines/pipeline.py
class PipelineExecutor:
    async def execute(
        self,
        pipeline: Pipeline,
        params: Dict[str, Any]
    ) -> PipelineResult:
        """Execute with stage-level timeouts."""
        run_id = self._generate_run_id()

        try:
            # Overall pipeline timeout
            async with asyncio.timeout(3600.0):  # 1 hour

                # Extract stage with timeout
                async with asyncio.timeout(900.0):  # 15 minutes
                    records = await self._execute_extract(run_id, pipeline, params)

                # Transform stage with timeout
                async with asyncio.timeout(2700.0):  # 45 minutes
                    transformed = await self._execute_transform(run_id, records)

                # Evaluate stage with timeout
                async with asyncio.timeout(1800.0):  # 30 minutes
                    validated = await self._execute_evaluate(run_id, transformed)

                # Load stage with timeout
                async with asyncio.timeout(900.0):  # 15 minutes
                    loaded = await self._execute_load(run_id, validated)

                return self._finalize_run(run_id, "success", ...)

        except asyncio.TimeoutError as e:
            # Determine which stage timed out
            stage = self._get_current_stage(run_id)
            await self._finalize_run(
                run_id,
                status="failed",
                error=f"Pipeline timeout: {stage} stage exceeded limit"
            )
            raise PipelineTimeoutError(f"Pipeline {pipeline.name} timed out in {stage} stage")
```

---

## Timeout Configuration

### Global Configuration

```yaml
# loom.config.yaml
timeouts:
  # LLM timeouts by model
  llm:
    gpt-4o-mini: {connect: 5s, read: 30s, total: 35s}
    gpt-4o: {connect: 5s, read: 60s, total: 65s}
    default: {connect: 5s, read: 30s, total: 35s}

  # Database timeouts
  database:
    connect: 3s
    query: 10s
    transaction: 60s

  # Network timeouts
  network:
    s3: {connect: 5s, read: 30s, upload: 60s}
    http: {connect: 5s, read: 30s, total: 35s}

  # Cache timeouts
  cache:
    redis: {connect: 1s, operation: 2s}

  # Pipeline timeouts
  pipeline:
    max_duration: 1h
    extract_stage: 15m
    transform_stage: 45m
    evaluate_stage: 30m
    load_stage: 15m
```

### Per-Pipeline Overrides

```yaml
# pipelines/customer_sentiment.yaml
name: customer_sentiment

timeouts:
  # Override defaults for this pipeline
  transform_stage: 30m    # Smaller dataset, faster processing
  llm_timeout: 20s        # Using fast model

transform:
  model: gpt-4o-mini
  batch_size: 100
```

---

## Error Handling

### Timeout Error Types

```python
# loom/errors.py
class TimeoutError(LoomError):
    """Base timeout error."""
    pass

class LLMTimeoutError(TimeoutError):
    """LLM API call timeout."""
    pass

class DatabaseTimeoutError(TimeoutError):
    """Database query timeout."""
    pass

class PipelineTimeoutError(TimeoutError):
    """Overall pipeline timeout."""
    pass
```

### Timeout Handling Strategy

```python
# loom/retry.py
from dataclasses import dataclass

@dataclass
class TimeoutConfig:
    """Timeout configuration with retry behavior."""
    operation_timeout: float
    retry_on_timeout: bool = True
    max_retries: int = 3
    backoff_multiplier: float = 2.0

    def get_retry_timeout(self, attempt: int) -> float:
        """Get timeout for retry attempt."""
        if not self.retry_on_timeout:
            return self.operation_timeout

        # Increase timeout on retries (2x, 3x, 4x)
        return self.operation_timeout * (1 + attempt)

async def retry_with_timeout(
    func: Callable,
    config: TimeoutConfig,
    *args,
    **kwargs
) -> Any:
    """Execute function with timeout and retry."""
    for attempt in range(config.max_retries):
        timeout = config.get_retry_timeout(attempt)

        try:
            async with asyncio.timeout(timeout):
                return await func(*args, **kwargs)

        except asyncio.TimeoutError:
            if attempt == config.max_retries - 1:
                raise  # Final attempt, propagate timeout

            logger.warning(
                f"Timeout on attempt {attempt + 1}/{config.max_retries}, "
                f"retrying with {timeout * 2}s timeout..."
            )
            await asyncio.sleep(config.backoff_multiplier ** attempt)
```

---

## Monitoring and Alerts

### Timeout Metrics

```python
# Prometheus metrics
timeout_total = Counter(
    'loom_timeout_total',
    'Total timeout occurrences',
    ['pipeline_name', 'stage', 'operation_type']
)

timeout_duration_seconds = Histogram(
    'loom_timeout_duration_seconds',
    'Duration before timeout',
    ['pipeline_name', 'operation_type'],
    buckets=[1, 5, 10, 30, 60, 120, 300]  # 1s, 5s, 10s, 30s, 1m, 2m, 5m
)
```

### Alert Rules

```yaml
# alerting_rules.yml
groups:
- name: loom_timeout_alerts
  interval: 30s
  rules:

  # High LLM timeout rate
  - alert: HighLLMTimeoutRate
    expr: |
      rate(loom_timeout_total{operation_type="llm"}[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High LLM timeout rate"
      description: "Pipeline {{ $labels.pipeline_name }} experiencing >10% LLM timeouts"

  # Pipeline stage timeout
  - alert: PipelineStageTimeout
    expr: |
      increase(loom_timeout_total{stage=~"extract|transform|evaluate|load"}[5m]) > 0
    labels:
      severity: critical
    annotations:
      summary: "Pipeline stage timeout"
      description: "Stage {{ $labels.stage }} timed out for pipeline {{ $labels.pipeline_name }}"

  # Database query timeouts
  - alert: HighDatabaseTimeoutRate
    expr: |
      rate(loom_timeout_total{operation_type="database"}[10m]) > 0.05
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "High database timeout rate"
      description: "Database operations timing out at >5% rate"
```

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/test_timeouts.py
import pytest
import asyncio

@pytest.mark.asyncio
async def test_llm_timeout_enforced():
    """LLM call timeout is enforced."""
    async def slow_llm_call():
        await asyncio.sleep(40)  # Exceeds 30s timeout
        return "result"

    with pytest.raises(asyncio.TimeoutError):
        async with asyncio.timeout(30.0):
            await slow_llm_call()

@pytest.mark.asyncio
async def test_timeout_retry_increases():
    """Timeout increases on retry attempts."""
    config = TimeoutConfig(
        operation_timeout=10.0,
        retry_on_timeout=True,
        max_retries=3
    )

    assert config.get_retry_timeout(0) == 10.0  # First attempt
    assert config.get_retry_timeout(1) == 20.0  # Second attempt (2x)
    assert config.get_retry_timeout(2) == 30.0  # Third attempt (3x)
```

### Integration Tests

```python
# tests/integration/test_pipeline_timeouts.py
@pytest.mark.asyncio
async def test_pipeline_stage_timeout():
    """Pipeline stage timeout prevents runaway execution."""
    loom = Loom()

    # Create pipeline with very short timeout
    pipeline = Pipeline(
        name="test_timeout",
        timeouts={"transform_stage": "5s"}
    )

    # Transform that takes >5s should timeout
    with pytest.raises(PipelineTimeoutError) as exc_info:
        await loom.run_pipeline(pipeline)

    assert "transform_stage" in str(exc_info.value)
```

---

## Production Runbook

### Timeout Investigation Procedure

**1. Identify Timeout Pattern**
```sql
-- Check timeout distribution by operation type
SELECT
  operation_type,
  COUNT(*) as timeout_count,
  AVG(duration_seconds) as avg_duration
FROM pipeline_errors
WHERE error_type = 'TimeoutError'
  AND occurred_at > NOW() - INTERVAL '1 hour'
GROUP BY operation_type
ORDER BY timeout_count DESC;
```

**2. Analyze Specific Timeouts**
```bash
# Check Prometheus for timeout trends
curl -G http://localhost:9090/api/v1/query \
  --data-urlencode 'query=rate(loom_timeout_total[5m])'

# Check recent timeout logs
loom logs --filter "timeout" --last 1h
```

**3. Adjust Timeouts if Needed**
```yaml
# If legitimate operations timing out, increase timeouts
# loom.config.yaml
timeouts:
  llm:
    gpt-4o:
      read_timeout: 90s  # Increased from 60s
```

---

## Conclusion

This specification provides **concrete timeout values** for all Loom operations:

1. ✅ **LLM API calls**: 30-60s depending on model complexity
2. ✅ **Database queries**: 3-30s with statement-level enforcement
3. ✅ **Evaluation calls**: 15-45s based on evaluator type
4. ✅ **Network operations**: 5-60s for S3 and HTTP
5. ✅ **Cache operations**: 1-2s with graceful fallback
6. ✅ **Pipeline stages**: 15-45 minutes per stage, 1 hour total

**Key Principles:**
- **Fail fast**: Short timeouts for operations that should be quick
- **Graceful degradation**: Cache timeouts don't block pipeline
- **Progressive retry**: Increase timeout on retry attempts
- **Stage isolation**: Timeout one stage without killing entire pipeline

**Next Steps:**
1. Implement timeout configuration system
2. Add timeout parameters to all async operations
3. Set up monitoring and alerting for timeout rates
4. Document timeout tuning guidelines for operators
