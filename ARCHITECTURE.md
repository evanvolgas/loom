# Loom Architecture

**Version:** 1.0.0
**Status:** Design Phase
**Last Updated:** 2025-11-14

This document provides comprehensive architectural design for the Loom orchestration framework. It covers system architecture, component design, API specifications, database design, and deployment patterns.

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [API Design](#api-design)
3. [Core Component Architecture](#core-component-architecture)
4. [Database & Storage Architecture](#database--storage-architecture)
5. [Security Architecture](#security-architecture)
6. [Deployment Architecture](#deployment-architecture)
7. [Integration Patterns](#integration-patterns)
8. [Monitoring & Observability](#monitoring--observability)

---

## System Architecture Overview

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Loom Framework                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐      ┌──────────────┐      ┌──────────────┐  │
│  │              │      │              │      │              │  │
│  │  CLI Layer   │──────│  API Layer   │──────│  SDK Layer   │  │
│  │              │      │              │      │              │  │
│  └──────┬───────┘      └──────┬───────┘      └──────┬───────┘  │
│         │                     │                     │           │
│         └─────────────────────┼─────────────────────┘           │
│                               │                                 │
│                    ┌──────────▼──────────┐                      │
│                    │                     │                      │
│                    │  Pipeline Executor  │                      │
│                    │                     │                      │
│                    └──────────┬──────────┘                      │
│                               │                                 │
│         ┌─────────────────────┼─────────────────────┐           │
│         │                     │                     │           │
│  ┌──────▼──────┐      ┌──────▼──────┐      ┌──────▼──────┐    │
│  │             │      │             │      │             │    │
│  │  Extract    │      │  Transform  │      │  Evaluate   │    │
│  │  Engine     │      │  Engine     │      │  Engine     │    │
│  │             │      │             │      │             │    │
│  └──────┬──────┘      └──────┬──────┘      └──────┬──────┘    │
│         │                     │                     │           │
│         │                     │                     │           │
│  ┌──────▼──────┐      ┌──────▼──────┐      ┌──────▼──────┐    │
│  │             │      │             │      │             │    │
│  │ Connectors  │      │   Arbiter   │      │  Quality    │    │
│  │  (Source)   │      │   Client    │      │   Gates     │    │
│  │             │      │             │      │             │    │
│  └──────┬──────┘      └─────────────┘      └──────┬──────┘    │
│         │                                          │           │
│         │                                          │           │
│  ┌──────▼──────────────────────────────────────────▼──────┐   │
│  │                                                          │   │
│  │                    Load Engine                           │   │
│  │                                                          │   │
│  └──────┬───────────────────────────────────────────────────┘   │
│         │                                                       │
│  ┌──────▼──────┐                                               │
│  │             │                                               │
│  │ Connectors  │                                               │
│  │ (Dest)      │                                               │
│  │             │                                               │
│  └─────────────┘                                               │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Cross-Cutting Concerns                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │          │  │          │  │          │  │          │      │
│  │  Cache   │  │  Config  │  │  Metrics │  │ Lineage  │      │
│  │          │  │          │  │          │  │          │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
│                                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐      │
│  │          │  │          │  │          │  │          │      │
│  │  Error   │  │  Cost    │  │  Retry   │  │  State   │      │
│  │ Recovery │  │ Tracking │  │  Logic   │  │ Manager  │      │
│  │          │  │          │  │          │  │          │      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                               │
                               │
                  ┌────────────▼────────────┐
                  │                         │
                  │  Storage Layer          │
                  │  (PostgreSQL/SQLite)    │
                  │                         │
                  └─────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Dependencies |
|-----------|---------------|--------------|
| **CLI Layer** | Command-line interface, user interaction | API Layer |
| **API Layer** | Python API, programmatic access | Pipeline Executor |
| **SDK Layer** | High-level abstractions, helper functions | API Layer |
| **Pipeline Executor** | Orchestrate AETL flow, manage execution | All engines |
| **Extract Engine** | Pull data from sources via connectors | Source Connectors |
| **Transform Engine** | Apply AI transformations via Arbiter | Arbiter Client |
| **Evaluate Engine** | Run evaluations, enforce quality gates | Arbiter, Quality Gates |
| **Load Engine** | Write validated data to destinations | Destination Connectors |
| **Connectors** | Abstract data sources/destinations | External systems |
| **Quality Gates** | Evaluation logic, pass/fail decisions | Evaluate Engine |
| **Cache Manager** | Cache LLM responses, avoid redundant calls | Storage Layer |
| **Config Manager** | Load and validate configurations | File system |
| **Metrics Collector** | Track performance, cost, quality metrics | Storage Layer |
| **Lineage Tracker** | Record data lineage and provenance | Storage Layer |
| **Error Recovery** | Handle failures, retries, quarantine | Storage Layer |
| **Cost Tracker** | Monitor token usage and costs | Metrics Collector |
| **Retry Logic** | Exponential backoff, failure handling | Error Recovery |
| **State Manager** | Track pipeline execution state | Storage Layer |

### Data Flow

```
User Input (YAML Pipeline)
         │
         ▼
    Config Parse
         │
         ▼
  Pipeline Executor
         │
         ├──────────────────────────────────────┐
         │                                      │
         ▼                                      ▼
    Extract Phase                         State Tracking
    │                                     │
    ├─ Source Connector                   ├─ Update: EXTRACTING
    ├─ Batch Iterator                     ├─ Record count
    ├─ Schema Validation                  └─ Timestamp
    │
    ▼
    Transform Phase
    │
    ├─ Load Context                       ├─ Update: TRANSFORMING
    ├─ Render Prompts                     ├─ Token usage
    ├─ Call Arbiter LLM                   ├─ Cost calculation
    ├─ Cache Check/Store                  └─ Latency tracking
    │
    ▼
    Evaluate Phase
    │
    ├─ Run Evaluators                     ├─ Update: EVALUATING
    ├─ Collect Scores                     ├─ Evaluation scores
    ├─ Apply Quality Gate                 └─ Pass/fail status
    │
    ├─────────┬─────────┐
    │         │         │
    PASS      FAIL      QUARANTINE
    │         │         │
    ▼         ▼         ▼
    Load     Reject   Store in
    Phase    Record   Quarantine
    │                 Table
    ├─ Destination Connector
    ├─ Batch Write               ├─ Update: LOADING
    ├─ Transaction Control        ├─ Records loaded
    ├─ Lineage Recording          └─ Final metrics
    │
    ▼
    Complete
    │
    ├─ Final State: SUCCESS/FAILED
    ├─ Write Lineage
    ├─ Calculate Total Cost
    └─ Generate Report
```

---

## API Design

### CLI Interface

#### Pipeline Management

```bash
# Run a pipeline
loom run <pipeline_name> [options]
  --config PATH          Path to config file (default: loom.yaml)
  --param KEY=VALUE      Override pipeline parameters
  --dry-run              Validate without executing
  --verbose              Detailed output
  --no-cache             Disable caching

# List pipelines
loom list [options]
  --format table|json    Output format

# Validate pipeline
loom validate <pipeline_file>

# Show pipeline status
loom status <run_id>

# Show pipeline history
loom history <pipeline_name> [options]
  --limit N              Show last N runs
  --format table|json

# Test pipeline
loom test <pipeline_name> [options]
  --sample N             Use N sample records

# Generate pipeline template
loom init <pipeline_name> [options]
  --template basic|rag|sentiment

# Show pipeline lineage
loom lineage <pipeline_name> [options]
  --run-id RUN_ID       Specific run
  --graph                Show graph visualization
```

#### Configuration Management

```bash
# Initialize project
loom config init

# Show configuration
loom config show

# Validate configuration
loom config validate

# Set configuration value
loom config set <key> <value>
```

#### Monitoring & Debugging

```bash
# Show metrics
loom metrics <pipeline_name> [options]
  --run-id RUN_ID
  --format table|json

# Show cost breakdown
loom cost <pipeline_name> [options]
  --run-id RUN_ID
  --aggregate-by day|week|month

# Show quarantined records
loom quarantine list [options]
  --pipeline PIPELINE_NAME
  --limit N

# Retry quarantined records
loom quarantine retry <run_id> [options]
  --record-ids ID1,ID2,ID3
```

### Python API

#### Core API

```python
from loom import Loom, Pipeline, PipelineConfig

# Initialize Loom
loom = Loom(config_path="loom.yaml")

# Load pipeline from YAML
pipeline = loom.load_pipeline("customer_sentiment.yaml")

# Or create programmatically
pipeline = Pipeline(
    name="customer_sentiment",
    version="2.1.0",
    extract=ExtractConfig(
        source="postgres://customers/reviews",
        query="SELECT * FROM reviews WHERE created_at > {execution_date}"
    ),
    transform=TransformConfig(
        type="ai",
        prompt_path="prompts/classify_sentiment.txt",
        model="gpt-4o-mini",
        temperature=0.0,
        batch_size=50
    ),
    evaluate=EvaluateConfig(
        evaluators=[
            {"type": "semantic", "threshold": 0.8},
            {"type": "custom_criteria", "criteria": "No hallucination", "threshold": 0.75}
        ],
        quality_gate="all_pass"
    ),
    load=LoadConfig(
        destination="postgres://analytics/sentiment_scores"
    )
)

# Run pipeline
result = await loom.run_pipeline(
    pipeline=pipeline,
    params={"execution_date": "2024-01-01"}
)

print(f"Run ID: {result.run_id}")
print(f"Records processed: {result.records_processed}")
print(f"Records validated: {result.records_validated}")
print(f"Cost: ${result.cost_total:.4f}")
```

#### Advanced API Usage

```python
# Custom connector
from loom.connectors import DataConnector

class MyCustomConnector(DataConnector):
    async def extract(self, query: str, batch_size: int) -> ExtractResult:
        # Custom extraction logic
        pass

    async def load(self, records: List[Record], destination: str) -> LoadResult:
        # Custom loading logic
        pass

# Register custom connector
loom.register_connector("mycustom", MyCustomConnector)

# Use in pipeline
pipeline = Pipeline(
    extract=ExtractConfig(
        source="mycustom://my-data-source"
    )
)

# Custom quality gate
from loom.quality_gates import QualityGate

class CustomQualityGate(QualityGate):
    def apply(self, scores: List[Score]) -> bool:
        # Custom gate logic
        return all(s.value > 0.7 for s in scores if s.name == "critical")

# Register and use
loom.register_quality_gate("custom_gate", CustomQualityGate)
```

#### Streaming API

```python
# Stream records through pipeline
async for record in loom.stream_pipeline(pipeline):
    print(f"Record: {record.id}, Status: {record.status}")
    if record.status == "validated":
        print(f"  Score: {record.evaluation_score}")
    elif record.status == "quarantined":
        print(f"  Reason: {record.failure_reason}")

# Process with custom handler
async def handle_record(record: Record) -> None:
    if record.status == "validated":
        await send_to_destination(record)
    elif record.status == "quarantined":
        await notify_admin(record)

await loom.stream_pipeline(
    pipeline=pipeline,
    handler=handle_record,
    max_concurrent=50
)
```

#### Monitoring API

```python
# Get pipeline status
status = await loom.get_status(run_id="run_abc123")

# Get metrics
metrics = await loom.get_metrics(
    pipeline_name="customer_sentiment",
    run_id="run_abc123"
)

print(f"Latency: {metrics.avg_latency_ms}ms")
print(f"Cost: ${metrics.total_cost:.4f}")
print(f"Quality Score: {metrics.avg_evaluation_score:.2f}")

# Get lineage
lineage = await loom.get_lineage(
    pipeline_name="customer_sentiment",
    run_id="run_abc123"
)

for node in lineage.nodes:
    print(f"{node.stage}: {node.records_in} → {node.records_out}")

# Query quarantined records
quarantined = await loom.get_quarantined_records(
    pipeline_name="customer_sentiment",
    limit=10
)

for record in quarantined:
    print(f"ID: {record.id}, Reason: {record.failure_reason}")
```

### REST API (Future)

Future REST API for web integration:

```
GET    /api/v1/pipelines                    # List pipelines
GET    /api/v1/pipelines/{name}             # Get pipeline details
POST   /api/v1/pipelines/{name}/run         # Run pipeline
GET    /api/v1/runs/{run_id}                # Get run status
GET    /api/v1/runs/{run_id}/metrics        # Get run metrics
GET    /api/v1/runs/{run_id}/lineage        # Get run lineage
GET    /api/v1/quarantine                   # List quarantined records
POST   /api/v1/quarantine/{record_id}/retry # Retry quarantined record
```

---

## Core Component Architecture

### Pipeline Executor

**Responsibility:** Orchestrate the Extract → Transform → Evaluate → Load flow.

```python
class PipelineExecutor:
    """Orchestrates pipeline execution with state management."""

    def __init__(
        self,
        storage: StorageBackend,
        cache: CacheManager,
        metrics: MetricsCollector,
        lineage: LineageTracker,
        error_recovery: RecoveryManager
    ):
        self.storage = storage
        self.cache = cache
        self.metrics = metrics
        self.lineage = lineage
        self.error_recovery = error_recovery

        # Engines
        self.extract_engine: Optional[ExtractEngine] = None
        self.transform_engine: Optional[TransformEngine] = None
        self.evaluate_engine: Optional[EvaluateEngine] = None
        self.load_engine: Optional[LoadEngine] = None

    async def execute(
        self,
        pipeline: Pipeline,
        params: Dict[str, Any]
    ) -> PipelineResult:
        """Execute pipeline with full orchestration."""

        # Initialize run
        run_id = self._generate_run_id()
        await self._initialize_run(run_id, pipeline)

        try:
            # Initialize engines
            await self._initialize_engines(pipeline)

            # Execute AETL flow
            records_extracted = await self._execute_extract(run_id, pipeline, params)
            records_transformed = await self._execute_transform(run_id, records_extracted)
            records_validated = await self._execute_evaluate(run_id, records_transformed)
            records_loaded = await self._execute_load(run_id, records_validated)

            # Finalize run
            result = await self._finalize_run(
                run_id=run_id,
                status="success",
                records_processed=len(records_extracted),
                records_validated=len(records_validated),
                records_loaded=len(records_loaded)
            )

            return result

        except Exception as e:
            # Handle failure
            await self._finalize_run(
                run_id=run_id,
                status="failed",
                error=str(e)
            )
            raise

    async def _execute_extract(
        self,
        run_id: str,
        pipeline: Pipeline,
        params: Dict[str, Any]
    ) -> List[Record]:
        """Execute extract phase."""

        await self._update_state(run_id, PipelineState.EXTRACTING)

        records = []
        async for batch in self.extract_engine.extract_batches(
            source=pipeline.extract.source,
            query=pipeline.extract.query,
            batch_size=pipeline.extract.batch_size,
            params=params
        ):
            records.extend(batch)
            await self.metrics.record_extracted(run_id, len(batch))

        await self.lineage.record_extract(
            run_id=run_id,
            source=pipeline.extract.source,
            records_count=len(records)
        )

        return records

    async def _execute_transform(
        self,
        run_id: str,
        records: List[Record]
    ) -> List[TransformResult]:
        """Execute transform phase with AI."""

        await self._update_state(run_id, PipelineState.TRANSFORMING)

        results = []
        for batch in self._batch_records(records):
            try:
                batch_results = await self.transform_engine.transform_batch(
                    run_id=run_id,
                    records=batch
                )
                results.extend(batch_results)

                # Track metrics
                for result in batch_results:
                    await self.metrics.record_transform(
                        run_id=run_id,
                        tokens=result.tokens_used,
                        cost=result.cost,
                        latency_ms=result.latency_ms
                    )

            except Exception as e:
                # Handle batch failure
                for record in batch:
                    recovery_action = await self.error_recovery.handle_transform_error(
                        run_id=run_id,
                        record=record,
                        error=e
                    )

                    if recovery_action == RecoveryAction.QUARANTINE:
                        # Skip this batch
                        continue
                    elif recovery_action == RecoveryAction.RETRY:
                        # Retry with exponential backoff
                        await asyncio.sleep(1)
                        # ... retry logic

        await self.lineage.record_transform(
            run_id=run_id,
            records_in=len(records),
            records_out=len(results)
        )

        return results

    async def _execute_evaluate(
        self,
        run_id: str,
        transform_results: List[TransformResult]
    ) -> List[Record]:
        """Execute evaluate phase with quality gates."""

        await self._update_state(run_id, PipelineState.EVALUATING)

        validated_records = []
        rejected_records = []

        for result in transform_results:
            try:
                # Run evaluators
                evaluation = await self.evaluate_engine.evaluate(
                    run_id=run_id,
                    output=result.output,
                    reference=result.reference
                )

                # Apply quality gate
                passed = self.evaluate_engine.apply_quality_gate(
                    evaluation=evaluation
                )

                if passed:
                    validated_records.append(result.to_record(evaluation))
                else:
                    rejected_records.append(result.to_record(evaluation))
                    await self.error_recovery.handle_evaluation_failure(
                        run_id=run_id,
                        record=result.record,
                        evaluation=evaluation
                    )

                # Track metrics
                await self.metrics.record_evaluation(
                    run_id=run_id,
                    score=evaluation.overall_score,
                    passed=passed
                )

            except Exception as e:
                # Handle evaluation error
                await self.error_recovery.handle_evaluation_error(
                    run_id=run_id,
                    record=result.record,
                    error=e
                )

        await self.lineage.record_evaluate(
            run_id=run_id,
            records_in=len(transform_results),
            records_validated=len(validated_records),
            records_rejected=len(rejected_records)
        )

        return validated_records

    async def _execute_load(
        self,
        run_id: str,
        validated_records: List[Record]
    ) -> int:
        """Execute load phase to destination."""

        await self._update_state(run_id, PipelineState.LOADING)

        records_loaded = 0

        for batch in self._batch_records(validated_records):
            try:
                result = await self.load_engine.load_batch(
                    run_id=run_id,
                    records=batch
                )

                records_loaded += result.records_loaded

                await self.metrics.record_loaded(
                    run_id=run_id,
                    count=result.records_loaded
                )

            except Exception as e:
                # Handle load error
                await self.error_recovery.handle_load_error(
                    run_id=run_id,
                    records=batch,
                    error=e
                )

        await self.lineage.record_load(
            run_id=run_id,
            records_loaded=records_loaded
        )

        return records_loaded
```

### Extract Engine

**Responsibility:** Pull data from sources via connector abstraction.

```python
class ExtractEngine:
    """Extract data from sources using connectors."""

    def __init__(self, connector_registry: ConnectorRegistry):
        self.connectors = connector_registry

    async def extract_batches(
        self,
        source: str,
        query: str,
        batch_size: int,
        params: Dict[str, Any]
    ) -> AsyncIterator[List[Record]]:
        """Extract records in batches."""

        # Parse source URI (e.g., "postgres://customers/reviews")
        connector_type, connection_string = self._parse_source(source)

        # Get connector
        connector = self.connectors.get(connector_type)

        # Initialize connection
        await connector.connect(connection_string)

        try:
            # Render query with parameters
            rendered_query = self._render_query(query, params)

            # Extract in batches
            cursor = None
            while True:
                result = await connector.extract(
                    query=rendered_query,
                    batch_size=batch_size,
                    cursor=cursor
                )

                if not result.records:
                    break

                yield result.records

                cursor = result.next_cursor
                if cursor is None:
                    break

        finally:
            await connector.disconnect()

    def _parse_source(self, source: str) -> Tuple[str, str]:
        """Parse source URI into connector type and connection string."""
        # postgres://customers/reviews -> ("postgres", "customers/reviews")
        parts = source.split("://", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""

    def _render_query(self, query: str, params: Dict[str, Any]) -> str:
        """Render query template with parameters."""
        # Support Jinja2 templates
        from jinja2 import Template
        template = Template(query)
        return template.render(**params)
```

### Transform Engine

**Responsibility:** Apply AI transformations using Arbiter's LLM client.

```python
class TransformEngine:
    """Transform records using AI via Arbiter."""

    def __init__(
        self,
        config: TransformConfig,
        cache: CacheManager,
        cost_tracker: CostTracker
    ):
        self.config = config
        self.cache = cache
        self.cost_tracker = cost_tracker
        self.llm_client: Optional[Any] = None

    async def initialize(self):
        """Initialize LLM client from Arbiter."""
        from arbiter.core.llm_client import LLMManager

        self.llm_client = await LLMManager.get_client(
            model=self.config.model,
            temperature=self.config.temperature
        )

    async def transform_batch(
        self,
        run_id: str,
        records: List[Record]
    ) -> List[TransformResult]:
        """Transform a batch of records."""

        results = []

        for record in records:
            # Check cache
            cache_key = self._compute_cache_key(record)
            cached_result = await self.cache.get(cache_key)

            if cached_result:
                results.append(cached_result)
                continue

            # Load context
            context = await self._load_context(record)

            # Render prompt
            prompt = await self._render_prompt(record, context)

            # Call LLM
            start_time = time.time()

            response = await self.llm_client.generate(
                prompt=prompt,
                **self.config.llm_params
            )

            latency_ms = (time.time() - start_time) * 1000

            # Calculate cost
            cost = self.cost_tracker.calculate_cost(
                model=self.config.model,
                tokens_used=response.tokens_used
            )

            # Create result
            result = TransformResult(
                record=record,
                output=response.text,
                tokens_used=response.tokens_used,
                cost=cost,
                latency_ms=latency_ms,
                model=self.config.model,
                prompt_version=self.config.prompt_version
            )

            # Cache result
            await self.cache.set(cache_key, result)

            results.append(result)

        return results

    def _compute_cache_key(self, record: Record) -> str:
        """Compute cache key from record content + model + prompt."""
        content_hash = hashlib.sha256(
            record.content.encode()
        ).hexdigest()

        return f"{content_hash}:{self.config.model}:{self.config.prompt_version}"

    async def _load_context(self, record: Record) -> Dict[str, str]:
        """Load context files for prompt rendering."""
        context = {}

        for context_path in self.config.context_paths:
            with open(context_path, 'r') as f:
                context[context_path] = f.read()

        return context

    async def _render_prompt(
        self,
        record: Record,
        context: Dict[str, str]
    ) -> str:
        """Render prompt template with record and context."""
        from jinja2 import Template

        # Load prompt template
        with open(self.config.prompt_path, 'r') as f:
            template_text = f.read()

        template = Template(template_text)

        return template.render(
            record=record,
            context=context,
            **self.config.template_vars
        )
```

### Evaluate Engine

**Responsibility:** Run evaluations using Arbiter and apply quality gates.

```python
class EvaluateEngine:
    """Evaluate transformed outputs using Arbiter."""

    def __init__(
        self,
        config: EvaluateConfig,
        quality_gate_registry: QualityGateRegistry
    ):
        self.config = config
        self.quality_gates = quality_gate_registry

    async def evaluate(
        self,
        run_id: str,
        output: str,
        reference: Optional[str] = None
    ) -> EvaluationResult:
        """Evaluate output using configured evaluators."""

        from arbiter import evaluate

        # Extract evaluator types
        evaluator_types = [
            ev["type"] for ev in self.config.evaluators
        ]

        # Run evaluation
        result = await evaluate(
            output=output,
            reference=reference,
            evaluators=evaluator_types,
            model=self.config.evaluation_model or "gpt-4o-mini"
        )

        return result

    def apply_quality_gate(
        self,
        evaluation: EvaluationResult
    ) -> bool:
        """Apply quality gate to evaluation result."""

        gate_type = self.config.quality_gate
        gate = self.quality_gates.get(gate_type)

        return gate.apply(
            scores=evaluation.scores,
            thresholds=self._extract_thresholds()
        )

    def _extract_thresholds(self) -> Dict[str, float]:
        """Extract thresholds from evaluator configs."""
        thresholds = {}

        for ev_config in self.config.evaluators:
            if "threshold" in ev_config:
                thresholds[ev_config["type"]] = ev_config["threshold"]

        return thresholds
```

### Load Engine

**Responsibility:** Write validated records to destinations via connectors.

```python
class LoadEngine:
    """Load validated records to destinations."""

    def __init__(self, connector_registry: ConnectorRegistry):
        self.connectors = connector_registry

    async def load_batch(
        self,
        run_id: str,
        records: List[Record]
    ) -> LoadResult:
        """Load a batch of records to destination."""

        # Parse destination
        connector_type, connection_string = self._parse_destination(
            self.config.destination
        )

        # Get connector
        connector = self.connectors.get(connector_type)

        # Connect
        await connector.connect(connection_string)

        try:
            # Load records
            result = await connector.load(
                records=records,
                destination=connection_string
            )

            return result

        finally:
            await connector.disconnect()
```

### Quality Gate Implementations

```python
class QualityGate(ABC):
    """Base class for quality gates."""

    @abstractmethod
    def apply(
        self,
        scores: List[Score],
        thresholds: Dict[str, float]
    ) -> bool:
        """Apply gate logic."""
        pass


class AllPassQualityGate(QualityGate):
    """All evaluators must pass their thresholds."""

    def apply(
        self,
        scores: List[Score],
        thresholds: Dict[str, float]
    ) -> bool:
        for score in scores:
            threshold = thresholds.get(score.name, 0.0)
            if score.value < threshold:
                return False
        return True


class MajorityPassQualityGate(QualityGate):
    """Majority of evaluators must pass."""

    def apply(
        self,
        scores: List[Score],
        thresholds: Dict[str, float]
    ) -> bool:
        passed = sum(
            1 for score in scores
            if score.value >= thresholds.get(score.name, 0.0)
        )
        return passed > len(scores) / 2


class WeightedQualityGate(QualityGate):
    """Weighted average must exceed threshold."""

    def __init__(self, weights: Dict[str, float], threshold: float):
        self.weights = weights
        self.threshold = threshold

    def apply(
        self,
        scores: List[Score],
        thresholds: Dict[str, float]
    ) -> bool:
        weighted_sum = sum(
            score.value * self.weights.get(score.name, 1.0)
            for score in scores
        )
        total_weight = sum(
            self.weights.get(score.name, 1.0)
            for score in scores
        )

        weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0.0

        return weighted_avg >= self.threshold
```

---

## Database & Storage Architecture

### Storage Schema

See IMPLEMENTATION_SPEC.md for complete SQL schema. Key tables:

1. **pipeline_runs** - Pipeline execution tracking
2. **evaluation_results** - Aggregated evaluation results per run
3. **record_evaluations** - Individual record evaluations
4. **quarantined_records** - Failed records for review
5. **pipeline_lineage** - Data lineage tracking
6. **cost_tracking** - Token usage and cost metrics
7. **pipeline_metrics** - Performance metrics
8. **pipeline_state** - Execution state tracking
9. **cache_entries** - LLM response caching
10. **retry_log** - Retry attempt tracking

### Storage Backend Interface

```python
class StorageBackend(ABC):
    """Abstract storage backend."""

    @abstractmethod
    async def initialize(self):
        """Initialize storage (create tables, etc.)."""
        pass

    @abstractmethod
    async def create_run(self, run: PipelineRun) -> str:
        """Create new pipeline run record."""
        pass

    @abstractmethod
    async def update_run(self, run_id: str, updates: Dict[str, Any]):
        """Update pipeline run."""
        pass

    @abstractmethod
    async def get_run(self, run_id: str) -> PipelineRun:
        """Get pipeline run by ID."""
        pass

    @abstractmethod
    async def save_evaluation(self, evaluation: EvaluationRecord):
        """Save evaluation result."""
        pass

    @abstractmethod
    async def quarantine_record(self, record: QuarantineRecord):
        """Quarantine failed record."""
        pass

    @abstractmethod
    async def record_lineage(self, lineage: LineageRecord):
        """Record lineage information."""
        pass


class PostgreSQLStorage(StorageBackend):
    """PostgreSQL storage implementation."""

    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool: Optional[asyncpg.Pool] = None

    async def initialize(self):
        """Create connection pool and tables."""
        self.pool = await asyncpg.create_pool(self.connection_string)

        # Create tables
        async with self.pool.acquire() as conn:
            await conn.execute(PIPELINE_RUNS_SCHEMA)
            await conn.execute(EVALUATION_RESULTS_SCHEMA)
            # ... create all tables


class SQLiteStorage(StorageBackend):
    """SQLite storage for local/testing."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn: Optional[aiosqlite.Connection] = None

    async def initialize(self):
        """Create database and tables."""
        self.conn = await aiosqlite.connect(self.db_path)

        # Create tables
        await self.conn.execute(PIPELINE_RUNS_SCHEMA)
        await self.conn.execute(EVALUATION_RESULTS_SCHEMA)
        # ... create all tables
```

### Caching Strategy

```python
class CacheManager:
    """Manage LLM response caching."""

    def __init__(
        self,
        backend: CacheBackend,
        ttl_seconds: int = 3600
    ):
        self.backend = backend
        self.ttl = ttl_seconds

    async def get(self, key: str) -> Optional[TransformResult]:
        """Get cached result."""
        cached = await self.backend.get(key)

        if cached is None:
            return None

        # Check TTL
        if cached.timestamp + self.ttl < time.time():
            await self.backend.delete(key)
            return None

        return cached.result

    async def set(self, key: str, result: TransformResult):
        """Cache result."""
        entry = CacheEntry(
            key=key,
            result=result,
            timestamp=time.time()
        )

        await self.backend.set(key, entry)

    async def invalidate(self, pattern: str):
        """Invalidate cache entries matching pattern."""
        await self.backend.delete_pattern(pattern)


class RedisCacheBackend:
    """Redis cache backend."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.client: Optional[redis.Redis] = None

    async def connect(self):
        self.client = await redis.from_url(self.redis_url)

    async def get(self, key: str) -> Optional[CacheEntry]:
        data = await self.client.get(key)
        if data:
            return CacheEntry.parse_raw(data)
        return None

    async def set(self, key: str, entry: CacheEntry):
        await self.client.set(
            key,
            entry.json(),
            ex=3600  # TTL
        )
```

---

## Security Architecture

### Credential Management

```python
class CredentialProvider(ABC):
    """Abstract credential provider."""

    @abstractmethod
    async def get_credential(self, key: str) -> str:
        """Get credential value."""
        pass


class EnvCredentialProvider(CredentialProvider):
    """Load credentials from environment variables."""

    async def get_credential(self, key: str) -> str:
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Credential {key} not found in environment")
        return value


class AWSSecretsProvider(CredentialProvider):
    """Load credentials from AWS Secrets Manager."""

    def __init__(self, region: str):
        self.client = boto3.client('secretsmanager', region_name=region)

    async def get_credential(self, key: str) -> str:
        response = self.client.get_secret_value(SecretId=key)
        return response['SecretString']


class CredentialManager:
    """Manage credential resolution."""

    def __init__(self):
        self.providers: Dict[str, CredentialProvider] = {}

    def register_provider(self, name: str, provider: CredentialProvider):
        self.providers[name] = provider

    async def resolve_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve credential placeholders like ${DB_PASSWORD}."""
        resolved = {}

        for key, value in config.items():
            if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                # Extract credential key and provider
                credential_key = value[2:-1]  # Remove ${ and }

                if ":" in credential_key:
                    provider_name, cred_key = credential_key.split(":", 1)
                    provider = self.providers.get(provider_name)
                else:
                    provider = self.providers.get("env")
                    cred_key = credential_key

                if provider:
                    resolved[key] = await provider.get_credential(cred_key)
                else:
                    raise ValueError(f"Provider not found for {credential_key}")
            else:
                resolved[key] = value

        return resolved
```

### Access Control

```python
class AccessControl:
    """Control access to pipelines and resources."""

    def __init__(self):
        self.policies: List[AccessPolicy] = []

    def add_policy(self, policy: AccessPolicy):
        self.policies.append(policy)

    async def check_access(
        self,
        user: str,
        resource: str,
        action: str
    ) -> bool:
        """Check if user can perform action on resource."""

        for policy in self.policies:
            if policy.matches(user, resource, action):
                return policy.allow

        # Default deny
        return False


@dataclass
class AccessPolicy:
    """Access control policy."""
    user_pattern: str
    resource_pattern: str
    actions: List[str]
    allow: bool

    def matches(self, user: str, resource: str, action: str) -> bool:
        import re

        user_match = re.match(self.user_pattern, user)
        resource_match = re.match(self.resource_pattern, resource)
        action_match = action in self.actions or "*" in self.actions

        return bool(user_match and resource_match and action_match)
```

---

## Deployment Architecture

### Local Development

```
┌─────────────────────────────┐
│   Developer Machine         │
│                             │
│  ┌───────────────────────┐  │
│  │  Loom CLI             │  │
│  └──────────┬────────────┘  │
│             │               │
│  ┌──────────▼────────────┐  │
│  │  Pipeline Executor    │  │
│  └──────────┬────────────┘  │
│             │               │
│  ┌──────────▼────────────┐  │
│  │  SQLite Storage       │  │
│  └───────────────────────┘  │
│                             │
└─────────────────────────────┘
```

### Single Server Deployment

```
┌──────────────────────────────────────────┐
│   Application Server                     │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │  Loom Service                      │  │
│  │  - Pipeline Executor               │  │
│  │  - REST API (Optional)             │  │
│  └────────┬───────────────────────────┘  │
│           │                              │
│  ┌────────▼───────────┐                  │
│  │  PostgreSQL        │                  │
│  │  - Pipeline data   │                  │
│  │  - Metrics         │                  │
│  └────────────────────┘                  │
│                                          │
└──────────────────────────────────────────┘
```

### Production Deployment (Distributed)

```
┌────────────────────────────────────────────────────────┐
│                   Load Balancer                         │
└──────────┬─────────────────────────────────┬───────────┘
           │                                 │
     ┌─────▼──────┐                   ┌─────▼──────┐
     │  Loom      │                   │  Loom      │
     │  Instance  │                   │  Instance  │
     │  #1        │                   │  #2        │
     └─────┬──────┘                   └─────┬──────┘
           │                                 │
           └────────────┬────────────────────┘
                        │
           ┌────────────▼────────────┐
           │   PostgreSQL Cluster    │
           │   - Primary + Replicas  │
           └────────────┬────────────┘
                        │
           ┌────────────▼────────────┐
           │   Redis Cluster         │
           │   - Caching             │
           └─────────────────────────┘
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: loom-executor
spec:
  replicas: 3
  selector:
    matchLabels:
      app: loom-executor
  template:
    metadata:
      labels:
        app: loom-executor
    spec:
      containers:
      - name: loom
        image: loom:latest
        env:
        - name: LOOM_STORAGE_URL
          valueFrom:
            secretKeyRef:
              name: loom-secrets
              key: storage-url
        - name: LOOM_CACHE_URL
          valueFrom:
            secretKeyRef:
              name: loom-secrets
              key: cache-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: loom-service
spec:
  selector:
    app: loom-executor
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

### Airflow Integration Deployment

```
┌─────────────────────────────────────────┐
│   Airflow Cluster                       │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │  Airflow Scheduler                │  │
│  └──────────┬────────────────────────┘  │
│             │                           │
│  ┌──────────▼────────────────────────┐  │
│  │  DAG: customer_sentiment          │  │
│  │                                   │  │
│  │  Task 1: LoomPipelineOperator    │  │
│  │          (runs Loom pipeline)     │  │
│  │                                   │  │
│  │  Task 2: Check Quality           │  │
│  │                                   │  │
│  │  Task 3: Downstream Task         │  │
│  └──────────┬────────────────────────┘  │
│             │                           │
└─────────────┼───────────────────────────┘
              │
     ┌────────▼────────┐
     │  Loom Service   │
     │  - API          │
     │  - Executor     │
     └────────┬────────┘
              │
     ┌────────▼────────┐
     │  PostgreSQL     │
     └─────────────────┘
```

---

## Integration Patterns

### Airflow Integration

```python
# DAG definition
from airflow import DAG
from airflow.operators.python import PythonOperator
from loom.integrations.airflow import LoomPipelineOperator

with DAG(
    dag_id="customer_sentiment_pipeline",
    schedule_interval="@daily",
    start_date=datetime(2024, 1, 1)
) as dag:

    run_loom = LoomPipelineOperator(
        task_id="run_sentiment_analysis",
        pipeline_name="customer_sentiment",
        loom_config_path="/opt/loom/loom.yaml",
        params={
            "execution_date": "{{ ds }}"
        }
    )

    check_quality = PythonOperator(
        task_id="check_quality",
        python_callable=check_quality_metrics,
        op_kwargs={
            "run_id": "{{ task_instance.xcom_pull(task_ids='run_sentiment_analysis') }}"
        }
    )

    run_loom >> check_quality
```

### dbt Integration (Future)

```yaml
# dbt model that uses Loom results
# models/analytics/sentiment_summary.sql

with loom_sentiment as (
  select * from {{ ref('loom_customer_sentiment') }}
),

summary as (
  select
    date_trunc('day', created_at) as date,
    sentiment_category,
    count(*) as review_count,
    avg(evaluation_score) as avg_quality_score
  from loom_sentiment
  where evaluation_passed = true
  group by 1, 2
)

select * from summary
```

### Jupyter Notebook Integration

```python
# In Jupyter notebook
from loom import Loom
import pandas as pd

# Initialize Loom
loom = Loom()

# Load pipeline
pipeline = loom.load_pipeline("customer_sentiment.yaml")

# Run on sample data
sample_df = pd.read_csv("sample_reviews.csv")

results = []
async for record in loom.stream_pipeline(
    pipeline=pipeline,
    input_records=sample_df.to_dict('records')[:10]
):
    results.append({
        'id': record.id,
        'sentiment': record.output,
        'score': record.evaluation_score,
        'passed': record.status == 'validated'
    })

# Analyze results
results_df = pd.DataFrame(results)
print(results_df.describe())
```

### CI/CD Integration

```yaml
# .github/workflows/test-pipelines.yml

name: Test Loom Pipelines

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Install Loom
      run: |
        pip install loom

    - name: Validate Pipelines
      run: |
        loom validate pipelines/*.yaml

    - name: Test Pipeline
      run: |
        loom test customer_sentiment --sample 10
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

    - name: Check Quality Gates
      run: |
        python scripts/check_quality.py
```

---

## Monitoring & Observability

### Metrics Collection

```python
class MetricsCollector:
    """Collect and aggregate pipeline metrics."""

    def __init__(self, storage: StorageBackend):
        self.storage = storage

    async def record_extracted(self, run_id: str, count: int):
        """Record extracted records count."""
        await self.storage.increment_metric(
            run_id=run_id,
            metric_name="records_extracted",
            value=count
        )

    async def record_transform(
        self,
        run_id: str,
        tokens: int,
        cost: float,
        latency_ms: float
    ):
        """Record transform metrics."""
        await self.storage.record_metrics(
            run_id=run_id,
            metrics={
                "tokens_used": tokens,
                "cost": cost,
                "transform_latency_ms": latency_ms
            }
        )

    async def record_evaluation(
        self,
        run_id: str,
        score: float,
        passed: bool
    ):
        """Record evaluation metrics."""
        await self.storage.record_metrics(
            run_id=run_id,
            metrics={
                "evaluation_score": score,
                "evaluation_passed": 1 if passed else 0
            }
        )

    async def get_aggregates(
        self,
        pipeline_name: str,
        time_window: str = "1d"
    ) -> Dict[str, Any]:
        """Get aggregated metrics."""
        return await self.storage.aggregate_metrics(
            pipeline_name=pipeline_name,
            time_window=time_window
        )
```

### Logging Strategy

```python
import structlog

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Usage in code
logger.info(
    "pipeline_started",
    run_id=run_id,
    pipeline_name=pipeline.name,
    pipeline_version=pipeline.version
)

logger.info(
    "transform_complete",
    run_id=run_id,
    records_processed=len(records),
    tokens_used=total_tokens,
    cost=total_cost
)
```

### Prometheus Metrics Export

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
pipeline_runs_total = Counter(
    'loom_pipeline_runs_total',
    'Total pipeline runs',
    ['pipeline_name', 'status']
)

pipeline_duration_seconds = Histogram(
    'loom_pipeline_duration_seconds',
    'Pipeline execution duration',
    ['pipeline_name']
)

records_processed_total = Counter(
    'loom_records_processed_total',
    'Total records processed',
    ['pipeline_name', 'stage']
)

evaluation_score = Histogram(
    'loom_evaluation_score',
    'Evaluation scores',
    ['pipeline_name', 'evaluator']
)

cost_total_dollars = Counter(
    'loom_cost_total_dollars',
    'Total cost in dollars',
    ['pipeline_name']
)

# Update in code
pipeline_runs_total.labels(
    pipeline_name=pipeline.name,
    status="success"
).inc()

pipeline_duration_seconds.labels(
    pipeline_name=pipeline.name
).observe(duration)

cost_total_dollars.labels(
    pipeline_name=pipeline.name
).inc(total_cost)
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Loom Pipeline Monitoring",
    "panels": [
      {
        "title": "Pipeline Success Rate",
        "targets": [
          {
            "expr": "rate(loom_pipeline_runs_total{status=\"success\"}[5m]) / rate(loom_pipeline_runs_total[5m])"
          }
        ]
      },
      {
        "title": "Average Evaluation Score",
        "targets": [
          {
            "expr": "avg(loom_evaluation_score)"
          }
        ]
      },
      {
        "title": "Cost per Hour",
        "targets": [
          {
            "expr": "rate(loom_cost_total_dollars[1h])"
          }
        ]
      },
      {
        "title": "Records Processed",
        "targets": [
          {
            "expr": "sum by (stage) (rate(loom_records_processed_total[5m]))"
          }
        ]
      }
    ]
  }
}
```

### Alerting Rules

```yaml
# alerting_rules.yml

groups:
- name: loom_alerts
  interval: 30s
  rules:

  # Pipeline failure rate too high
  - alert: HighPipelineFailureRate
    expr: |
      rate(loom_pipeline_runs_total{status="failed"}[5m]) /
      rate(loom_pipeline_runs_total[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High pipeline failure rate"
      description: "Pipeline {{ $labels.pipeline_name }} has >10% failure rate"

  # Evaluation scores dropping
  - alert: LowEvaluationScores
    expr: avg(loom_evaluation_score) < 0.7
    for: 10m
    labels:
      severity: warning
    annotations:
      summary: "Low evaluation scores"
      description: "Average evaluation score below 0.7 for 10 minutes"

  # Cost spike
  - alert: CostSpike
    expr: |
      rate(loom_cost_total_dollars[1h]) >
      avg_over_time(rate(loom_cost_total_dollars[1h])[7d]) * 2
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Cost spike detected"
      description: "Hourly cost is 2x the 7-day average"

  # Pipeline duration too long
  - alert: SlowPipeline
    expr: |
      loom_pipeline_duration_seconds >
      avg_over_time(loom_pipeline_duration_seconds[7d]) * 1.5
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Pipeline running slow"
      description: "Pipeline {{ $labels.pipeline_name }} taking 50% longer than usual"
```

---

## Appendix: Design Decisions

### Why Arbiter as Hard Dependency?

**Decision:** Loom has a hard dependency on Arbiter for evaluation (not pluggable in v1.0).

**Context:** During design, we considered whether Loom should abstract evaluation backends (Arbiter, DeepEval, RAGAS, custom solutions) or use Arbiter directly. We chose hard dependency.

**Rationale:**

**1. YAGNI Principle (You Aren't Gonna Need It)**
- No proven demand for alternative evaluation frameworks yet
- Both projects are alpha - no users requesting alternatives
- Premature abstraction adds complexity without demonstrated value
- Build abstractions when needed, not speculatively

**2. Owned Dependency Benefits**
- Both Arbiter and Loom are under same ownership
- Can coordinate breaking changes across both codebases
- Tight coupling enables faster co-evolution during early development
- Iterate quickly without abstraction overhead

**3. Development Velocity**
- Hard dependency: Start Phase 1 implementation immediately
- Pluggable backends: 2-3 weeks designing interfaces, adapters, normalization logic
- At alpha stage, velocity matters more than flexibility
- Focus on core value (orchestration), not abstraction architecture

**4. Separation of Concerns**
- Arbiter's value: Evaluation quality, multiple evaluator types, interaction tracking
- Loom's value: Orchestration, lineage, cost tracking, quality gates
- Loom uses Arbiter like it uses PostgreSQL (storage) or Redis (caching)
- Don't reinvent evaluation - use best-in-class tool

**5. Low Migration Risk**
- Abstraction can be added in v2.0 if demand emerges
- Migration path: Introduce `EvaluationBackend` interface, wrap Arbiter in `ArbiterBackend`
- Default to `ArbiterBackend` - no breaking changes for existing users
- Users who don't need custom backends see no difference

**6. Complexity Avoidance**
- Pluggable backends require ~100+ lines of abstraction code
- Need result normalization across different frameworks (Arbiter, DeepEval, RAGAS have incompatible APIs)
- Configuration complexity - each backend has different config schemas
- Testing burden - validate against multiple backend implementations
- Permanent maintenance cost even if never used

**Learning from Others:**
- **LangChain:** Over-abstraction created middleware sprawl and complexity burden
- **Airflow/dbt:** Abstracted executors/adapters because execution environments are fundamentally diverse
- **Evaluation:** Not yet proven diverse enough to warrant abstraction

**The Relationship:**
```
Loom: Orchestration framework
  ├── Extract: Data sources via connectors
  ├── Transform: AI transformations via LLM clients
  ├── Evaluate: Quality assessment via Arbiter (hard dependency)
  └── Load: Data destinations via connectors

Arbiter: Evaluation framework
  ├── SemanticEvaluator: Similarity scoring
  ├── CustomCriteriaEvaluator: Domain-specific evaluation
  ├── PairwiseComparator: A/B testing
  └── Extensible evaluator registry
```

**Implementation:**
```toml
# pyproject.toml
[project]
dependencies = [
    "arbiter>=0.1.0",  # Hard dependency
    # ...
]
```

```python
# loom/engines/evaluate.py
from arbiter import evaluate  # Direct import

class EvaluateEngine:
    async def evaluate(self, output, reference, config):
        result = await evaluate(
            output=output,
            reference=reference,
            evaluators=config.evaluators,
            model=config.model
        )
        return result
```

**Future Path (v2.0+):**
If demand emerges for alternative evaluation backends:

```python
# Add abstraction without breaking changes
class EvaluationBackend(ABC):
    @abstractmethod
    async def evaluate(...) -> EvaluationResult: pass

class ArbiterBackend(EvaluationBackend):
    """Default backend - wraps Arbiter"""
    async def evaluate(...):
        from arbiter import evaluate
        return await evaluate(...)

class CustomBackend(EvaluationBackend):
    """User-provided backend"""
    async def evaluate(...):
        # Custom implementation
        pass

# Default to Arbiter - backward compatible
backend = ArbiterBackend()  # Or: CustomBackend()
```

**Conclusion:** Hard dependency is the right choice for v1.0. It's not "too difficult" to abstract - it's "too early" to abstract. Let real-world usage patterns emerge before adding complexity.

### Production Resilience Patterns

**Decision:** Implement circuit breaker, timeout specifications, and quality gate semantics before Phase 1 completion.

**Context:** Expert specification review (2025-11-14) identified critical production readiness gaps:
1. No circuit breaker for upstream service protection
2. Missing timeout specifications
3. Ambiguous quality gate semantics

**Implementation:**

**1. Circuit Breaker Pattern** (see `loom/resilience/circuit_breaker.py`)
- Protects against cascading failures from upstream services (Arbiter, databases, S3)
- Three states: CLOSED (normal), OPEN (failing), HALF_OPEN (testing recovery)
- Configurable failure threshold, timeout, and success threshold
- Prevents DoS on recovering services by blocking requests when open

```python
# Example: Protect LLM API calls with circuit breaker
from loom.resilience import CircuitBreaker, CircuitBreakerConfig

llm_breaker = CircuitBreaker(
    name="arbiter_llm",
    config=CircuitBreakerConfig(
        failure_threshold=5,      # Open after 5 consecutive failures
        timeout_seconds=60,       # Stay open for 60s before testing
        success_threshold=3       # Close after 3 consecutive successes
    )
)

# Use in transform engine
try:
    response = await llm_breaker.call(
        self.llm_client.generate,
        prompt=prompt
    )
except CircuitBreakerOpenError:
    # Fallback: use cached response or skip record
    response = get_cached_response(prompt)
```

**2. Timeout Specifications** (see `TIMEOUTS.md`)
- Precise timeout values for all external operations
- LLM API calls: 30-60s depending on model
- Database queries: 3-30s with statement-level enforcement
- S3 operations: 5-60s
- Cache operations: 1-2s with graceful fallback
- Pipeline stages: 15-45min per stage, 1h total

**3. Quality Gate Semantics** (see `QUALITY_GATES.md`)
- Unambiguous definitions for `all_pass`, `majority_pass`, `any_pass`, `weighted`
- Concrete Given/When/Then examples for all gate types
- SQL verification queries
- Complete implementation reference with tests

**Rationale:**
- **Circuit breaker prevents cascading failures**: Without it, 10K records × 3 retries = 30K failed API calls overwhelming recovering service
- **Timeouts prevent resource exhaustion**: Unbounded waits lead to thread pool exhaustion and cascading failures
- **Quality gate precision enables correct implementation**: Ambiguous specifications lead to implementation errors and rework

**Related Documents:**
- `QUALITY_GATES.md` - Detailed quality gate specifications
- `TIMEOUTS.md` - Complete timeout configuration
- `loom/resilience/` - Circuit breaker and retry implementations

### Why YAML for Pipeline Definitions?

**Decision:** Use YAML as the primary pipeline definition format.

**Rationale:**
- **Declarative:** Matches dbt's approach, familiar to data engineers
- **Version Control Friendly:** Text-based, easy to diff
- **Human Readable:** Non-developers can understand and modify
- **Standard:** YAML is ubiquitous in data engineering (Airflow, dbt, Kubernetes)

### Why Async/Await Architecture?

**Decision:** Use async/await throughout the codebase.

**Rationale:**
- **I/O Bound:** Pipeline operations are dominated by I/O (database, LLM API calls)
- **Efficiency:** Handle thousands of records concurrently without threading overhead
- **Backpressure:** Easier to implement flow control with async patterns
- **Modern Python:** async/await is standard for I/O-heavy applications

### Why PostgreSQL as Primary Storage?

**Decision:** Use PostgreSQL for production, SQLite for local development.

**Rationale:**
- **ACID Transactions:** Critical for pipeline consistency
- **JSON Support:** Store complex evaluation results as JSONB
- **Performance:** Mature indexing and query optimization
- **Ecosystem:** Wide deployment support, tooling, monitoring

### Why Connector Abstraction?

**Decision:** Abstract data sources/destinations behind connector interface.

**Rationale:**
- **Extensibility:** Users can add custom connectors for any data source
- **Testing:** Easy to mock connectors for testing
- **Provider Independence:** Not locked to specific databases or APIs
- **Reusability:** Connectors can be shared across pipelines

### Why Python (Not Go)?

**Decision:** Implement Loom in Python 3.10+, not Go.

**Context:** During initial setup, we considered whether Python or Go would be better for Loom's implementation.

**Rationale:**

**1. Arbiter Integration (Decisive)**
- Arbiter is Python with hard dependency design
- Python: Direct import (`from arbiter import evaluate`) - zero integration overhead
- Go: Would require HTTP/gRPC service wrapper, serialization, network latency
- Saves 2-3 weeks of development time for integration layer alone

**2. Data Engineering Ecosystem**
- Target audience: Data engineers using dbt (Python), Airflow (Python), Pandas (Python)
- ALL major data orchestration tools are Python (dbt, Airflow, Prefect, Dagster)
- Cultural fit - data engineers expect Python tools
- Rich connector ecosystem (SQLAlchemy, database drivers)

**3. I/O-Bound Workload**
- Pipeline bottleneck: LLM API calls (100-1000ms) + database I/O (10-100ms)
- Python overhead (~1ms function call) is negligible compared to external I/O
- Go's performance advantage doesn't matter for I/O-bound workloads
- Even processing 10K records, limited by LLM API rate limits, not Python speed

**4. Development Velocity**
- Both Arbiter and Loom in same language
- Share patterns: async/await, Pydantic models, error handling
- Reuse infrastructure: LLM client abstraction from Arbiter
- Single development environment and testing framework
- Faster iteration during alpha phase

**5. Maintenance**
- One language to maintain vs context switching Python ↔ Go
- Coordinated evolution of both projects
- Easy to coordinate breaking changes
- Shared code patterns and conventions

**Go's Advantages Don't Apply:**

**Single Binary Distribution:**
- Modern Python packaging (uv, pipx) makes distribution simple:
  ```bash
  uv tool install loom  # Global installation
  loom run pipeline     # Just works
  ```
- Data engineers already have Python installed
- Docker deployment is equivalent for both languages

**Performance:**
- Irrelevant for I/O-bound workload
- Pipeline execution time dominated by external API calls, not CPU

**Concurrency:**
- Python's asyncio handles thousands of concurrent LLM calls efficiently
- Goroutines offer no practical advantage for this use case

**Real-World Pattern:**
Every major data orchestration tool chose Python for the same reasons:
- **dbt:** Python (orchestrates SQL transformations)
- **Airflow:** Python (orchestrates data pipelines)
- **Prefect:** Python (modern workflow orchestration)
- **Dagster:** Python (data orchestration with testing)

**Exception Scenario:**
If Loom later needs CPU-intensive transformations at millions of records/second, THEN consider Go rewrite. But start with Python following "Make it work, make it right, make it fast."

**Conclusion:** Python is the clear choice for Loom v1.0. It's not "Go can't work" - it's "Python is dramatically better suited for this specific use case."

---

## Summary

This architecture document provides the complete technical design for Loom:

1. **System Architecture** - High-level component organization and data flow
2. **API Design** - CLI, Python API, and future REST API specifications
3. **Component Architecture** - Detailed design of core engines and components
4. **Database Architecture** - Storage schema and backend implementations
5. **Security Architecture** - Credential management and access control
6. **Deployment Architecture** - From local development to production Kubernetes
7. **Integration Patterns** - Airflow, dbt, Jupyter, CI/CD integrations
8. **Monitoring & Observability** - Metrics, logging, alerting strategies
9. **Production Resilience** - Circuit breaker, timeouts, quality gate semantics

**Related Specifications:**
- **IMPLEMENTATION_SPEC.md** - Detailed implementation tasks and SQL schemas
- **QUALITY_GATES.md** - Precise quality gate semantics with executable examples
- **TIMEOUTS.md** - Timeout specifications for all external operations

This design is ready for Phase 1 implementation.
