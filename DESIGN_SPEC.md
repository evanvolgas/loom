# Loom Design Specification

**Version:** 0.1.0-alpha
**Status:** Design Phase
**Created:** 2025-11-14

---

## Vision

**Loom is the orchestration framework for AI pipelines—the "dbt for AI(E)TL."**

### The Problem

Teams building AI pipelines face a critical gap:

**Airflow/Prefect** orchestrate workflows but know nothing about:
- LLM evaluation and quality gates
- Prompt versioning and context management
- AI-specific cost tracking and optimization
- Semantic lineage (model versions, prompt versions)

**LangChain/LlamaIndex** handle LLM interactions but aren't built for:
- Production pipeline orchestration
- Declarative pipeline definitions
- Cost budgeting and SLA monitoring
- Multi-stage evaluation workflows

**dbt** revolutionized SQL pipelines but doesn't handle:
- Probabilistic transformations (LLMs)
- Evaluation gates for non-deterministic outputs
- Prompt versioning as code
- Token cost management

**The gap:** No framework exists for declarative, testable, observable AI pipelines with built-in evaluation.

### The Solution

Loom fills this gap by providing:

1. **Declarative Pipeline Definitions** → YAML-based AI(E)TL pipelines
2. **Prompt Versioning** → Prompts as code, version-controlled
3. **Built-in Evaluation** → Quality gates using Arbiter
4. **Cost Management** → Budgets, tracking, optimization
5. **Testing Framework** → Unit tests for prompts, distributional tests
6. **Semantic Lineage** → Track prompts, models, context versions
7. **CLI Tools** → `loom run`, `loom test`, `loom cost`

---

## Core Concepts

### 1. Pipeline

A declarative definition of an AI(E)TL workflow:

```yaml
# pipelines/customer_sentiment.yaml
name: customer_sentiment
version: 2.1.0
description: Classify customer review sentiment

extract:
  source: postgres://customers/reviews
  query: "SELECT * FROM reviews WHERE created_at > {{ execution_date }}"

transform:
  type: ai
  prompt: prompts/classify_sentiment.txt
  prompt_version: v2.3.0
  model: gpt-4o-mini
  model_version: 2024-07-18
  temperature: 0.0
  context:
    - context/customer_domain.md
    - context/sentiment_taxonomy.md
  batch_size: 50
  cache:
    enabled: true
    ttl: 7d
    key_fields: [review_text, model, prompt_version]

evaluate:
  evaluators:
    - name: semantic_similarity
      type: semantic
      config:
        reference_field: expected_sentiment
        threshold: 0.8
    - name: quality_criteria
      type: custom_criteria
      config:
        criteria: "Accurate sentiment, appropriate tone, no hallucination"
        threshold: 0.75
  quality_gate: all_pass  # Options: all_pass, majority_pass, any_pass, weighted
                                # See QUALITY_GATES.md for detailed semantics

load:
  destination: postgres://analytics/sentiment_scores
  on_failure: quarantine  # Options: quarantine, skip, fallback, retry
  schema:
    sentiment: string
    confidence: float
    reasoning: string

monitoring:
  cost_budget: 10.00  # USD
  latency_sla:
    p95: 5.0  # seconds
  quality_threshold:
    min_score: 0.7
  alerts:
    - type: score_degradation
      threshold: 0.7
      channel: slack
    - type: cost_spike
      threshold: 2x  # 2x baseline
      channel: pagerduty
```

### 2. Prompt

Version-controlled prompt files with template variables:

```
# prompts/classify_sentiment.txt (v2.3.0)
You are a sentiment classifier for customer reviews.

## Context

{{ context.customer_domain }}

## Sentiment Taxonomy

{{ context.sentiment_taxonomy }}

## Task

Review: {{ input.review }}

Classify the sentiment as one of: positive, negative, neutral

Provide:
- sentiment: The classification
- confidence: 0.0 to 1.0
- reasoning: Brief explanation

## Output Format

{
  "sentiment": "positive|negative|neutral",
  "confidence": 0.95,
  "reasoning": "Customer expresses satisfaction with product quality"
}
```

Context files:

```markdown
# context/customer_domain.md (v1.5)
Our customers are B2B SaaS companies using our platform for analytics.
Reviews typically mention: performance, ease of use, support quality, pricing.
```

### 3. Evaluator

Quality gate using Arbiter for validation:

```yaml
evaluators:
  - name: semantic_check
    type: semantic  # From Arbiter
    config:
      reference_field: expected_output
      threshold: 0.8

  - name: domain_criteria
    type: custom_criteria  # From Arbiter
    config:
      criteria: |
        - Factually accurate based on input
        - No hallucinated information
        - Appropriate tone and language
        - Follows output format specification
      threshold: 0.75
```

### 4. Run

Execution with full lineage tracking:

```json
{
  "run_id": "2025-11-14-001",
  "pipeline": "customer_sentiment",
  "pipeline_version": "2.1.0",
  "execution_date": "2025-11-14T10:00:00Z",

  "config": {
    "prompt_version": "v2.3.0",
    "model": "gpt-4o-mini",
    "model_version": "2024-07-18",
    "context_versions": {
      "customer_domain": "v1.5",
      "sentiment_taxonomy": "v2.0"
    }
  },

  "results": {
    "status": "success",
    "records_processed": 10000,
    "records_validated": 9847,
    "records_rejected": 153,
    "rejection_rate": 0.015
  },

  "evaluation": {
    "semantic_similarity": {
      "mean": 0.87,
      "std": 0.12,
      "min": 0.45,
      "max": 1.0
    },
    "quality_criteria": {
      "mean": 0.82,
      "std": 0.15
    }
  },

  "cost": {
    "total": 2.34,
    "breakdown": {
      "transform": 1.80,
      "evaluate": 0.54
    }
  },

  "performance": {
    "duration": 125.3,
    "latency": {
      "p50": 1.2,
      "p95": 3.4,
      "p99": 5.8
    }
  }
}
```

---

## Architecture

### High-Level Design

```
┌─────────────────────────────────────────────────────────┐
│                     Loom CLI                            │
│  loom run | loom test | loom cost | loom lineage       │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                  Pipeline Engine                        │
│  • Parse YAML pipelines                                 │
│  • Build execution DAG                                  │
│  • Coordinate Extract → Transform → Evaluate → Load    │
└─────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
    ┌────────┐    ┌─────────┐   ┌──────────┐   ┌────────┐
    │Extract │    │Transform│   │ Evaluate │   │  Load  │
    │ Engine │    │  Engine │   │  Engine  │   │ Engine │
    └────────┘    └─────────┘   └──────────┘   └────────┘
         │              │              │              │
         │              │         ┌────▼────┐         │
         │              │         │ Arbiter │         │
         │              │         │Framework│         │
         │              │         └─────────┘         │
         ▼              ▼                              ▼
    ┌─────────────────────────────────────────────────────┐
    │              Observability Layer                    │
    │  • Lineage Tracking                                 │
    │  • Cost Monitoring                                  │
    │  • Performance Metrics                              │
    │  • Quality Scoring                                  │
    └─────────────────────────────────────────────────────┘
                           │
                           ▼
                   ┌──────────────┐
                   │Storage Backend│
                   │SQLite/Postgres│
                   └──────────────┘
```

### Component Breakdown

**1. Pipeline Engine**
- Parse YAML pipeline definitions
- Build DAG from dependencies
- Schedule execution order
- Handle retries and failures

**2. Extract Engine**
- Connect to data sources (Postgres, S3, APIs)
- Apply filters and transformations
- Batch data for processing

**3. Transform Engine**
- Load prompts and context
- Execute LLM calls with batching
- Cache responses intelligently
- Handle model fallbacks

**4. Evaluate Engine**
- **Integrates with Arbiter** for quality gates
- Run multiple evaluators in parallel
- Aggregate scores
- Apply quality gates (all_pass, majority_pass)

**5. Load Engine**
- Write validated data to destinations
- Quarantine failed records
- Handle schema validation

**6. Observability Layer**
- Track lineage (prompts, models, context versions)
- Monitor costs and latency
- Record evaluation scores
- Generate compliance reports

---

## Integration with Arbiter

Loom **uses** Arbiter as its evaluation engine:

```python
# Loom's Evaluate Engine uses Arbiter
from arbiter import evaluate

async def evaluate_outputs(
    outputs: List[dict],
    evaluator_configs: List[dict],
    quality_gate: str
) -> EvaluationResult:
    """Use Arbiter to evaluate pipeline outputs."""

    results = []
    for output in outputs:
        # Use Arbiter's evaluate() function
        result = await evaluate(
            output=output["text"],
            reference=output.get("reference"),
            criteria=output.get("criteria"),
            evaluators=[cfg["type"] for cfg in evaluator_configs],
            model="gpt-4o-mini"
        )
        results.append(result)

    # Apply quality gate
    if quality_gate == "all_pass":
        passed = all(r.passed for r in results)
    elif quality_gate == "majority_pass":
        passed = sum(r.passed for r in results) > len(results) / 2

    return EvaluationResult(
        passed=passed,
        results=results,
        mean_score=sum(r.overall_score for r in results) / len(results)
    )
```

**Separation of Concerns:**
- **Arbiter**: Evaluates individual LLM outputs (what)
- **Loom**: Orchestrates pipelines with evaluation gates (when/how)

---

## CLI Design

### Commands

**`loom run <pipeline>`**
```bash
$ loom run customer_sentiment

Pipeline: customer_sentiment (v2.1.0)
Status: Running...

Extract:   ✅ 10,000 records (2.3s)
Transform: ✅ 10,000 processed (45.2s)
Evaluate:  ✅ 9,847 passed, 153 rejected (78.1s)
Load:      ✅ 9,847 loaded (3.2s)

Results:
  Processed: 10,000
  Validated: 9,847 (98.5%)
  Rejected:  153 (1.5%)

Evaluation Scores:
  Semantic Similarity: 0.87 (±0.12)
  Quality Criteria:    0.82 (±0.15)

Cost: $2.34
  Transform: $1.80
  Evaluate:  $0.54

Performance:
  Total Duration: 128.8s
  p50 Latency:    1.2s
  p95 Latency:    3.4s

✅ Pipeline completed successfully
```

**`loom test <pipeline>`**
```bash
$ loom test customer_sentiment

Running tests for: customer_sentiment (v2.1.0)

Unit Tests:
  ✅ obvious_positive (score: 0.95)
  ✅ obvious_negative (score: 0.92)
  ✅ subtle_neutral (score: 0.78)

Distributional Tests:
  ✅ mean_score > 0.75 (actual: 0.87)
  ✅ std_score < 0.3 (actual: 0.12)
  ✅ hallucination_rate < 0.05 (actual: 0.02)

Integration Tests:
  ✅ end_to_end_pipeline

All tests passed (7/7)
```

**`loom cost <pipeline> --dry-run`**
```bash
$ loom cost customer_sentiment --dry-run

Cost Estimate for: customer_sentiment

Input:
  Records: 10,000
  Avg tokens per record: 250

Transform:
  Model: gpt-4o-mini ($0.15 / 1M tokens)
  Input tokens:  2,500,000
  Output tokens: 1,000,000
  Cost: $1.80

Evaluate:
  Model: gpt-4o-mini ($0.15 / 1M tokens)
  Input tokens:  1,500,000
  Output tokens: 500,000
  Cost: $0.54

Total Estimated Cost: $2.34
Budget: $10.00 (23% utilized)
```

**`loom lineage <pipeline>`**
```bash
$ loom lineage customer_sentiment

Pipeline: customer_sentiment (v2.1.0)
Last Run: 2025-11-14T10:00:00Z

Lineage:
  Prompt:  classify_sentiment.txt (v2.3.0)
  Model:   gpt-4o-mini (2024-07-18)
  Context:
    - customer_domain.md (v1.5)
    - sentiment_taxonomy.md (v2.0)

Inputs:
  - postgres://customers/reviews

Outputs:
  - postgres://analytics/sentiment_scores

Evaluators:
  - semantic_similarity (Arbiter)
  - quality_criteria (Arbiter)

Dependencies:
  Upstream:   None
  Downstream: customer_insights (v1.2.0)
```

**`loom validate <pipeline>`**
```bash
$ loom validate customer_sentiment

Validating: customer_sentiment (v2.1.0)

Pipeline Definition:
  ✅ YAML syntax valid
  ✅ Required fields present
  ✅ Evaluator configs valid

Prompts:
  ✅ classify_sentiment.txt exists
  ✅ Context variables resolved
  ✅ Output format specified

Context:
  ✅ customer_domain.md (v1.5) exists
  ✅ sentiment_taxonomy.md (v2.0) exists

Connections:
  ✅ Source: postgres://customers/reviews reachable
  ✅ Destination: postgres://analytics/sentiment_scores writable

Pipeline is valid ✅
```

---

## Testing Framework

### Unit Tests (Prompt-Level)

```yaml
# tests/customer_sentiment_test.yaml
pipeline: customer_sentiment

unit_tests:
  - name: obvious_positive
    input:
      review: "This product is amazing! I love it!"
    expected:
      sentiment: positive
      min_confidence: 0.9
    assertions:
      - evaluation_score > 0.9

  - name: obvious_negative
    input:
      review: "Terrible product. Total waste of money."
    expected:
      sentiment: negative
      min_confidence: 0.9

  - name: subtle_neutral
    input:
      review: "The product works as described."
    expected:
      sentiment: neutral
      min_confidence: 0.7
```

### Distributional Tests (Statistical)

```yaml
distributional_tests:
  - name: quality_distribution
    type: statistical
    sample_size: 1000
    assertions:
      - mean(evaluation_scores) > 0.75
      - std(evaluation_scores) < 0.3
      - min(evaluation_scores) > 0.5

  - name: hallucination_check
    type: hallucination_rate
    sample_size: 1000
    threshold: 0.05  # Less than 5%

  - name: cost_check
    type: cost_budget
    sample_size: 1000
    max_cost: 2.50  # Per 1000 records
```

### Integration Tests (End-to-End)

```yaml
integration_tests:
  - name: end_to_end_pipeline
    steps:
      - extract: 100 sample records
      - transform: classify sentiment
      - evaluate: check quality gates
      - load: verify destination
    assertions:
      - records_loaded > 95  # At least 95% pass
      - total_cost < 0.50
      - duration < 60  # seconds
```

---

## Technical Decisions

### Technology Stack

- **Language**: Python 3.10+
- **Pipeline Definitions**: YAML
- **Prompt Templates**: Jinja2
- **Evaluation**: Arbiter (integration)
- **Storage**: SQLite (dev), Postgres (prod)
- **CLI**: Click or Typer
- **Async**: asyncio for parallel execution
- **Observability**: Structured logging (structlog)
- **Testing**: pytest + custom test runner

### Key Design Choices

**1. YAML over Python DSL**
- **Why**: Declarative, version-controllable, non-programmers can write pipelines
- **Trade-off**: Less flexible than code, but more maintainable

**2. Arbiter Integration (not reimplementation)**
- **Why**: Arbiter already solves evaluation, don't rebuild it
- **Trade-off**: Dependency, but creates ecosystem

**3. Storage Backend Required**
- **Why**: Need persistent lineage, run history, cost tracking
- **Trade-off**: Setup complexity, but necessary for production

**4. No Built-in Workflow Orchestration**
- **Why**: Airflow/Prefect handle this well
- **Trade-off**: Loom pipelines can be called from Airflow tasks

**5. Prompts as Files (not inline)**
- **Why**: Versioning, testing, reusability
- **Trade-off**: More files, but better organization

---

## Phase Plan

### Phase 1: Core Pipeline Engine (4 weeks)
- YAML parser for pipeline definitions
- Basic Extract → Transform → Load engine
- Simple CLI (`loom run`, `loom validate`)
- SQLite storage backend
- Manual Arbiter integration (hardcoded)

**Deliverable**: Can run basic AI pipeline from YAML

### Phase 2: Arbiter Integration (2 weeks)
- Evaluate engine using Arbiter
- Quality gate logic (all_pass, majority_pass)
- Evaluation result storage
- CLI: `loom test` for unit tests

**Deliverable**: Evaluation gates work, pipelines can fail on quality

### Phase 3: Observability & Lineage (3 weeks)
- Full lineage tracking (prompts, models, context)
- Cost tracking and reporting
- Performance metrics (latency, throughput)
- CLI: `loom cost`, `loom lineage`

**Deliverable**: Complete visibility into pipeline runs

### Phase 4: Testing Framework (2 weeks)
- Unit test runner for prompts
- Distributional test assertions
- Integration test support
- CI/CD integration patterns

**Deliverable**: Pipelines can be tested like code

### Phase 5: Optimization (3 weeks)
- Smart caching (content + model + prompt hash)
- Automatic batching
- Model fallback strategies
- Retry logic with exponential backoff

**Deliverable**: Production-ready performance

### Phase 6: Monitoring & Alerts (2 weeks)
- Score degradation detection
- Cost spike alerts
- Latency SLA monitoring
- Integration with Slack, PagerDuty

**Deliverable**: Production monitoring

### Phase 7: Advanced Features (4 weeks)
- Dependency management (pipeline DAG)
- Parallel execution
- Incremental processing
- Data versioning

**Deliverable**: Enterprise-grade orchestration

---

## Example Use Cases

### Use Case 1: Customer Sentiment Pipeline

**Before Loom:**
- 200 lines of Python to extract, classify, evaluate, load
- Manual cost tracking (if at all)
- No testing framework
- Prompt changes require code changes
- No lineage tracking

**With Loom:**
- 50 lines of YAML
- Automatic cost tracking
- Built-in testing
- Prompts version-controlled separately
- Complete lineage automatically

### Use Case 2: Document Summarization

**Pipeline:**
```yaml
name: doc_summarization
extract:
  source: s3://documents/raw
transform:
  prompt: prompts/summarize_document.txt
  model: gpt-4o  # Use powerful model
evaluate:
  - factuality
  - completeness
  - no_hallucination
load:
  destination: postgres://summaries
  on_failure: fallback  # Use extractive summary
```

**Value:** Ensures high-quality summaries, catches hallucinations

### Use Case 3: RAG Pipeline Evaluation

**Pipeline:**
```yaml
name: rag_qa
extract:
  source: postgres://questions
transform:
  type: rag
  prompt: prompts/answer_question.txt
  context: {{ retrieved_documents }}
evaluate:
  - groundedness  # Answer supported by docs
  - relevance     # Answers the question
  - attribution   # Cites sources
load:
  destination: postgres://answers
  quality_gate: all_pass  # Must pass all
```

**Value:** Prevents hallucinated answers in RAG systems

---

## Differentiation from Existing Tools

### vs. Airflow/Prefect
- **They**: Workflow orchestration (general-purpose)
- **Loom**: AI pipeline orchestration (specialized)
- **Difference**: Built-in evaluation, prompt versioning, cost tracking

### vs. LangChain
- **They**: LLM application framework
- **Loom**: Pipeline orchestration with evaluation
- **Difference**: Declarative YAML, testing framework, lineage

### vs. dbt
- **They**: SQL transformation pipelines
- **Loom**: AI transformation pipelines
- **Difference**: Handles probabilistic transforms, evaluation gates

### vs. Great Expectations
- **They**: Data quality testing
- **Loom**: AI output quality testing
- **Difference**: Semantic evaluation, distributional tests, LLM-specific

---

## Success Metrics

**Adoption Metrics:**
- Teams using Loom for production AI pipelines
- Number of pipelines defined in YAML
- GitHub stars and community engagement

**Quality Metrics:**
- Reduction in hallucination rate (compared to no evaluation)
- Increase in evaluation coverage (% of AI outputs evaluated)
- Faster debugging (time to identify pipeline issues)

**Efficiency Metrics:**
- Cost savings from caching and optimization
- Reduction in code (YAML vs custom Python)
- Time to deploy new AI pipelines

---

## Open Questions

1. **Should Loom have its own workflow orchestration, or integrate with Airflow?**
   - Initial: Integrate with Airflow (Loom pipelines as Airflow tasks)
   - Future: Optional standalone orchestration

2. **How to handle streaming pipelines vs batch?**
   - Phase 1: Batch only
   - Phase 7+: Streaming support

3. **Should prompts support multiple template engines?**
   - Start: Jinja2 only
   - Future: Pluggable (Mustache, etc.)

4. **How deeply to integrate with vector databases?**
   - Phase 1: Generic extract/load
   - Future: Native Milvus/Pinecone integration

5. **Should Loom provide hosted evaluation (SaaS)?**
   - Phase 1: Open source only
   - Future: Evaluate market need

---

## Related Work

- **dbt**: SQL transformation framework
- **Airflow**: Workflow orchestration
- **Great Expectations**: Data quality
- **LangChain**: LLM application framework
- **Arbiter**: LLM evaluation framework (our integration partner)
- **Metaflow**: ML pipeline orchestration

---

## Conclusion

Loom fills a critical gap in the AI infrastructure stack: **declarative, testable, observable orchestration for AI pipelines with built-in evaluation.**

By integrating with Arbiter for quality gates and providing YAML-based pipeline definitions, Loom enables teams to:
1. Build AI pipelines as code
2. Test prompts like SQL queries
3. Track costs and quality automatically
4. Scale evaluation from 10 to 10,000 outputs

**Next Step:** Begin Phase 1 implementation - Core Pipeline Engine
