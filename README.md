# Loom

**Orchestration framework for AI pipelines with built-in evaluation**

<p><em>⚠️ Design Phase: Not yet implemented. See DESIGN_SPEC.md for vision.</em></p>

---

## What is Loom?

Loom is the **"dbt for AI(E)TL"** - a declarative orchestration framework for AI pipelines.

Traditional ETL becomes AI(E)TL: **Extract → Transform → Evaluate → Load**

Loom handles the orchestration with:
- **Declarative YAML pipelines** → Define AI workflows as code
- **Built-in evaluation** → Quality gates using [Arbiter](https://github.com/evanvolgas/arbiter)
- **Prompt versioning** → Prompts as code, version-controlled
- **Cost tracking** → Automatic token cost monitoring
- **Testing framework** → Unit tests for prompts, distributional tests
- **Semantic lineage** → Track prompts, models, context versions

## Why Loom?

**The Problem:** No framework exists for declarative, testable, observable AI pipelines with built-in evaluation.

- **Airflow/Prefect**: Orchestrate workflows but don't understand AI evaluation
- **LangChain**: LLM framework but not pipeline orchestration
- **dbt**: Revolutionized SQL pipelines but doesn't handle probabilistic AI transforms

**The Solution:** Loom fills the gap.

## Quick Example

```yaml
# pipelines/customer_sentiment.yaml
name: customer_sentiment
version: 2.1.0

extract:
  source: postgres://customers/reviews

transform:
  prompt: prompts/classify_sentiment.txt
  model: gpt-4o-mini
  batch_size: 50

evaluate:
  evaluators:
    - type: semantic
      threshold: 0.8
    - type: custom_criteria
      criteria: "Accurate, no hallucination"
      threshold: 0.75
  quality_gate: all_pass

load:
  destination: postgres://analytics/sentiment_scores
```

Run it:
```bash
loom run customer_sentiment
```

## Documentation

**Current Phase:** Design

- **[DESIGN_SPEC.md](DESIGN_SPEC.md)** - Vision, high-level architecture, and roadmap
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Comprehensive system design with component diagrams, API specs, and deployment patterns
- **[IMPLEMENTATION_SPEC.md](IMPLEMENTATION_SPEC.md)** - Technical details, storage schema, and Phase 1 implementation plan

## Getting Started

```bash
# Clone the repository
git clone https://github.com/evanvolgas/loom.git
cd loom

# Install dependencies
uv venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -e ".[dev]"

# Run tests
pytest
```

## Relationship to Arbiter

Loom **uses** [Arbiter](https://github.com/evanvolgas/arbiter) as its evaluation engine:
- **Arbiter**: Evaluates individual LLM outputs (what)
- **Loom**: Orchestrates pipelines with evaluation gates (when/how)

Separate projects, complementary goals.

## Roadmap

- **Phase 1**: Core pipeline engine (YAML parsing, basic Extract/Transform/Load)
- **Phase 2**: Arbiter integration (evaluation gates, quality checks)
- **Phase 3**: Observability & lineage (cost tracking, performance metrics)
- **Phase 4**: Testing framework (unit tests, distributional tests)
- **Phase 5**: Optimization (caching, batching, retries)
- **Phase 6**: Monitoring & alerts (score degradation, cost spikes)
- **Phase 7**: Advanced features (DAG dependencies, incremental processing)

See [DESIGN_SPEC.md](DESIGN_SPEC.md) for details.

## License

MIT License

## Acknowledgments

Inspired by dbt, Airflow, and the need for better AI pipeline tooling.
