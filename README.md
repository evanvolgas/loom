<div align="center">
  <h1>Loom</h1>

  <p><strong>Orchestration framework for AI pipelines with built-in evaluation</strong></p>

  <p>
    <a href="https://python.org"><img src="https://img.shields.io/badge/python-3.10+-blue" alt="Python"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License"></a>
    <a href="https://github.com/evanvolgas/loom"><img src="https://img.shields.io/badge/version-0.1.0--alpha-blue" alt="Version"></a>
  </p>

  <p><em>⚠️ Alpha Software: Early development stage. Use for evaluation and experimentation.</em></p>
</div>

---

## What is Loom?

Loom is the **"dbt for AI(E)TL"** - a declarative orchestration framework for AI pipelines.

Traditional ETL becomes AI(E)TL: **Extract → Transform → Evaluate → Load**

Declarative YAML pipelines with built-in quality gates ensure your AI outputs meet quality thresholds before reaching production.

**Core Value**: Production-grade AI pipeline orchestration without complexity, vendor lock-in, or hidden evaluation gaps.

**Status**: Alpha software (v0.1.0-alpha). Functional but early-stage. Best suited for evaluation, experimentation, and development.

## Why Loom?

**The Problem:** Building production AI pipelines requires orchestration AND evaluation. Existing tools do one or the other, not both.

**What Loom Provides:**
- **Declarative Pipelines**: Define AI workflows as version-controlled YAML
- **Built-in Evaluation**: Quality gates using [Arbiter](https://github.com/evanvolgas/arbiter) prevent bad outputs from reaching production
- **Provider-Agnostic**: Works with OpenAI, Anthropic, Google, Groq - no vendor lock-in
- **Production-Ready**: Circuit breakers, retry logic, timeout enforcement

**Use Case Example:**
A sentiment analysis pipeline needs quality assurance. Loom provides:
1. Declarative YAML pipeline definition (Extract → Transform → Evaluate → Load)
2. Automatic evaluation with configurable quality gates
3. Quarantine pattern for failed records
4. Complete audit trail of transformations and evaluations

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

## Key Features

- **✅ Declarative Pipelines**: YAML-based pipeline definitions (Extract, Transform, Evaluate, Load)
- **✅ Built-in Evaluation**: Arbiter integration with quality gates (all_pass, majority_pass, any_pass, weighted)
- **✅ Provider-Agnostic LLMs**: OpenAI, Anthropic, Google, Groq support
- **✅ Multiple Data Formats**: CSV, JSON, JSONL, Parquet support
- **✅ Quality Gates**: Four gate types with precise mathematical definitions
- **✅ Circuit Breaker Pattern**: Production resilience for LLM calls
- **✅ Quarantine Pattern**: Failed records logged with failure reasons for investigation
- **✅ CLI Interface**: `loom run`, `loom validate` commands

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

**Note:** This is a personal project. Roadmap items are ideas and explorations, not commitments. Priorities and timelines may change based on what's useful.

**Phase 1 - Foundation** ✅ (Completed)
- [x] Core pipeline engine (Extract, Transform, Evaluate, Load)
- [x] YAML pipeline parser
- [x] Arbiter integration with quality gates
- [x] Circuit breaker and resilience patterns
- [x] Basic CLI (`loom run`, `loom validate`)

**Future Ideas** (No timeline, exploring as needed)
- [ ] Database connectors (PostgreSQL, MySQL)
- [ ] Cost tracking and monitoring
- [ ] Semantic caching for duplicate inputs
- [ ] Smart retry logic with failure-type awareness
- [ ] Testing framework for pipelines
- [ ] More advanced monitoring and alerting

**Contributions welcome!** This is a personal project, but if you find it useful and want to contribute, pull requests are appreciated.

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

Inspired by dbt's declarative approach to data pipelines and built on top of [Arbiter](https://github.com/evanvolgas/arbiter) for evaluation.
