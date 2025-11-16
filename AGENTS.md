# AGENTS.md - Project Context (Layer 2)

**Purpose:** How AI agents should work with the Loom repository
**Type:** Project Context (Layer 2 of 4-layer context framework)
**Last Updated:** 2025-11-14
**Status:** Design Complete, Implementation Starting

---

## Four-Layer Context Framework

This repository uses a four-layer context system for AI agent interactions:

### Layer 1: Global Context (~/.claude/CLAUDE.md)
**What:** Universal rules applying across ALL projects
**Contains:**
- Communication preferences
- General expertise areas
- Things AI should never do
- Known gotchas

**Update Frequency:** Rarely (monthly or less)

### Layer 2: Project Context (THIS FILE - AGENTS.md)
**What:** Repository-specific rules and architecture
**Contains:**
- Tech stack and architecture
- Development workflow
- Critical constraints
- Code ownership
- Repository structure

**Update Frequency:** When architecture changes (monthly)

### Layer 3: Running Context (PROJECT_TODO.md)
**What:** Current session state and active tasks
**Contains:**
- Current phase and milestone tasks
- In-progress items with checkboxes
- Decisions made during work
- Blockers and notes

**Update Frequency:** Daily/weekly during active development

### Layer 4: Prompt Context
**What:** The immediate, specific request
**Contains:**
- Single-use instruction
- Builds on layers 1-3

**Update Frequency:** Every interaction

### How They Work Together

Each layer constrains the next:
```
Global → Project → Running → Prompt
  ↓        ↓         ↓         ↓
Rules → Architecture → Status → Request
```

**Example:**
- **Global:** "Never use placeholders or TODO comments"
- **Project:** "Use Arbiter for all evaluation logic"
- **Running:** "Currently implementing ExtractEngine"
- **Prompt:** "Add CSV file support to ExtractEngine"

---

## Project Status

**Current Phase:** Pre-Phase 1 (Design Complete, Ready for Implementation)
**Design:** 100% complete
**Implementation:** 0% complete
**Expert Review Score:** 8.7/10

### Completed Work
- ✅ docs/DESIGN_SPEC.md - Vision and high-level architecture
- ✅ docs/ARCHITECTURE.md - Detailed system design with component diagrams
- ✅ docs/IMPLEMENTATION_SPEC.md - Technical details and Phase 1 plan
- ✅ docs/QUALITY_GATES.md - Precise quality gate semantics with examples
- ✅ docs/TIMEOUTS.md - Comprehensive timeout specifications
- ✅ Circuit breaker pattern implementation (loom/resilience/)
- ✅ Expert specification review with 5 domain experts

### Next Milestone
**Phase 1: Core Pipeline Engine**
- Bootstrap Python package structure
- Implement Extract/Transform/Evaluate/Load engines
- YAML pipeline parser
- Arbiter integration
- Quality gate logic
- CLI interface

---

## Repository Architecture

### Directory Structure (Planned)

```
loom/
├── loom/                       # Main package
│   ├── __init__.py             # Public API exports
│   ├── core/                   # Core infrastructure
│   │   ├── __init__.py
│   │   ├── models.py           # Pydantic data models (Pipeline, Record, Stage)
│   │   ├── exceptions.py       # Custom exception hierarchy
│   │   ├── config.py           # Configuration management
│   │   └── types.py            # Enums and type definitions
│   ├── engines/                # Pipeline stage engines
│   │   ├── __init__.py
│   │   ├── extract.py          # ExtractEngine (Phase 1: local files)
│   │   ├── transform.py        # TransformEngine (LLM integration)
│   │   ├── evaluate.py         # EvaluateEngine (Arbiter integration)
│   │   └── load.py             # LoadEngine (Phase 1: local files)
│   ├── parsers/                # Pipeline definition parsers
│   │   ├── __init__.py
│   │   └── yaml_parser.py      # YAML pipeline parser
│   ├── quality_gates/          # Quality gate implementations
│   │   ├── __init__.py
│   │   ├── base.py             # BaseQualityGate
│   │   └── gates.py            # all_pass, majority_pass, any_pass, weighted
│   ├── resilience/             # ✅ Production resilience patterns
│   │   ├── __init__.py
│   │   ├── circuit_breaker.py  # Circuit breaker implementation
│   │   └── retry.py            # Retry with exponential backoff
│   ├── runner/                 # Pipeline execution
│   │   ├── __init__.py
│   │   └── pipeline_runner.py  # Main pipeline orchestrator
│   └── cli/                    # Command-line interface
│       ├── __init__.py
│       └── main.py             # CLI entry point
├── tests/
│   ├── unit/
│   │   └── test_circuit_breaker.py  # ✅ Comprehensive circuit breaker tests
│   ├── integration/
│   └── fixtures/
│       └── pipelines/          # Test pipeline YAML files
├── examples/
│   └── pipelines/              # Example pipeline definitions
├── docs/                       # Documentation (future: MkDocs)
├── pyproject.toml              # Project configuration (to be created)
├── README.md                   # ✅ Project overview
├── AGENTS.md                   # THIS FILE (Layer 2)
├── PROJECT_TODO.md             # Current tasks (Layer 3, git-ignored)
├── docs/                       # Documentation (design specs)
│   ├── DESIGN_SPEC.md          # ✅ Vision and architecture
│   ├── ARCHITECTURE.md         # ✅ Detailed system design
│   ├── IMPLEMENTATION_SPEC.md  # ✅ Technical implementation details
│   ├── QUALITY_GATES.md        # ✅ Quality gate semantics
│   ├── TIMEOUTS.md             # ✅ Timeout specifications
│   └── ARBITER_INTEGRATION_ROADMAP.md  # ✅ Arbiter integration timeline
└── CONTRIBUTING.md             # Contribution guidelines (future)
```

### Module Responsibilities (Planned)

#### `core/` - Foundation
**Purpose:** Core data models and infrastructure
**Components:**
- **models.py:** Pipeline, Record, Stage, Config (Pydantic models)
- **exceptions.py:** LoomError hierarchy
- **config.py:** Configuration management
- **types.py:** Enums (StageType, QualityGateType, SourceType)

#### `engines/` - Stage Implementations
**Purpose:** Extract/Transform/Evaluate/Load logic
**Pattern:** Each engine implements a common interface
**Phase 1 Scope:**
- ExtractEngine: Local files only (CSV, JSON, Parquet)
- TransformEngine: LLM API calls with prompt templates
- EvaluateEngine: Arbiter integration with quality gates
- LoadEngine: Local files only (CSV, JSON, Parquet)

#### `parsers/` - Pipeline Definitions
**Purpose:** Parse YAML pipeline definitions
**Components:**
- yaml_parser.py: Parse and validate pipeline YAML

#### `quality_gates/` - Evaluation Gates
**Purpose:** Implement quality gate logic from docs/QUALITY_GATES.md
**Components:**
- BaseQualityGate: Abstract interface
- AllPassGate, MajorityPassGate, AnyPassGate, WeightedGate

#### `resilience/` - Production Patterns ✅
**Purpose:** Circuit breaker, retry, timeout enforcement
**Status:** Implemented
**Components:**
- circuit_breaker.py: Three-state circuit breaker
- retry.py: Exponential backoff retry

#### `runner/` - Pipeline Orchestration
**Purpose:** Execute complete pipelines
**Components:**
- PipelineRunner: Main orchestrator coordinating all engines

#### `cli/` - User Interface
**Purpose:** Command-line interface
**Commands:**
- `loom run <pipeline>`: Run a pipeline
- `loom validate <pipeline>`: Validate pipeline YAML
- `loom version`: Show version

---

## Tech Stack

### Core Technologies (To Be Implemented)
- **Python:** 3.10+ (required for modern type hints)
- **Pydantic:** 2.12+ (data validation and serialization)
- **PyYAML:** 6.0+ (YAML parsing)
- **Click:** 8.0+ (CLI framework)
- **Arbiter:** Latest (evaluation engine - hard dependency)

### LLM Integration (Via Arbiter)
- **Provider-agnostic:** OpenAI, Anthropic, Google, Groq
- **PydanticAI:** Structured outputs
- **HTTPX:** Async HTTP client

### Data Formats (Phase 1)
- **Polars:** 0.19+ (DataFrame processing - faster than pandas)
- **PyArrow:** 14.0+ (Parquet support)

### Development Tools
- **pytest:** 9.0+ (testing framework)
- **pytest-asyncio:** 1.0+ (async test support)
- **black:** 25.0+ (code formatting)
- **ruff:** 0.14+ (linting)
- **mypy:** 1.18+ (type checking - strict mode)
- **uv:** Latest (package management - faster than pip)

### Future Dependencies (Post-Phase 1)
- **PostgreSQL:** Database source/destination (Phase 2)
- **Redis:** Caching layer (Phase 3)
- **Prometheus:** Metrics export (Phase 3)
- **ByteWax:** Streaming support (Phase 7)

---

## Development Workflow

### 1. Feature Development

#### Branch Strategy
```bash
# Always work on feature branches
git checkout -b feature/extract-engine

# Never commit to main directly
# Use PRs for all changes
```

#### Implementation Flow
1. **Read Context:**
   - docs/DESIGN_SPEC.md (understand vision)
   - docs/ARCHITECTURE.md (understand system design)
   - docs/IMPLEMENTATION_SPEC.md (implementation details)
   - PROJECT_TODO.md (check current milestone)

2. **Plan:**
   - What are we building?
   - What's the interface/protocol?
   - What tests are needed?
   - Update PROJECT_TODO.md with tasks

3. **Implement:**
   - Follow Pydantic model patterns
   - Add type hints (strict mypy)
   - Write tests (>80% coverage)
   - Update __init__.py exports

4. **Validate:**
   - Run tests: `pytest`
   - Check types: `mypy loom/`
   - Lint code: `ruff check loom/`
   - Format: `black loom/`

5. **Document:**
   - Add docstrings
   - Update examples/
   - Update PROJECT_TODO.md

### 2. Testing Requirements

**Minimum Coverage:** 80% (strict)

**Test Types:**
1. **Unit Tests** (tests/unit/)
   - Test individual functions and classes
   - Mock external dependencies
   - Fast execution (<1s per test)

2. **Integration Tests** (tests/integration/)
   - Test end-to-end pipeline flows
   - Use real Arbiter calls (with mocking option)
   - Test quality gate logic
   - Validate timeout enforcement

3. **Fixture Tests** (tests/fixtures/pipelines/)
   - Test example pipeline YAML files
   - Validate pipeline definitions

**Running Tests:**
```bash
# All tests with coverage
pytest --cov=loom --cov-report=term-missing

# Specific test file
pytest tests/unit/test_extract.py

# Fast (unit only)
pytest tests/unit/

# With verbose output
pytest -v
```

### 3. Code Quality Standards

#### Type Safety (CRITICAL)
```toml
[tool.mypy]
strict = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
```

**Rules:**
- ✅ All functions must have type hints
- ✅ All parameters must be typed
- ✅ Return types must be explicit
- ❌ No `Any` without explicit justification

**Example:**
```python
# ✅ GOOD
async def extract_records(
    source_config: SourceConfig,
    timeout: float = 30.0
) -> List[Record]:
    ...

# ❌ BAD
async def extract_records(source_config, timeout=30.0):  # Missing types
    ...
```

#### Code Formatting
- **black:** Line length 88 (default)
- **ruff:** Follow configuration in pyproject.toml
- Run `black loom/` before committing

#### Docstrings
```python
async def extract_records(
    source_config: SourceConfig,
    timeout: float = 30.0
) -> List[Record]:
    """Extract records from configured source.

    Args:
        source_config: Source configuration (path, format, etc.)
        timeout: Maximum extraction time in seconds

    Returns:
        List of extracted records

    Raises:
        ExtractError: If extraction fails
        TimeoutError: If extraction exceeds timeout

    Example:
        >>> config = SourceConfig(path="data.csv", format="csv")
        >>> records = await extract_records(config)
        >>> print(len(records))
        1000
    """
```

---

## Critical Constraints

### 1. Hard Dependency on Arbiter (NON-NEGOTIABLE)

**Rule:** Arbiter is a direct Python import dependency, NOT a pluggable evaluation backend

**Rationale:**
- Owned by same author
- Ensures tight integration
- Simplifies architecture
- Version alignment

**Implementation:**
```python
# ✅ GOOD
from arbiter import evaluate
from arbiter.evaluators import SemanticEvaluator

result = await evaluate(
    output=transformed_text,
    reference=reference_text,
    evaluators=["semantic"]
)

# ❌ BAD
# Don't create pluggable evaluation abstraction
# Don't use other evaluation frameworks
```

### 2. Provider-Agnostic LLM (via Arbiter)

**Rule:** Must work with ANY LLM provider through Arbiter's abstraction

**Implementation:**
- Use Arbiter's LLMClient
- Test with multiple providers
- No OpenAI-specific assumptions

**Example:**
```python
# Arbiter handles provider abstraction
result = await evaluate(
    output=text,
    model="gpt-4o-mini",  # or "claude-3-5-sonnet", "gemini-1.5-pro"
    evaluators=["semantic"]
)
```

### 3. Quality Gates from docs/QUALITY_GATES.md (REQUIRED)

**Rule:** All quality gate implementations must match docs/QUALITY_GATES.md specifications

**Implementation:**
- Exact mathematical definitions
- Precise pass/fail logic
- Detailed failure messages

**Pattern:**
```python
class AllPassQualityGate(QualityGate):
    """All evaluators must pass their thresholds."""

    def check_record(self, evaluations: List[EvaluatorResult]) -> bool:
        """Pass condition: ∀ evaluator: score ≥ threshold"""
        return all(e.passed for e in evaluations)

    def get_failure_reason(self, evaluations: List[EvaluatorResult]) -> str:
        failed = [e for e in evaluations if not e.passed]
        if len(failed) == 1:
            e = failed[0]
            return f"{e.evaluator_name} below threshold ({e.score:.2f} < {e.threshold})"
        return f"{len(failed)} evaluators below threshold"
```

### 4. Timeout Enforcement from docs/TIMEOUTS.md (REQUIRED)

**Rule:** All external operations must enforce timeouts from docs/TIMEOUTS.md

**Implementation:**
- LLM calls: 30-60s (model-dependent)
- Database queries: 3-30s
- File I/O: 5-60s
- Use asyncio.timeout() for enforcement

**Pattern:**
```python
async def transform_with_llm(
    record: Record,
    prompt: str,
    model: str = "gpt-4o-mini"
) -> str:
    timeout = TIMEOUTS["llm"][model]["total"]  # 35s for gpt-4o-mini

    try:
        async with asyncio.timeout(timeout):
            response = await llm_client.generate(prompt=prompt)
            return response.text
    except asyncio.TimeoutError:
        raise TransformError(f"LLM call exceeded {timeout}s timeout")
```

### 5. Circuit Breaker for LLM Calls (REQUIRED)

**Rule:** All LLM API calls must use circuit breaker to prevent cascading failures

**Implementation:** Use loom/resilience/circuit_breaker.py

**Pattern:**
```python
from loom.resilience import CircuitBreaker, CircuitBreakerConfig

llm_breaker = CircuitBreaker(
    name="transform_llm",
    config=CircuitBreakerConfig(
        failure_threshold=5,
        timeout_seconds=60,
        success_threshold=3
    )
)

async def transform_record(record: Record) -> str:
    try:
        response = await llm_breaker.call(
            llm_client.generate,
            prompt=build_prompt(record)
        )
        return response.text
    except CircuitBreakerOpenError:
        # Fall back to cached response or skip
        return get_cached_or_skip(record)
```

### 6. No Partial Features (STRICT)

**Rule:** If you start a feature, complete it fully

**Complete means:**
- ✅ Implementation finished
- ✅ Tests written (>80% coverage)
- ✅ Docstrings added
- ✅ Example usage provided
- ✅ Exported in __init__.py
- ✅ PROJECT_TODO.md updated

### 7. No Placeholders or TODOs (STRICT)

**Rule:** Never leave `TODO`, `FIXME`, or placeholder implementations

**Example:**
```python
# ❌ NEVER DO THIS
def parse_yaml(path: str) -> Pipeline:
    # TODO: implement YAML parsing
    raise NotImplementedError("Coming soon")

# ✅ DO THIS
def parse_yaml(path: str) -> Pipeline:
    """Parse pipeline definition from YAML file."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    return Pipeline.model_validate(data)
```

---

## Code Ownership

### Specifications (Read-Only, Change via PR)
- docs/DESIGN_SPEC.md
- docs/ARCHITECTURE.md
- docs/IMPLEMENTATION_SPEC.md
- docs/QUALITY_GATES.md
- docs/TIMEOUTS.md

### Core Infrastructure (loom/core/, loom/resilience/)
**Owner:** Architecture decisions
**Review:** Required for changes
**Critical:** Affects all pipeline components

### Engines (loom/engines/)
**Owner:** Pipeline stage logic
**Review:** Recommended
**Extensible:** Easy to add new sources/destinations

### Quality Gates (loom/quality_gates/)
**Owner:** Evaluation logic
**Review:** Required (must match docs/QUALITY_GATES.md)
**Critical:** Core feature

### CLI (loom/cli/)
**Owner:** User-facing interface
**Review:** Recommended
**Stability:** High - breaking changes avoided

---

## Agent Guidelines

### When Starting Work

1. **Read specifications:**
   - docs/DESIGN_SPEC.md (vision)
   - docs/ARCHITECTURE.md (system design)
   - docs/IMPLEMENTATION_SPEC.md (technical details)
   - PROJECT_TODO.md (current tasks)

2. **Check git status:**
   - `git status` - ensure clean working directory
   - `git branch` - ensure on feature branch

3. **Create feature branch:**
   - `git checkout -b feature/your-feature`

### During Development

1. **Follow specifications:**
   - Quality gates match docs/QUALITY_GATES.md
   - Timeouts match docs/TIMEOUTS.md
   - Use circuit breaker for LLM calls

2. **Write tests as you code** (not after)
3. **Run tests frequently:** `pytest`
4. **Update PROJECT_TODO.md** with progress
5. **Document decisions** in commit messages

### Before Committing

1. **Run full test suite:** `pytest --cov=loom`
2. **Check type safety:** `mypy loom/`
3. **Lint code:** `ruff check loom/`
4. **Format code:** `black loom/`
5. **Update __init__.py** exports if needed
6. **Update PROJECT_TODO.md** checkbox

### After Completing Feature

1. **Mark task complete** in PROJECT_TODO.md
2. **Add example** to examples/ if user-facing
3. **Update README.md** if API changed
4. **Create PR** with clear description

---

## Common Patterns (To Be Established)

### Pattern 1: Adding a New Source Type

*To be documented after Phase 1 implementation*

### Pattern 2: Adding a New Quality Gate

*To be documented based on docs/QUALITY_GATES.md implementations*

### Pattern 3: Pipeline Definition

```yaml
# Example pipeline structure
name: customer_sentiment
version: 1.0.0

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
  quality_gate: all_pass

load:
  destination: postgres://analytics/sentiment_scores
```

---

## Troubleshooting

### Issue: Type Errors

**Problem:** `mypy` reports type errors

**Solution:**
1. Run `mypy loom/` to see errors
2. Add missing type hints
3. Use `cast()` for runtime type narrowing
4. Check pyproject.toml mypy config

### Issue: Import Errors

**Problem:** `ImportError` or circular imports

**Solution:**
1. Check __init__.py exports
2. Use `if TYPE_CHECKING:` for type-only imports
3. Avoid circular dependencies

### Issue: Tests Failing

**Problem:** Tests fail unexpectedly

**Solution:**
1. Run `pytest -v` for verbose output
2. Check if you're using async/await correctly
3. Verify mocks are set up properly
4. Ensure test isolation (no shared state)

---

## Versioning & Releases

### Current Version: 0.0.0 (Pre-Alpha)

**Status:** Design complete, implementation not started

**Semantic Versioning:** MAJOR.MINOR.PATCH

- **MAJOR:** Breaking API changes
- **MINOR:** New features (backward compatible)
- **PATCH:** Bug fixes

### Release Checklist (Future)

Before releasing:
- [ ] All tests pass
- [ ] Type checking clean
- [ ] Linting clean
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in pyproject.toml
- [ ] Git tag created

---

## Related Documents

### Essential Reading (Priority Order)
1. **docs/DESIGN_SPEC.md** - What we're building and why (vision)
2. **docs/ARCHITECTURE.md** - How the system works (design)
3. **docs/IMPLEMENTATION_SPEC.md** - Technical implementation details
4. **PROJECT_TODO.md** - Current milestone tasks (git-ignored)
5. **AGENTS.md** - THIS FILE (how to work here)

### Specification Documents
- **docs/QUALITY_GATES.md** - Precise quality gate semantics with examples
- **docs/TIMEOUTS.md** - Comprehensive timeout specifications
- **docs/ARBITER_INTEGRATION_ROADMAP.md** - Arbiter integration timeline
- **loom/resilience/** - Circuit breaker and retry implementations

### Future Documents
- **CONTRIBUTING.md** - Contribution workflow (to be created)
- **CHANGELOG.md** - Version history (to be created)

### External References
- [Arbiter](https://github.com/evanvolgas/arbiter) - Evaluation engine
- [PydanticAI Docs](https://ai.pydantic.dev/)
- [Pydantic Docs](https://docs.pydantic.dev/)
- [Polars Docs](https://pola-rs.github.io/polars/)

---

## Questions?

If you're unsure about:
- **Architecture:** Read docs/DESIGN_SPEC.md and docs/ARCHITECTURE.md
- **Current work:** Check PROJECT_TODO.md
- **Quality gates:** See docs/QUALITY_GATES.md
- **Timeouts:** See docs/TIMEOUTS.md
- **Circuit breaker:** See loom/resilience/circuit_breaker.py

**Still unclear?** Open an issue or ask in the PR.

---

**Last Updated:** 2025-11-14 | **Next Review:** When Phase 1 implementation begins
