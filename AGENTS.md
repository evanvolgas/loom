---
name: loom-agent
description: LLM-powered data pipeline framework developer
---

# AGENTS.md - Project Context (Layer 2)

**Purpose:** How AI agents should work with the Loom repository
**Type:** Project Context (Layer 2 of 4-layer context framework)
**Last Updated:** 2025-01-22
**Status:** Phase 1 Complete, Testing In Progress

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

**Current Phase:** Phase 1 Complete, Testing In Progress
**Design:** 100% complete
**Implementation:** ~95% complete (core features done, quality gates pending)
**Test Coverage:** <20% (needs comprehensive test suite)
**Expert Review Score:** 8.7/10

### Completed Work

**Phase 0: Design & Planning**
- ✅ docs/DESIGN_SPEC.md - Vision and high-level architecture
- ✅ docs/ARCHITECTURE.md - Detailed system design with component diagrams
- ✅ docs/IMPLEMENTATION_SPEC.md - Technical details and Phase 1 plan
- ✅ docs/QUALITY_GATES.md - Precise quality gate semantics with examples
- ✅ docs/TIMEOUTS.md - Comprehensive timeout specifications
- ✅ docs/ARBITER_INTEGRATION_ROADMAP.md - Integration timeline
- ✅ Expert specification review with 5 domain experts

**Phase 1: Core Pipeline Engine** (Completed Nov 14, 2024)
- ✅ Python package structure (pyproject.toml with full dependency specification)
- ✅ Core infrastructure (loom/core/: models, exceptions, types, config)
- ✅ Extract engine (loom/engines/extract.py - CSV, JSON, JSONL, Parquet support)
- ✅ Transform engine (loom/engines/transform.py - LLM integration via Arbiter)
- ✅ Evaluate engine (loom/engines/evaluate.py - Arbiter evaluation integration)
- ✅ Load engine (loom/engines/load.py - local file outputs)
- ✅ YAML pipeline parser (loom/parsers/yaml_parser.py)
- ✅ Pipeline runner (loom/runner/pipeline_runner.py with rich progress bars)
- ✅ CLI interface (loom/cli/main.py - run, validate, version commands)
- ✅ Resilience patterns (circuit_breaker.py with comprehensive tests, retry.py)
- ✅ ~1,315 lines of production code across engines, parsers, runner, CLI

### Current Gap: Testing
**What's Missing:**
- ❌ Quality gate implementations (loom/quality_gates/ directory exists but empty)
- ❌ Comprehensive test suite (only test_circuit_breaker.py exists)
- ❌ Integration tests for end-to-end pipeline flows
- ❌ Test fixtures and example pipeline YAMLs
- ❌ <20% test coverage (target: >80%)

### Next Milestone
**Testing & Quality Gate Implementation**
- Implement quality gate classes (AllPassGate, MajorityPassGate, AnyPassGate, WeightedGate)
- Write comprehensive unit tests for all engines
- Write integration tests for complete pipeline flows
- Create test fixtures (sample data files, pipeline YAMLs)
- Achieve >80% test coverage
- Validate all timeout enforcement
- Test circuit breaker integration with engines

---

## Repository Architecture

### Directory Structure

```
loom/
├── loom/                       # Main package ✅
│   ├── __init__.py             # Public API exports ✅
│   ├── core/                   # Core infrastructure ✅
│   │   ├── __init__.py         # ✅
│   │   ├── models.py           # ✅ Pydantic data models (Pipeline, Record, Stage)
│   │   ├── exceptions.py       # ✅ Custom exception hierarchy
│   │   ├── config.py           # ✅ Configuration management
│   │   └── types.py            # ✅ Enums and type definitions
│   ├── engines/                # Pipeline stage engines ✅
│   │   ├── __init__.py         # ✅
│   │   ├── extract.py          # ✅ ExtractEngine (CSV, JSON, JSONL, Parquet)
│   │   ├── transform.py        # ✅ TransformEngine (LLM via Arbiter)
│   │   ├── evaluate.py         # ✅ EvaluateEngine (Arbiter integration)
│   │   └── load.py             # ✅ LoadEngine (local file outputs)
│   ├── parsers/                # Pipeline definition parsers ✅
│   │   ├── __init__.py         # ✅
│   │   └── yaml_parser.py      # ✅ YAML pipeline parser
│   ├── quality_gates/          # ❌ PENDING - Directory exists but empty
│   │   ├── __init__.py         # ✅ Empty file only
│   │   ├── base.py             # ❌ TODO: BaseQualityGate abstract class
│   │   └── gates.py            # ❌ TODO: all_pass, majority_pass, any_pass, weighted
│   ├── resilience/             # ✅ Production resilience patterns
│   │   ├── __init__.py         # ✅
│   │   ├── circuit_breaker.py  # ✅ Circuit breaker with comprehensive tests
│   │   └── retry.py            # ✅ Exponential backoff retry
│   ├── runner/                 # Pipeline execution ✅
│   │   ├── __init__.py         # ✅
│   │   └── pipeline_runner.py  # ✅ Main orchestrator with rich progress
│   ├── cli/                    # Command-line interface ✅
│   │   ├── __init__.py         # ✅
│   │   └── main.py             # ✅ CLI (run, validate, version)
│   ├── connectors/             # ✅ Connector infrastructure (empty, for Phase 2+)
│   ├── pipeline/               # ✅ Pipeline utilities (empty, future use)
│   ├── storage/                # ✅ Storage backends (empty, for Phase 2+)
│   └── utils/                  # ✅ Utility functions (empty, future use)
├── tests/                      # ❌ CRITICAL GAP - Minimal test coverage
│   ├── __init__.py             # ✅
│   ├── unit/                   # ⚠️ Only 1 test file
│   │   ├── __init__.py         # ✅
│   │   └── test_circuit_breaker.py  # ✅ Comprehensive circuit breaker tests
│   ├── integration/            # ❌ Empty - needs end-to-end tests
│   │   └── __init__.py         # ✅
│   └── fixtures/               # ❌ TODO - needs sample data and pipeline YAMLs
│       └── pipelines/          # ❌ TODO - test pipeline definitions
├── examples/                   # ❌ TODO - needs example pipelines
│   └── pipelines/              # ❌ TODO - example YAML definitions
├── docs/                       # Design specifications ✅
│   ├── DESIGN_SPEC.md          # ✅ Vision and architecture
│   ├── ARCHITECTURE.md         # ✅ Detailed system design
│   ├── IMPLEMENTATION_SPEC.md  # ✅ Technical implementation details
│   ├── QUALITY_GATES.md        # ✅ Quality gate semantics
│   ├── TIMEOUTS.md             # ✅ Timeout specifications
│   └── ARBITER_INTEGRATION_ROADMAP.md  # ✅ Arbiter integration timeline
├── pyproject.toml              # ✅ Complete with all dependencies
├── README.md                   # ✅ Project overview
├── LICENSE                     # ✅ MIT License
├── AGENTS.md                   # ✅ THIS FILE (Layer 2 context)
├── PROJECT_TODO.md             # Current tasks (Layer 3, git-ignored)
└── CONTRIBUTING.md             # ❌ TODO - Contribution guidelines
```

### Module Responsibilities

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

### Core Technologies ✅
- **Python:** 3.10-3.13 (modern type hints, async/await)
- **Pydantic:** 2.12+ (data validation and serialization) ✅
- **PyYAML:** 6.0+ (YAML parsing) ✅
- **Click:** 8.0+ (CLI framework) ✅
- **Rich:** 14.0+ (CLI formatting and progress bars) ✅
- **Arbiter:** Latest via git (evaluation engine - hard dependency) ✅

### LLM Integration (Via Arbiter) ✅
- **Provider-agnostic:** OpenAI, Anthropic, Google, Groq ✅
- **PydanticAI:** Structured outputs (via Arbiter) ✅
- **HTTPX:** 0.28+ Async HTTP client ✅
- **python-dotenv:** 1.0+ Environment variable management ✅

### Data Formats (Phase 1) ✅
- **Polars:** 0.19+ (DataFrame processing - faster than pandas) ✅
- **PyArrow:** 14.0+ (Parquet support) ✅
- **aiofiles:** 25.0+ (Async file I/O) ✅
- **Supported formats:** CSV, JSON, JSONL, Parquet ✅

### Development Tools ✅
- **pytest:** 9.0+ (testing framework) ✅
- **pytest-asyncio:** 1.0+ (async test support) ✅
- **pytest-cov:** 7.0+ (coverage reporting) ✅
- **pytest-mock:** 3.15+ (mocking utilities) ✅
- **black:** 25.0+ (code formatting) ✅
- **ruff:** 0.14+ (linting) ✅
- **mypy:** 1.18+ (type checking - strict mode) ✅
- **uv:** Recommended (package management - faster than pip)

### Future Dependencies (Post-Phase 1)
- **PostgreSQL:** Database source/destination (Phase 2)
- **Redis:** Caching layer (Phase 3)
- **Prometheus:** Metrics export (Phase 3)
- **ByteWax:** Streaming support (Phase 7)

---

## Boundaries

### ✅ Always Do (No Permission Needed)
- Run tests: `pytest`, `pytest --cov=loom`, `pytest -v`
- Format code: `black loom/`
- Lint code: `ruff check loom/`
- Type check: `mypy loom/` (strict mode required)
- Add unit tests for new engines in `tests/unit/`
- Add integration tests for pipeline flows in `tests/integration/`
- Update docstrings when changing function signatures
- Export new components in `__init__.py` files
- Add test fixtures for new data formats in `tests/fixtures/`
- Create example pipelines in `examples/pipelines/`

### ⚠️ Ask First
- Add new engines to `loom/engines/`
- Modify core models in `loom/core/models.py` (Pipeline, Record, Stage)
- Change quality gate logic in `loom/quality_gates/` (must match docs/QUALITY_GATES.md)
- Add/update dependencies in `pyproject.toml`
- Modify timeout specifications (must align with docs/TIMEOUTS.md)
- Change circuit breaker configuration in `loom/resilience/circuit_breaker.py`
- Add new storage backends in `loom/storage/`
- Modify YAML pipeline parser in `loom/parsers/yaml_parser.py`
- Change CLI commands in `loom/cli/main.py`
- Update specification documents in `docs/` (DESIGN_SPEC.md, ARCHITECTURE.md, etc.)

### 🚫 Never Touch

**CRITICAL SECURITY VIOLATION** ⚠️:
- **NEVER EVER COMMIT CREDENTIALS TO GITHUB**
- No API keys, tokens, passwords, secrets in ANY file
- No credentials in code, documentation, examples, tests, or configuration files
- Use environment variables (.env files in .gitignore) ONLY
- This is NON-NEGOTIABLE - violating this rule has serious security consequences

**Other Prohibitions**:
- `.env` files or API keys (use environment variables)
- Production deployment configurations
- Git history manipulation (no force push, interactive rebase on shared branches)
- User's `~/.claude/` configuration files
- Any files outside the `loom/` repository
- Specification documents without PR review (docs/DESIGN_SPEC.md, docs/QUALITY_GATES.md, etc.)
- Test files to make them pass (fix the code, not the tests)
- Type checking strictness settings in `pyproject.toml`
- Coverage thresholds (must maintain >80%)
- Arbiter dependency implementation (hard dependency, non-negotiable)

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
# All tests with coverage (requires >80%, shows missing lines)
pytest --cov=loom --cov-report=term-missing

# Specific test file with verbose output
pytest tests/unit/test_extract.py -v

# Fast unit tests only (mocked dependencies, <1s per test)
pytest tests/unit/

# Integration tests only (end-to-end pipeline flows)
pytest tests/integration/

# Verbose output (shows test names and pass/fail status)
pytest -v

# Run tests matching specific pattern
pytest -k "test_circuit_breaker" -v

# Quiet mode (minimal output, only failures)
pytest -q
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

### Current Version: 0.0.1 (Pre-Alpha)

**Status:** Phase 1 implementation complete, comprehensive testing needed

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

## Working with AI Agents

### Task Management
**TodoWrite enforcement (MANDATORY)**: For ANY task with 3+ distinct steps, use TodoWrite to track progress - even if the user doesn't request it explicitly. This ensures nothing gets forgotten and provides visibility into progress for everyone working on the project.

**Plan before executing**: For complex tasks, create a plan first. Understand requirements, identify dependencies, then execute systematically.

### Output Quality
**Full data display**: Show complete data structures, not summaries or truncations. Examples should display real, useful output (not "[truncated]" or "...").

**Debugging context**: When showing debug output, include enough detail to actually debug - full prompts, complete responses, actual data structures. Truncating output defeats the purpose.

**Verify usefulness**: Before showing output, verify it's actually helpful for the user's goal. Test that examples demonstrate real functionality, not abstractions.

### Audience & Context Recognition
**Auto-detect technical audiences**: Code examples, technical docs, developer presentations → eliminate ALL marketing language automatically. Engineering contexts get technical tone (no superlatives like "blazingly fast", "magnificent", "revolutionary").

**Recognize audience immediately**: Engineers get technical tone, no marketing language. Business audiences get value/ROI focus. Academic audiences get methodology and rigor. Adapt tone and content immediately based on context.

**Separate material types**: Code examples stay clean (no narratives or marketing). Presentation materials (openers, talking points) live in separate files. Documentation explains architecture and usage patterns.

### Quality & Testing
**Test output quality, not just functionality**: Run code AND verify the output is actually useful. Truncated or abstracted output defeats the purpose of examples. Show real data structures, not summaries.

**Verify before committing**: Run tests and verify examples work before showing output. Test both functionality and usefulness.

**Connect work to strategy**: Explicitly reference project milestones, coverage targets, and strategic priorities when completing work. Celebrate milestones when achieved.

### Workflow Patterns
**Iterate fast**: Ship → test → get feedback → fix → commit. Don't perfect upfront. Progressive refinement beats upfront perfection.

**Proactive problem solving**: Use tools like Glob to check file existence before execution. Anticipate common issues and handle them gracefully.

**Parallel execution**: Batch independent operations (multiple reads, parallel test execution) to improve efficiency.

### Communication & Feedback
**Direct feedback enables fast iteration**: Clear, immediate feedback on what's wrong enables rapid course correction. Specific, actionable requests work better than vague suggestions.

**Match user communication style**: Some users prefer speed over process formality, results over explanations. Adapt communication style accordingly while maintaining quality standards.

### Git & Commit Hygiene
**Commit hygiene**: Each meaningful change gets its own commit with clear message (what + why). This makes progress tracking and rollback easier.

**Clean git workflow**: Always check `git status` and `git branch` before operations. Use feature branches for all changes.

---

**Last Updated:** 2025-01-22 | **Next Review:** After comprehensive test suite implementation
