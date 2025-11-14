# CLAUDE.md - Project Context (Layer 2)

**Purpose:** How AI agents should work with the Loom repository
**Type:** Project Context (Layer 2 of 4-layer context framework)
**Last Updated:** 2025-11-14

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

### Layer 2: Project Context (THIS FILE - CLAUDE.md)
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
- Current phase/milestone tasks
- In-progress items with checkboxes
- Decisions made during work
- Blockers and notes

**Update Frequency:** Daily/weekly during active development
**Note:** Git-ignored (ephemeral session state)

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
- **Project:** "Hard dependency on Arbiter for evaluation"
- **Running:** "Phase 1: Setting up project structure and context"
- **Prompt:** "Create pyproject.toml with dependencies"

---

## Repository Architecture

### Project Status

**Phase:** Design → Implementation
**Version:** 0.1.0-alpha (upcoming)
**Language:** Python 3.10+
**Status:** Initial setup

### Vision

Loom is the **"dbt for AI(E)TL"** - a declarative orchestration framework for AI pipelines.

Traditional ETL becomes AI(E)TL: **Extract → Transform → Evaluate → Load**

### Directory Structure

```
loom/
├── loom/                       # Main package
│   ├── __init__.py             # Public API exports
│   ├── cli/                    # Command-line interface
│   │   ├── __init__.py
│   │   └── main.py             # Click-based CLI
│   ├── core/                   # Core infrastructure
│   │   ├── __init__.py
│   │   ├── config.py           # Configuration management
│   │   ├── exceptions.py       # Custom exceptions
│   │   ├── models.py           # Pydantic data models
│   │   └── types.py            # Type definitions and enums
│   ├── engines/                # AETL engines
│   │   ├── __init__.py
│   │   ├── extract.py          # Extract engine
│   │   ├── transform.py        # Transform engine (AI)
│   │   ├── evaluate.py         # Evaluate engine (Arbiter)
│   │   └── load.py             # Load engine
│   ├── connectors/             # Data source/destination connectors
│   │   ├── __init__.py
│   │   ├── base.py             # Connector interface
│   │   ├── postgres.py         # PostgreSQL connector
│   │   └── s3.py               # S3 connector
│   ├── quality_gates/          # Quality gate implementations
│   │   ├── __init__.py
│   │   ├── base.py             # QualityGate interface
│   │   └── gates.py            # AllPass, MajorityPass, Weighted
│   ├── storage/                # Storage backends
│   │   ├── __init__.py
│   │   ├── base.py             # StorageBackend interface
│   │   ├── postgres.py         # PostgreSQL storage
│   │   └── sqlite.py           # SQLite storage
│   ├── pipeline/               # Pipeline orchestration
│   │   ├── __init__.py
│   │   ├── executor.py         # Pipeline executor
│   │   ├── parser.py           # YAML parser
│   │   └── validator.py        # Pipeline validator
│   └── utils/                  # Utilities
│       ├── __init__.py
│       ├── cache.py            # Caching utilities
│       ├── cost.py             # Cost tracking
│       ├── lineage.py          # Lineage tracking
│       └── retry.py            # Retry logic
├── tests/
│   ├── unit/
│   └── integration/
├── examples/
│   ├── pipelines/              # Example pipeline YAML files
│   └── scripts/                # Example Python scripts
├── docs/                       # Documentation
├── .gitignore
├── CLAUDE.md                   # THIS FILE (Project context)
├── PROJECT_TODO.md             # Running context (git-ignored)
├── README.md                   # User-facing overview
├── DESIGN_SPEC.md              # High-level architecture
├── ARCHITECTURE.md             # Detailed system design
├── IMPLEMENTATION_SPEC.md      # Technical implementation details
├── pyproject.toml              # Project configuration
└── LICENSE                     # MIT License
```

---

## Tech Stack

### Core Technologies
- **Python:** 3.10+ (required for modern type hints, match statements)
- **Pydantic:** 2.12+ (data validation and serialization)
- **Click:** 8.0+ (CLI framework)
- **SQLAlchemy:** 2.0+ (ORM for storage)
- **PyYAML:** 6.0+ (pipeline YAML parsing)
- **Jinja2:** 3.0+ (prompt templating)

### Critical Dependencies
- **Arbiter:** >=0.1.0 (HARD DEPENDENCY - evaluation engine)
  - Direct import: `from arbiter import evaluate`
  - See ARCHITECTURE.md for rationale on hard dependency

### LLM Integration
- Arbiter provides LLM client abstraction
- Supports OpenAI, Anthropic, Google, Groq via Arbiter

### Storage
- **PostgreSQL:** Production storage
- **SQLite:** Local development and testing
- **Redis:** Caching layer (optional)

### Development Tools
- **pytest:** 9.0+ (testing framework)
- **mypy:** 1.18+ (type checking - strict mode)
- **ruff:** 0.14+ (linting and formatting)
- **uv:** Package management and virtual environments

---

## Development Workflow

### 1. Setup

```bash
# Clone repository
git clone https://github.com/evanvolgas/loom.git
cd loom

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# Install in development mode
uv pip install -e ".[dev]"
```

### 2. Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit/

# With coverage
pytest --cov=loom --cov-report=html

# Type checking
mypy loom/

# Linting
ruff check loom/

# Format code
ruff format loom/
```

### 3. Development Pattern

**Every implementation must:**
1. Have type hints (strict mypy)
2. Include tests (>80% coverage)
3. Have docstrings
4. Pass ruff linting
5. Follow async/await patterns for I/O

**Never:**
- Use TODO comments or placeholders
- Leave incomplete functions
- Skip type hints
- Create partial features

---

## Critical Constraints

### 1. Arbiter Hard Dependency (NON-NEGOTIABLE)

**Rule:** Loom has a hard dependency on Arbiter for evaluation.

**Why:** See ARCHITECTURE.md "Why Arbiter as Hard Dependency?" for full rationale.

**Implementation:**
```python
# ✅ CORRECT
from arbiter import evaluate

async def evaluate_output(output, reference, config):
    result = await evaluate(
        output=output,
        reference=reference,
        evaluators=config.evaluators
    )
    return result

# ❌ WRONG - Don't abstract evaluation backends in v1.0
class EvaluationBackend(ABC):  # NO - premature abstraction
    pass
```

### 2. Async/Await Throughout (REQUIRED)

**Rule:** All I/O operations must use async/await.

**Why:** Pipeline operations are I/O-bound (database, LLM APIs). Async enables thousands of concurrent operations efficiently.

**Pattern:**
```python
# ✅ CORRECT
async def extract_records(connector, query):
    async for batch in connector.extract_batches(query):
        yield batch

# ❌ WRONG - Synchronous I/O
def extract_records(connector, query):
    return connector.extract(query)  # Blocks
```

### 3. Type Safety (STRICT)

**Rule:** All functions must have complete type hints. Strict mypy mode.

```toml
[tool.mypy]
strict = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
```

**Example:**
```python
# ✅ CORRECT
async def transform_batch(
    records: List[Record],
    config: TransformConfig
) -> List[TransformResult]:
    ...

# ❌ WRONG - Missing types
async def transform_batch(records, config):
    ...
```

### 4. No Placeholders or TODOs (ENFORCED)

**Rule:** Never leave TODO comments or placeholder implementations.

```python
# ❌ NEVER DO THIS
def calculate_cost(tokens: int) -> float:
    # TODO: implement cost calculation
    raise NotImplementedError()

# ✅ DO THIS
def calculate_cost(tokens: int, cost_per_1k: float = 0.01) -> float:
    """Calculate cost based on token usage."""
    return (tokens / 1000) * cost_per_1k
```

### 5. Complete Features Only (MANDATORY)

**Rule:** If you start implementing a feature, complete it fully.

**Complete means:**
- Implementation finished
- Tests written (>80% coverage)
- Docstrings added
- Type hints complete
- Exported in `__init__.py`

---

## Code Quality Standards

### Docstrings

Follow Google-style docstrings:

```python
async def evaluate_pipeline(
    pipeline: Pipeline,
    params: Dict[str, Any]
) -> PipelineResult:
    """Execute pipeline with AETL orchestration.

    Args:
        pipeline: Pipeline configuration with extract/transform/evaluate/load steps
        params: Runtime parameters for query templating (e.g., execution_date)

    Returns:
        PipelineResult with metrics, costs, and lineage

    Raises:
        PipelineExecutionError: If any AETL step fails
        ValidationError: If pipeline configuration is invalid

    Example:
        >>> pipeline = loom.load_pipeline("customer_sentiment.yaml")
        >>> result = await evaluate_pipeline(
        ...     pipeline=pipeline,
        ...     params={"execution_date": "2024-01-01"}
        ... )
        >>> print(f"Processed {result.records_processed} records")
    """
```

### Error Handling

Use specific exceptions:

```python
# loom/core/exceptions.py
class LoomError(Exception):
    """Base exception for Loom."""
    pass

class PipelineExecutionError(LoomError):
    """Pipeline execution failed."""
    pass

class ConfigurationError(LoomError):
    """Invalid configuration."""
    pass

class ConnectorError(LoomError):
    """Connector operation failed."""
    pass

# Usage
async def execute_pipeline(pipeline):
    if not pipeline.extract:
        raise ConfigurationError("Pipeline missing extract configuration")
```

---

## Design Patterns

### 1. Template Method for Extensibility

Connectors, quality gates, and storage backends use template method pattern:

```python
class DataConnector(ABC):
    """Base class for data connectors."""

    @abstractmethod
    async def connect(self, connection_string: str):
        """Establish connection."""
        pass

    @abstractmethod
    async def extract(self, query: str, batch_size: int) -> ExtractResult:
        """Extract records."""
        pass
```

### 2. Dependency Injection

```python
class PipelineExecutor:
    def __init__(
        self,
        storage: StorageBackend,
        cache: CacheManager,
        metrics: MetricsCollector
    ):
        self.storage = storage
        self.cache = cache
        self.metrics = metrics
```

### 3. Builder Pattern for Configuration

```python
# Programmatic pipeline creation
pipeline = (
    Pipeline.builder()
    .name("customer_sentiment")
    .extract(source="postgres://customers/reviews")
    .transform(prompt="prompts/classify.txt", model="gpt-4o-mini")
    .evaluate(evaluators=["semantic"], threshold=0.8)
    .load(destination="postgres://analytics/scores")
    .build()
)
```

---

## Agent Guidelines

### When Starting Work

1. **Read documentation:**
   - DESIGN_SPEC.md - Understand vision
   - ARCHITECTURE.md - Understand system design
   - PROJECT_TODO.md - Check current phase

2. **Check git status:** Ensure clean working directory

3. **Create feature branch:** `git checkout -b feature/your-feature`

### During Development

1. **Follow async/await patterns:** All I/O operations
2. **Write tests as you code:** Not after
3. **Use type hints:** Strict mypy compliance
4. **Update PROJECT_TODO.md:** Track progress

### Before Committing

1. **Run tests:** `pytest`
2. **Type check:** `mypy loom/`
3. **Lint:** `ruff check loom/`
4. **Format:** `ruff format loom/`
5. **Update exports:** Add to `__init__.py` if public API

---

## Related Documents

### Essential Reading (Priority Order)
1. **DESIGN_SPEC.md** - What we're building and why
2. **ARCHITECTURE.md** - How we're building it
3. **CLAUDE.md** - THIS FILE (how to work here)
4. **IMPLEMENTATION_SPEC.md** - Technical implementation details

### Reference Documents
- **README.md** - User-facing overview
- **PROJECT_TODO.md** - Current work tracking (git-ignored)

---

**Last Updated:** 2025-11-14 | **Next Review:** 2025-12-14
