# Contributing to Loom

Thank you for your interest in contributing to Loom! This document provides guidelines for contributing to the declarative AI pipeline orchestration framework.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Code Quality](#code-quality)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

Be respectful and professional. We're building production-grade pipeline orchestration together.

## Getting Started

### Prerequisites

- Python 3.10+
- At least one LLM provider API key (OpenAI, Anthropic, Google, Groq)
- Arbiter (evaluation engine, hard dependency)

### Development Setup

```bash
# Clone the repository
git clone https://github.com/evanvolgas/loom.git
cd loom

# Install dependencies
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Set up environment variables
# Add your LLM provider API keys

# Run tests to verify setup
pytest tests/unit/test_circuit_breaker.py -v
```

## Making Changes

### Before You Start

1. **Read specifications** - Check docs/DESIGN_SPEC.md, docs/ARCHITECTURE.md, docs/IMPLEMENTATION_SPEC.md
2. **Check existing issues** - Look for related issues or discussions
3. **Create an issue first** - For significant changes, discuss your approach
4. **Create a feature branch** - Never work directly on `main`

```bash
git checkout -b feature/your-feature-name
```

### Code Standards

- **Type hints required** - All functions must have complete type annotations (strict mypy)
- **Async/await patterns** - All pipeline stages must be async
- **Docstrings required** - Use Google-style docstrings with Args, Returns, Raises, Example
- **No placeholders** - No TODO comments or NotImplementedError in production code
- **Complete features** - Finish what you start (implementation + tests + docs + examples)

### Pipeline Engine Pattern

All engines must follow the common interface:

```python
class MyEngine:
    async def execute(self, config: dict, records: List[Record]) -> List[Record]:
        """Execute pipeline stage.
        
        Args:
            config: Stage configuration from YAML
            records: Input records to process
            
        Returns:
            Processed records
            
        Raises:
            EngineError: If execution fails
        """
        pass
```

### Quality Gate Pattern

Quality gates must follow docs/QUALITY_GATES.md specifications:

```python
class MyQualityGate(QualityGate):
    def check_record(self, evaluations: List[EvaluatorResult]) -> bool:
        """Pass condition with exact mathematical definition.
        
        Args:
            evaluations: Arbiter evaluation results
            
        Returns:
            True if record passes quality gate
        """
        pass
        
    def get_failure_reason(self, evaluations: List[EvaluatorResult]) -> str:
        """Detailed failure message for quarantine logs."""
        pass
```

## Testing

### Running Tests

```bash
# All tests with coverage (target: >80%)
pytest --cov=loom --cov-report=term-missing

# Specific test file
pytest tests/unit/test_circuit_breaker.py -v

# Fast unit tests only
pytest tests/unit/

# Integration tests (end-to-end pipelines)
pytest tests/integration/
```

### Test Requirements

- **Coverage** - Maintain >80% coverage for all new code
- **Unit tests** - Test individual engines and quality gates
- **Integration tests** - Test complete pipeline flows
- **Mock Arbiter calls** - Don't hit real evaluation APIs in unit tests
- **Use temporary files** - Use pytest.tmp_path for file I/O

## Code Quality

### Pre-Commit Checklist

Run these commands before every commit:

```bash
# 1. Run tests with coverage
pytest --cov=loom --cov-report=term-missing
# Exit if failed or coverage <80%

# 2. Type checking clean
mypy loom/
# Exit if failed

# 3. Linting clean
ruff check loom/
# Exit if failed

# 4. Formatted
black loom/

# 5. No TODOs or placeholders
grep -r "TODO\|FIXME\|NotImplementedError" loom/ && exit 1

# 6. No credentials
grep -r "API_KEY\|SECRET\|PASSWORD" loom/ tests/ examples/ && exit 1
```

### Required Tools

- **black** - Code formatting (line length 88)
- **ruff** - Fast linting
- **mypy** - Strict type checking
- **pytest** - Testing framework with pytest-asyncio

## Pull Request Process

### Before Submitting

1. **Update tests** - Add tests for new features or bug fixes
2. **Update documentation** - Update README.md if API changed
3. **Add examples** - Create example pipeline YAML in `examples/pipelines/`
4. **Validate against specs** - Ensure quality gates match docs/QUALITY_GATES.md
5. **Run all checks** - Ensure all pre-commit checks pass

### PR Requirements

- **Clear description** - Explain what and why (not just what)
- **Reference issues** - Link to related issues
- **One feature per PR** - Keep PRs focused and reviewable
- **Passing CI** - All checks must pass
- **No merge conflicts** - Rebase on main if needed

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New engine (Extract/Transform/Evaluate/Load)
- [ ] New quality gate
- [ ] Enhancement
- [ ] Documentation

## Testing
- [ ] Added unit tests
- [ ] Added integration tests
- [ ] All tests pass
- [ ] Coverage >80%

## Specification Compliance (if applicable)
- [ ] Quality gates match docs/QUALITY_GATES.md
- [ ] Timeouts match docs/TIMEOUTS.md
- [ ] Circuit breaker used for LLM calls
- [ ] Complete observability (audit trail)

## Checklist
- [ ] Code follows style guidelines (black, ruff, mypy pass)
- [ ] Added/updated docstrings
- [ ] Updated README.md if needed
- [ ] Added example pipeline YAML if needed
- [ ] Updated __init__.py exports
- [ ] No credentials in code
```

## Issue Guidelines

### Bug Reports

Use the bug report template. Include:
- **Pipeline YAML** - Your pipeline definition
- **Data format** - Input data format (CSV, JSON, etc.)
- **Stage where error occurred** - Extract, Transform, Evaluate, or Load
- **Expected vs actual behavior**
- **Error messages and quarantine logs**
- **Minimal reproduction example**

### Feature Requests

Use the feature request template. Include:
- **Pipeline use case** - What workflow does this enable?
- **Proposed solution** - How should it work in YAML?
- **Impact on quality gates** - How does this affect evaluation?
- **Alternatives considered** - What else did you consider?

### Questions

For questions about usage:
- Check README.md and docs/ first
- Search existing issues
- Review examples/pipelines/ for patterns
- Provide context about your pipeline scenario

## What We Won't Build

To set clear expectations:

- **Pluggable evaluation backends** - Arbiter is hard dependency (non-negotiable)
- **Support for Python <3.10** - Modern type hints and async/await required
- **Hosted service** - Loom is self-hosted only
- **Non-declarative pipelines** - YAML pipeline definitions required

## Additional Resources

- **README.md** - User documentation and quickstart
- **AGENTS.md** - Detailed development guide (AI-focused)
- **docs/DESIGN_SPEC.md** - Vision and architecture
- **docs/ARCHITECTURE.md** - System design
- **docs/IMPLEMENTATION_SPEC.md** - Technical implementation
- **docs/QUALITY_GATES.md** - Quality gate specifications
- **docs/TIMEOUTS.md** - Timeout specifications
- **examples/pipelines/** - Example pipeline YAMLs

## Questions?

- Open an issue for bugs or features
- Check docs/ and examples/ first
- Be specific and provide pipeline context

Thank you for contributing to Loom!
