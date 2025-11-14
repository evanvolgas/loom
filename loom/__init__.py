"""Loom - Orchestration framework for AI pipelines with built-in evaluation.

Loom is the "dbt for AI(E)TL" - a declarative orchestration framework that extends
traditional ETL with evaluation gates: Extract → Transform → Evaluate → Load.
"""

__version__ = "0.1.0-alpha"

# Public API will be exported here as modules are implemented
# from .pipeline import Pipeline
# from .core.config import LoomConfig
# from .engines import ExtractEngine, TransformEngine, EvaluateEngine, LoadEngine

__all__ = [
    "__version__",
]
