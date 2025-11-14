"""Custom exceptions for Loom."""


class LoomError(Exception):
    """Base exception for all Loom errors."""

    pass


class ConfigurationError(LoomError):
    """Raised when pipeline configuration is invalid."""

    pass


class ExtractError(LoomError):
    """Raised when data extraction fails."""

    pass


class TransformError(LoomError):
    """Raised when data transformation fails."""

    pass


class EvaluateError(LoomError):
    """Raised when evaluation fails."""

    pass


class LoadError(LoomError):
    """Raised when data loading fails."""

    pass


class QualityGateError(LoomError):
    """Raised when quality gate check fails."""

    pass


class PipelineError(LoomError):
    """Raised when pipeline execution fails."""

    pass


class ValidationError(LoomError):
    """Raised when data validation fails."""

    pass


class ConnectorError(LoomError):
    """Raised when connector operation fails."""

    pass


class TimeoutError(LoomError):
    """Raised when operation exceeds timeout."""

    pass
