"""YAML pipeline definition parser."""

from pathlib import Path
from typing import Any, Dict, Union

import yaml
from pydantic import ValidationError as PydanticValidationError

from loom.core.exceptions import ConfigurationError, ValidationError
from loom.core.models import PipelineConfig


def parse_pipeline(pipeline_path: Union[str, Path]) -> PipelineConfig:
    """Parse pipeline definition from YAML file.

    Args:
        pipeline_path: Path to pipeline YAML file

    Returns:
        Validated PipelineConfig

    Raises:
        ConfigurationError: If file not found or invalid YAML
        ValidationError: If pipeline definition is invalid

    Example:
        >>> config = parse_pipeline("pipelines/customer_sentiment.yaml")
        >>> print(config.name)
        customer_sentiment
    """
    path = Path(pipeline_path)

    # Check file exists
    if not path.exists():
        raise ConfigurationError(f"Pipeline file not found: {path}")

    # Read and parse YAML
    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in {path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to read {path}: {e}")

    # Validate against Pydantic model
    try:
        pipeline_config = PipelineConfig.model_validate(data)
    except PydanticValidationError as e:
        raise ValidationError(f"Invalid pipeline definition in {path}:\n{e}")

    return pipeline_config


def validate_pipeline(pipeline_path: Union[str, Path]) -> bool:
    """Validate pipeline definition without raising exceptions.

    Args:
        pipeline_path: Path to pipeline YAML file

    Returns:
        True if valid, False otherwise

    Example:
        >>> if validate_pipeline("pipelines/example.yaml"):
        ...     print("Valid pipeline")
    """
    try:
        parse_pipeline(pipeline_path)
        return True
    except (ConfigurationError, ValidationError):
        return False


def parse_pipeline_from_dict(data: Dict[str, Any]) -> PipelineConfig:
    """Parse pipeline configuration from dictionary.

    Args:
        data: Pipeline configuration dictionary

    Returns:
        Validated PipelineConfig

    Raises:
        ValidationError: If pipeline definition is invalid

    Example:
        >>> data = {"name": "test", "version": "1.0", ...}
        >>> config = parse_pipeline_from_dict(data)
    """
    try:
        return PipelineConfig.model_validate(data)
    except PydanticValidationError as e:
        raise ValidationError(f"Invalid pipeline definition:\n{e}")
