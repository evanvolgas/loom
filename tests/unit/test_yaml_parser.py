"""Unit tests for YAML pipeline parser."""

import pytest
from pathlib import Path
from pydantic import ValidationError as PydanticValidationError

from loom.parsers.yaml_parser import parse_pipeline
from loom.core.models import PipelineConfig
from loom.core.exceptions import ConfigurationError, ValidationError
from loom.core.types import QualityGateType, SourceType, DestinationType


class TestYAMLParser:
    """Test YAML pipeline parser."""

    def test_parse_simple_pipeline(self):
        """Test parsing simple valid pipeline."""
        config = parse_pipeline("tests/fixtures/pipelines/simple_pipeline.yaml")

        assert isinstance(config, PipelineConfig)
        assert config.name == "simple_sentiment_analysis"
        assert config.version == "1.0.0"
        assert config.description == "Simple sentiment analysis pipeline for testing"

    def test_parse_pipeline_extract_config(self):
        """Test extract configuration is parsed correctly."""
        config = parse_pipeline("tests/fixtures/pipelines/simple_pipeline.yaml")

        assert config.extract.source == "tests/fixtures/data/sample.csv"
        assert config.extract.batch_size == 10

    def test_parse_pipeline_transform_config(self):
        """Test transform configuration is parsed correctly."""
        config = parse_pipeline("tests/fixtures/pipelines/simple_pipeline.yaml")

        assert config.transform.prompt == "tests/fixtures/prompts/classify_sentiment.txt"
        assert config.transform.model == "gpt-4o-mini"
        assert config.transform.provider == "openai"
        assert config.transform.batch_size == 5
        assert config.transform.temperature == 0.7
        assert config.transform.timeout == 60.0

    def test_parse_pipeline_evaluate_config(self):
        """Test evaluate configuration is parsed correctly."""
        config = parse_pipeline("tests/fixtures/pipelines/simple_pipeline.yaml")

        assert len(config.evaluate.evaluators) == 1
        assert config.evaluate.evaluators[0].name == "semantic"
        assert config.evaluate.evaluators[0].type == "semantic"
        assert config.evaluate.evaluators[0].threshold == 0.8
        assert config.evaluate.quality_gate == QualityGateType.ALL_PASS
        assert config.evaluate.reference_field == "expected_sentiment"
        assert config.evaluate.timeout == 30.0

    def test_parse_pipeline_load_config(self):
        """Test load configuration is parsed correctly."""
        config = parse_pipeline("tests/fixtures/pipelines/simple_pipeline.yaml")

        assert config.load.destination == "tests/fixtures/output/results.csv"
        assert config.load.mode == "overwrite"
        assert config.load.batch_size == 10

    def test_parse_weighted_gate_pipeline(self):
        """Test parsing pipeline with weighted quality gate."""
        config = parse_pipeline("tests/fixtures/pipelines/weighted_gate_pipeline.yaml")

        assert config.name == "weighted_quality_gate_test"
        assert len(config.evaluate.evaluators) == 2
        assert config.evaluate.evaluators[0].weight == 2.0
        assert config.evaluate.evaluators[1].weight == 1.0
        assert config.evaluate.quality_gate == QualityGateType.WEIGHTED
        assert config.evaluate.quality_gate_threshold == 0.75

    def test_parse_majority_gate_pipeline(self):
        """Test parsing pipeline with majority_pass quality gate."""
        config = parse_pipeline("tests/fixtures/pipelines/majority_gate_pipeline.yaml")

        assert config.name == "majority_quality_gate_test"
        assert len(config.evaluate.evaluators) == 3
        assert config.evaluate.quality_gate == QualityGateType.MAJORITY_PASS

    def test_parse_pipeline_file_not_found(self):
        """Test parsing non-existent pipeline file."""
        with pytest.raises(ConfigurationError) as exc_info:
            parse_pipeline("tests/fixtures/pipelines/nonexistent.yaml")

        assert "Pipeline file not found" in str(exc_info.value)

    def test_parse_pipeline_invalid_yaml(self, tmp_path):
        """Test parsing file with invalid YAML syntax."""
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("name: test\ninvalid: [unclosed")

        with pytest.raises(ConfigurationError) as exc_info:
            parse_pipeline(str(invalid_yaml))

        assert "Invalid YAML" in str(exc_info.value)

    def test_parse_pipeline_missing_required_field(self, tmp_path):
        """Test parsing pipeline missing required field."""
        incomplete_yaml = tmp_path / "incomplete.yaml"
        incomplete_yaml.write_text("""
name: test_pipeline
version: 1.0.0
# Missing extract, transform, evaluate, load
""")

        with pytest.raises(ValidationError) as exc_info:
            parse_pipeline(str(incomplete_yaml))

        assert "Invalid pipeline definition" in str(exc_info.value)

    def test_parse_pipeline_invalid_threshold(self, tmp_path):
        """Test parsing pipeline with invalid threshold value."""
        invalid_threshold = tmp_path / "invalid_threshold.yaml"
        invalid_threshold.write_text("""
name: test
version: 1.0.0
extract:
  source: data.csv
transform:
  prompt: prompt.txt
evaluate:
  evaluators:
    - name: test
      type: semantic
      threshold: 1.5  # Invalid: must be <= 1.0
  quality_gate: all_pass
load:
  destination: output.csv
""")

        with pytest.raises(ValidationError):
            parse_pipeline(str(invalid_threshold))

    def test_parse_pipeline_invalid_batch_size(self, tmp_path):
        """Test parsing pipeline with invalid batch size."""
        invalid_batch = tmp_path / "invalid_batch.yaml"
        invalid_batch.write_text("""
name: test
version: 1.0.0
extract:
  source: data.csv
  batch_size: 0  # Invalid: must be > 0
transform:
  prompt: prompt.txt
evaluate:
  evaluators:
    - name: test
      type: semantic
      threshold: 0.8
  quality_gate: all_pass
load:
  destination: output.csv
""")

        with pytest.raises(ValidationError):
            parse_pipeline(str(invalid_batch))

    def test_parse_pipeline_empty_evaluators(self, tmp_path):
        """Test parsing pipeline with empty evaluators list."""
        empty_evaluators = tmp_path / "empty_eval.yaml"
        empty_evaluators.write_text("""
name: test
version: 1.0.0
extract:
  source: data.csv
transform:
  prompt: prompt.txt
evaluate:
  evaluators: []  # Invalid: must have at least one
  quality_gate: all_pass
load:
  destination: output.csv
""")

        with pytest.raises(ValidationError):
            parse_pipeline(str(empty_evaluators))

    def test_parse_pipeline_with_path_object(self):
        """Test parsing pipeline using Path object instead of string."""
        path = Path("tests/fixtures/pipelines/simple_pipeline.yaml")
        config = parse_pipeline(path)

        assert config.name == "simple_sentiment_analysis"

    def test_parse_pipeline_source_type_auto_detect(self):
        """Test that source_type can be omitted for auto-detection."""
        config = parse_pipeline("tests/fixtures/pipelines/simple_pipeline.yaml")

        # source_type not specified in YAML, should be None for auto-detect
        assert config.extract.source_type is None

    def test_parse_pipeline_source_type_explicit(self):
        """Test that source_type can be explicitly specified."""
        config = parse_pipeline("tests/fixtures/pipelines/majority_gate_pipeline.yaml")

        assert config.extract.source_type == SourceType.PARQUET

    def test_parse_pipeline_destination_type_explicit(self):
        """Test that destination_type can be explicitly specified."""
        config = parse_pipeline("tests/fixtures/pipelines/weighted_gate_pipeline.yaml")

        assert config.load.destination_type == DestinationType.JSON

    def test_parse_pipeline_evaluator_with_criteria(self):
        """Test parsing evaluator with custom criteria."""
        config = parse_pipeline("tests/fixtures/pipelines/weighted_gate_pipeline.yaml")

        criteria_evaluator = config.evaluate.evaluators[1]
        assert criteria_evaluator.name == "criteria"
        assert criteria_evaluator.type == "custom_criteria"
        assert criteria_evaluator.criteria == "Is the sentiment classification accurate?"

    def test_parse_pipeline_default_values(self, tmp_path):
        """Test that pipeline uses default values when not specified."""
        minimal_yaml = tmp_path / "minimal.yaml"
        minimal_yaml.write_text("""
name: minimal
version: 1.0.0
extract:
  source: data.csv
transform:
  prompt: prompt.txt
evaluate:
  evaluators:
    - name: test
      type: semantic
      threshold: 0.8
load:
  destination: output.csv
""")

        config = parse_pipeline(str(minimal_yaml))

        # Check defaults
        assert config.extract.batch_size == 100  # default
        assert config.transform.model == "gpt-4o-mini"  # default
        assert config.transform.provider == "openai"  # default
        assert config.transform.temperature == 0.7  # default
        assert config.evaluate.quality_gate == QualityGateType.ALL_PASS  # default
        assert config.load.mode == "append"  # default

    def test_parse_pipeline_description_optional(self, tmp_path):
        """Test that description field is optional."""
        no_desc = tmp_path / "no_desc.yaml"
        no_desc.write_text("""
name: test
version: 1.0.0
extract:
  source: data.csv
transform:
  prompt: prompt.txt
evaluate:
  evaluators:
    - name: test
      type: semantic
      threshold: 0.8
load:
  destination: output.csv
""")

        config = parse_pipeline(str(no_desc))
        assert config.description is None

    def test_parse_pipeline_all_quality_gate_types(self, tmp_path):
        """Test parsing pipelines with different quality gate types."""
        gate_types = {
            "all_pass": QualityGateType.ALL_PASS,
            "majority_pass": QualityGateType.MAJORITY_PASS,
            "any_pass": QualityGateType.ANY_PASS,
            "weighted": QualityGateType.WEIGHTED,
        }

        for gate_str, gate_enum in gate_types.items():
            yaml_path = tmp_path / f"{gate_str}.yaml"
            yaml_content = f"""
name: test_{gate_str}
version: 1.0.0
extract:
  source: data.csv
transform:
  prompt: prompt.txt
evaluate:
  evaluators:
    - name: test
      type: semantic
      threshold: 0.8
  quality_gate: {gate_str}
  {'quality_gate_threshold: 0.75' if gate_str == 'weighted' else ''}
load:
  destination: output.csv
"""
            yaml_path.write_text(yaml_content)

            config = parse_pipeline(str(yaml_path))
            assert config.evaluate.quality_gate == gate_enum

    def test_parse_pipeline_max_tokens_optional(self, tmp_path):
        """Test that max_tokens is optional in transform config."""
        no_max_tokens = tmp_path / "no_max_tokens.yaml"
        no_max_tokens.write_text("""
name: test
version: 1.0.0
extract:
  source: data.csv
transform:
  prompt: prompt.txt
  # max_tokens not specified
evaluate:
  evaluators:
    - name: test
      type: semantic
      threshold: 0.8
load:
  destination: output.csv
""")

        config = parse_pipeline(str(no_max_tokens))
        assert config.transform.max_tokens is None
