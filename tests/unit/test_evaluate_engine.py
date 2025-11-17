"""Unit tests for EvaluateEngine and quality gates."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from dataclasses import dataclass

from loom.core.models import EvaluateConfig, EvaluatorConfig, Record
from loom.core.types import QualityGateType, RecordStatus
from loom.core.exceptions import EvaluateError
from loom.engines.evaluate import EvaluateEngine


@dataclass
class MockScore:
    """Mock Arbiter score object."""
    name: str
    value: float


@dataclass
class MockEvaluationResult:
    """Mock Arbiter evaluation result."""
    scores: list


class TestEvaluateEngineQualityGates:
    """Test quality gate logic in EvaluateEngine."""

    def test_init(self):
        """Test engine initialization."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="semantic", type="semantic", threshold=0.8)
            ]
        )
        engine = EvaluateEngine(config)

        assert engine.config == config

    @pytest.mark.asyncio
    async def test_all_pass_gate_both_pass(self, mocker):
        """Test all_pass gate when both evaluators pass."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="semantic", type="semantic", threshold=0.8),
                EvaluatorConfig(name="criteria", type="custom_criteria", threshold=0.75),
            ],
            quality_gate=QualityGateType.ALL_PASS,
        )
        engine = EvaluateEngine(config)

        # Mock Arbiter
        mock_result = MockEvaluationResult(
            scores=[
                MockScore(name="semantic", value=0.85),
                MockScore(name="criteria", value=0.80),
            ]
        )
        mock_evaluate = mocker.patch(
            "loom.engines.evaluate.arbiter_evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        )

        record = Record(
            id="1",
            data={"text": "Test"},
            transformed_data="Transformed",
            status=RecordStatus.TRANSFORMED,
        )

        result = await engine.evaluate_record(record)

        assert result.quality_gate_passed is True
        assert result.status == RecordStatus.PASSED
        assert result.evaluation_scores["semantic"] == 0.85
        assert result.evaluation_scores["criteria"] == 0.80

    @pytest.mark.asyncio
    async def test_all_pass_gate_one_fails(self, mocker):
        """Test all_pass gate when one evaluator fails."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="semantic", type="semantic", threshold=0.8),
                EvaluatorConfig(name="criteria", type="custom_criteria", threshold=0.75),
            ],
            quality_gate=QualityGateType.ALL_PASS,
        )
        engine = EvaluateEngine(config)

        # Mock Arbiter - one score below threshold
        mock_result = MockEvaluationResult(
            scores=[
                MockScore(name="semantic", value=0.85),
                MockScore(name="criteria", value=0.70),  # Below 0.75 threshold
            ]
        )
        mocker.patch(
            "loom.engines.evaluate.arbiter_evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        )

        record = Record(
            id="1",
            data={"text": "Test"},
            transformed_data="Transformed",
            status=RecordStatus.TRANSFORMED,
        )

        result = await engine.evaluate_record(record)

        assert result.quality_gate_passed is False
        assert result.status == RecordStatus.FAILED

    @pytest.mark.asyncio
    async def test_majority_pass_gate_2_of_3(self, mocker):
        """Test majority_pass gate with 2 of 3 passing."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="eval1", type="semantic", threshold=0.8),
                EvaluatorConfig(name="eval2", type="semantic", threshold=0.8),
                EvaluatorConfig(name="eval3", type="semantic", threshold=0.8),
            ],
            quality_gate=QualityGateType.MAJORITY_PASS,
        )
        engine = EvaluateEngine(config)

        # Mock Arbiter - 2 pass, 1 fail
        mock_result = MockEvaluationResult(
            scores=[
                MockScore(name="eval1", value=0.85),
                MockScore(name="eval2", value=0.82),
                MockScore(name="eval3", value=0.75),  # Below threshold
            ]
        )
        mocker.patch(
            "loom.engines.evaluate.arbiter_evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        )

        record = Record(
            id="1",
            data={"text": "Test"},
            transformed_data="Transformed",
        )

        result = await engine.evaluate_record(record)

        # 2 out of 3 = 66% > 50%, should pass
        assert result.quality_gate_passed is True
        assert result.status == RecordStatus.PASSED

    @pytest.mark.asyncio
    async def test_majority_pass_gate_1_of_2_fails(self, mocker):
        """Test majority_pass gate with 1 of 2 (50%) - should fail."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="eval1", type="semantic", threshold=0.8),
                EvaluatorConfig(name="eval2", type="semantic", threshold=0.8),
            ],
            quality_gate=QualityGateType.MAJORITY_PASS,
        )
        engine = EvaluateEngine(config)

        # Mock Arbiter - 1 pass, 1 fail
        mock_result = MockEvaluationResult(
            scores=[
                MockScore(name="eval1", value=0.85),
                MockScore(name="eval2", value=0.75),  # Below threshold
            ]
        )
        mocker.patch(
            "loom.engines.evaluate.arbiter_evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        )

        record = Record(id="1", data={"text": "Test"}, transformed_data="Transformed")

        result = await engine.evaluate_record(record)

        # 1 out of 2 = 50%, NOT > 50%, should fail
        assert result.quality_gate_passed is False
        assert result.status == RecordStatus.FAILED

    @pytest.mark.asyncio
    async def test_majority_pass_gate_single_evaluator_pass(self, mocker):
        """Test majority_pass with single evaluator passing."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="eval1", type="semantic", threshold=0.8),
            ],
            quality_gate=QualityGateType.MAJORITY_PASS,
        )
        engine = EvaluateEngine(config)

        mock_result = MockEvaluationResult(scores=[MockScore(name="eval1", value=0.85)])
        mocker.patch(
            "loom.engines.evaluate.arbiter_evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        )

        record = Record(id="1", data={"text": "Test"}, transformed_data="Transformed")
        result = await engine.evaluate_record(record)

        # 1 out of 1 = 100% > 50%, should pass
        assert result.quality_gate_passed is True

    @pytest.mark.asyncio
    async def test_any_pass_gate_one_passes(self, mocker):
        """Test any_pass gate when at least one evaluator passes."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="eval1", type="semantic", threshold=0.8),
                EvaluatorConfig(name="eval2", type="semantic", threshold=0.8),
                EvaluatorConfig(name="eval3", type="semantic", threshold=0.8),
            ],
            quality_gate=QualityGateType.ANY_PASS,
        )
        engine = EvaluateEngine(config)

        # Mock Arbiter - only one passes
        mock_result = MockEvaluationResult(
            scores=[
                MockScore(name="eval1", value=0.85),  # Only this passes
                MockScore(name="eval2", value=0.70),
                MockScore(name="eval3", value=0.60),
            ]
        )
        mocker.patch(
            "loom.engines.evaluate.arbiter_evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        )

        record = Record(id="1", data={"text": "Test"}, transformed_data="Transformed")
        result = await engine.evaluate_record(record)

        assert result.quality_gate_passed is True
        assert result.status == RecordStatus.PASSED

    @pytest.mark.asyncio
    async def test_any_pass_gate_all_fail(self, mocker):
        """Test any_pass gate when all evaluators fail."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="eval1", type="semantic", threshold=0.8),
                EvaluatorConfig(name="eval2", type="semantic", threshold=0.8),
            ],
            quality_gate=QualityGateType.ANY_PASS,
        )
        engine = EvaluateEngine(config)

        # Mock Arbiter - all fail
        mock_result = MockEvaluationResult(
            scores=[
                MockScore(name="eval1", value=0.70),
                MockScore(name="eval2", value=0.75),
            ]
        )
        mocker.patch(
            "loom.engines.evaluate.arbiter_evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        )

        record = Record(id="1", data={"text": "Test"}, transformed_data="Transformed")
        result = await engine.evaluate_record(record)

        assert result.quality_gate_passed is False
        assert result.status == RecordStatus.FAILED

    @pytest.mark.asyncio
    async def test_weighted_gate_passes(self, mocker):
        """Test weighted gate when weighted average exceeds threshold."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="eval1", type="semantic", threshold=0.7, weight=2.0),
                EvaluatorConfig(name="eval2", type="semantic", threshold=0.7, weight=1.0),
                EvaluatorConfig(name="eval3", type="semantic", threshold=0.7, weight=0.5),
            ],
            quality_gate=QualityGateType.WEIGHTED,
            quality_gate_threshold=0.75,
        )
        engine = EvaluateEngine(config)

        # Mock scores: (0.90*2.0 + 0.70*1.0 + 0.60*0.5) / (2.0+1.0+0.5) = 2.8/3.5 = 0.8
        mock_result = MockEvaluationResult(
            scores=[
                MockScore(name="eval1", value=0.90),
                MockScore(name="eval2", value=0.70),
                MockScore(name="eval3", value=0.60),
            ]
        )
        mocker.patch(
            "loom.engines.evaluate.arbiter_evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        )

        record = Record(id="1", data={"text": "Test"}, transformed_data="Transformed")
        result = await engine.evaluate_record(record)

        # Weighted avg 0.8 >= 0.75, should pass
        assert result.quality_gate_passed is True
        assert result.status == RecordStatus.PASSED

    @pytest.mark.asyncio
    async def test_weighted_gate_fails(self, mocker):
        """Test weighted gate when weighted average below threshold."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="eval1", type="semantic", threshold=0.7, weight=2.0),
                EvaluatorConfig(name="eval2", type="semantic", threshold=0.7, weight=1.0),
            ],
            quality_gate=QualityGateType.WEIGHTED,
            quality_gate_threshold=0.75,
        )
        engine = EvaluateEngine(config)

        # Mock scores: (0.70*2.0 + 0.75*1.0) / 3.0 = 2.15/3.0 = 0.716
        mock_result = MockEvaluationResult(
            scores=[
                MockScore(name="eval1", value=0.70),
                MockScore(name="eval2", value=0.75),
            ]
        )
        mocker.patch(
            "loom.engines.evaluate.arbiter_evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        )

        record = Record(id="1", data={"text": "Test"}, transformed_data="Transformed")
        result = await engine.evaluate_record(record)

        # Weighted avg 0.716 < 0.75, should fail
        assert result.quality_gate_passed is False

    @pytest.mark.asyncio
    async def test_weighted_gate_missing_threshold_raises_error(self, mocker):
        """Test weighted gate without quality_gate_threshold raises error."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="eval1", type="semantic", threshold=0.8, weight=1.0),
            ],
            quality_gate=QualityGateType.WEIGHTED,
            quality_gate_threshold=None,  # Missing!
        )
        engine = EvaluateEngine(config)

        mock_result = MockEvaluationResult(scores=[MockScore(name="eval1", value=0.85)])
        mocker.patch(
            "loom.engines.evaluate.arbiter_evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        )

        record = Record(id="1", data={"text": "Test"}, transformed_data="Transformed")

        with pytest.raises(EvaluateError) as exc_info:
            await engine.evaluate_record(record)

        assert "quality_gate_threshold required" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_evaluate_with_reference_field(self, mocker):
        """Test evaluation with reference field for semantic comparison."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="semantic", type="semantic", threshold=0.8),
            ],
            reference_field="expected_output",
        )
        engine = EvaluateEngine(config)

        mock_result = MockEvaluationResult(scores=[MockScore(name="semantic", value=0.90)])
        mock_evaluate = mocker.patch(
            "loom.engines.evaluate.arbiter_evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        )

        record = Record(
            id="1",
            data={"text": "Input", "expected_output": "Expected result"},
            transformed_data="Actual result",
        )

        result = await engine.evaluate_record(record)

        # Verify Arbiter was called with reference
        mock_evaluate.assert_called_once()
        call_kwargs = mock_evaluate.call_args.kwargs
        assert call_kwargs["output"] == "Actual result"
        assert call_kwargs["reference"] == "Expected result"
        assert result.quality_gate_passed is True

    @pytest.mark.asyncio
    async def test_evaluate_without_reference_field(self, mocker):
        """Test evaluation without reference field."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="semantic", type="semantic", threshold=0.8),
            ],
            reference_field=None,
        )
        engine = EvaluateEngine(config)

        mock_result = MockEvaluationResult(scores=[MockScore(name="semantic", value=0.90)])
        mock_evaluate = mocker.patch(
            "loom.engines.evaluate.arbiter_evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        )

        record = Record(
            id="1",
            data={"text": "Input"},
            transformed_data="Output",
        )

        await engine.evaluate_record(record)

        # Verify reference=None
        call_kwargs = mock_evaluate.call_args.kwargs
        assert call_kwargs["reference"] is None

    @pytest.mark.asyncio
    async def test_evaluate_no_transformed_data_raises_error(self):
        """Test evaluation fails when record has no transformed data."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="semantic", type="semantic", threshold=0.8),
            ]
        )
        engine = EvaluateEngine(config)

        record = Record(
            id="1",
            data={"text": "Test"},
            transformed_data=None,  # Missing!
        )

        with pytest.raises(EvaluateError) as exc_info:
            await engine.evaluate_record(record)

        assert "no transformed data" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_evaluate_timeout(self, mocker):
        """Test evaluation timeout handling."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="semantic", type="semantic", threshold=0.8),
            ],
            timeout=0.001,  # Very short timeout
        )
        engine = EvaluateEngine(config)

        # Mock Arbiter to take longer than timeout
        async def slow_evaluate(*args, **kwargs):
            import asyncio
            await asyncio.sleep(1)  # Takes 1 second
            return MockEvaluationResult(scores=[MockScore(name="semantic", value=0.9)])

        mocker.patch(
            "loom.engines.evaluate.arbiter_evaluate",
            side_effect=slow_evaluate,
        )

        record = Record(id="1", data={"text": "Test"}, transformed_data="Output")

        with pytest.raises(EvaluateError) as exc_info:
            await engine.evaluate_record(record)

        assert "timeout" in str(exc_info.value).lower()
        assert record.status == RecordStatus.ERROR
        assert record.error is not None

    @pytest.mark.asyncio
    async def test_evaluate_batch(self, mocker):
        """Test batch evaluation of multiple records."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="semantic", type="semantic", threshold=0.8),
            ]
        )
        engine = EvaluateEngine(config)

        mock_result = MockEvaluationResult(scores=[MockScore(name="semantic", value=0.90)])
        mocker.patch(
            "loom.engines.evaluate.arbiter_evaluate",
            new_callable=AsyncMock,
            return_value=mock_result,
        )

        records = [
            Record(id="1", data={"text": "Test 1"}, transformed_data="Output 1"),
            Record(id="2", data={"text": "Test 2"}, transformed_data="Output 2"),
            Record(id="3", data={"text": "Test 3"}, transformed_data="Output 3"),
        ]

        results = await engine.evaluate_batch(records)

        assert len(results) == 3
        assert all(r.quality_gate_passed is True for r in results)
        assert all(r.status == RecordStatus.PASSED for r in results)

    @pytest.mark.asyncio
    async def test_evaluate_batch_mixed_results(self, mocker):
        """Test batch evaluation with mixed pass/fail results."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="semantic", type="semantic", threshold=0.8),
            ]
        )
        engine = EvaluateEngine(config)

        # Different scores for different records
        call_count = 0

        async def mock_evaluate_varying(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            score = 0.90 if call_count % 2 == 0 else 0.70  # Alternate pass/fail
            return MockEvaluationResult(scores=[MockScore(name="semantic", value=score)])

        mocker.patch(
            "loom.engines.evaluate.arbiter_evaluate",
            side_effect=mock_evaluate_varying,
        )

        records = [
            Record(id="1", data={"text": "Test 1"}, transformed_data="Output 1"),
            Record(id="2", data={"text": "Test 2"}, transformed_data="Output 2"),
            Record(id="3", data={"text": "Test 3"}, transformed_data="Output 3"),
        ]

        results = await engine.evaluate_batch(records)

        passed = [r for r in results if r.quality_gate_passed]
        failed = [r for r in results if not r.quality_gate_passed and r.status != RecordStatus.ERROR]

        assert len(passed) > 0
        assert len(failed) > 0

    def test_check_batch_threshold_passes(self):
        """Test batch threshold check when enough records pass."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="semantic", type="semantic", threshold=0.8),
            ],
            batch_threshold=0.8,  # 80% must pass
        )
        engine = EvaluateEngine(config)

        records = [
            Record(id="1", data={}, quality_gate_passed=True),
            Record(id="2", data={}, quality_gate_passed=True),
            Record(id="3", data={}, quality_gate_passed=True),
            Record(id="4", data={}, quality_gate_passed=True),
            Record(id="5", data={}, quality_gate_passed=False),
        ]

        # 4/5 = 80%, should pass
        result = engine.check_batch_threshold(records)
        assert result is True

    def test_check_batch_threshold_fails(self):
        """Test batch threshold check when not enough records pass."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="semantic", type="semantic", threshold=0.8),
            ],
            batch_threshold=0.8,  # 80% must pass
        )
        engine = EvaluateEngine(config)

        records = [
            Record(id="1", data={}, quality_gate_passed=True),
            Record(id="2", data={}, quality_gate_passed=True),
            Record(id="3", data={}, quality_gate_passed=False),
            Record(id="4", data={}, quality_gate_passed=False),
            Record(id="5", data={}, quality_gate_passed=False),
        ]

        # 2/5 = 40%, should fail
        result = engine.check_batch_threshold(records)
        assert result is False

    def test_check_batch_threshold_none(self):
        """Test batch threshold check when no threshold configured."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="semantic", type="semantic", threshold=0.8),
            ],
            batch_threshold=None,  # No batch threshold
        )
        engine = EvaluateEngine(config)

        records = [
            Record(id="1", data={}, quality_gate_passed=False),
            Record(id="2", data={}, quality_gate_passed=False),
        ]

        # No threshold, should always pass
        result = engine.check_batch_threshold(records)
        assert result is True

    def test_check_batch_threshold_empty_records(self):
        """Test batch threshold with empty records list."""
        config = EvaluateConfig(
            evaluators=[
                EvaluatorConfig(name="semantic", type="semantic", threshold=0.8),
            ],
            batch_threshold=0.8,
        )
        engine = EvaluateEngine(config)

        # Empty list should pass
        result = engine.check_batch_threshold([])
        assert result is True
