# Quality Gate Specification

**Version:** 1.0.0
**Status:** Specification
**Created:** 2025-11-14

---

## Purpose

This document provides precise semantics for Loom's quality gate system, which determines whether AI-transformed records pass evaluation criteria and proceed to the load stage.

---

## Overview

Quality gates are **decision rules** that determine whether a record or batch passes evaluation based on evaluator scores and configured thresholds.

### Key Concepts

**Evaluator**: A single evaluation function (e.g., semantic_similarity, custom_criteria)
**Score**: Numeric value [0.0, 1.0] representing quality for one evaluator
**Threshold**: Minimum acceptable score for an evaluator to "pass"
**Record**: Single data item flowing through pipeline
**Batch**: Collection of records processed together

---

## Quality Gate Types

### 1. `all_pass` - All Evaluators Must Pass

**Semantics**: Every configured evaluator must achieve its threshold for the record to pass.

**Application**: Per-record basis (each record evaluated independently)

**Pass Condition**:
```
∀ evaluator ∈ evaluators: score(evaluator) ≥ threshold(evaluator)
```

**Example Configuration**:
```yaml
evaluate:
  evaluators:
    - name: semantic_similarity
      type: semantic
      threshold: 0.8
    - name: quality_criteria
      type: custom_criteria
      threshold: 0.75
  quality_gate: all_pass
```

**Concrete Examples**:

```gherkin
Scenario 1: Both evaluators pass threshold
  Given: Evaluators [semantic: 0.8, criteria: 0.75]
  When: Record evaluated with scores:
    | evaluator | score | threshold |
    | semantic  | 0.85  | 0.8       |
    | criteria  | 0.80  | 0.75      |
  Then: Quality gate PASSES
    And: Record proceeds to load stage
    And: record_evaluations table: passed = true

Scenario 2: One evaluator fails threshold
  Given: Evaluators [semantic: 0.8, criteria: 0.75]
  When: Record evaluated with scores:
    | evaluator | score | threshold |
    | semantic  | 0.85  | 0.8       |
    | criteria  | 0.70  | 0.75      |
  Then: Quality gate FAILS
    And: Record quarantined
    And: failure_reason = "criteria evaluator below threshold (0.70 < 0.75)"
    And: quarantined_records table updated
    And: Pipeline continues with next record

Scenario 3: Both evaluators fail
  Given: Evaluators [semantic: 0.8, criteria: 0.75]
  When: Record evaluated with scores:
    | evaluator | score | threshold |
    | semantic  | 0.60  | 0.8       |
    | criteria  | 0.65  | 0.75      |
  Then: Quality gate FAILS
    And: Record quarantined
    And: failure_reason = "Multiple evaluators failed: semantic (0.60 < 0.8), criteria (0.65 < 0.75)"
```

**SQL Implementation Check**:
```sql
-- Verify quality gate logic in record_evaluations table
SELECT
  run_id,
  record_id,
  COUNT(*) as total_evaluators,
  SUM(CASE WHEN passed THEN 1 ELSE 0 END) as passed_evaluators,
  CASE
    WHEN COUNT(*) = SUM(CASE WHEN passed THEN 1 ELSE 0 END) THEN 'PASS'
    ELSE 'FAIL'
  END as quality_gate_result
FROM record_evaluations
WHERE run_id = 'run_abc123'
GROUP BY run_id, record_id;
```

---

### 2. `majority_pass` - More Than Half Pass

**Semantics**: Strictly more than 50% of evaluators must pass their thresholds.

**Pass Condition**:
```
count(passed_evaluators) > count(total_evaluators) / 2
```

**Example Configuration**:
```yaml
evaluate:
  evaluators:
    - name: semantic
      threshold: 0.8
    - name: criteria
      threshold: 0.75
    - name: tone
      threshold: 0.7
  quality_gate: majority_pass
```

**Concrete Examples**:

```gherkin
Scenario 1: 2 of 3 evaluators pass (majority)
  Given: 3 evaluators with thresholds
  When: Record evaluated with scores:
    | evaluator | score | threshold | passed |
    | semantic  | 0.85  | 0.8       | ✓      |
    | criteria  | 0.80  | 0.75      | ✓      |
    | tone      | 0.65  | 0.7       | ✗      |
  Then: Quality gate PASSES (2/3 > 50%)
    And: Record proceeds to load stage

Scenario 2: Exactly 50% pass (not majority)
  Given: 2 evaluators with thresholds
  When: Record evaluated with scores:
    | evaluator | score | threshold | passed |
    | semantic  | 0.85  | 0.8       | ✓      |
    | criteria  | 0.70  | 0.75      | ✗      |
  Then: Quality gate FAILS (1/2 = 50%, not > 50%)
    And: Record quarantined
    And: failure_reason = "Majority not achieved: 1/2 passed (50%)"

Scenario 3: 3 of 4 evaluators pass (majority)
  Given: 4 evaluators with thresholds
  When: 3 evaluators pass, 1 fails
  Then: Quality gate PASSES (3/4 = 75% > 50%)
```

**Edge Case: Single Evaluator**:
```gherkin
Scenario: Single evaluator with majority_pass
  Given: 1 evaluator
  When: Evaluator passes threshold
  Then: Quality gate PASSES (1/1 = 100% > 50%)

  When: Evaluator fails threshold
  Then: Quality gate FAILS (0/1 = 0% not > 50%)
```

---

### 3. `any_pass` - At Least One Passes

**Semantics**: At least one evaluator must pass its threshold.

**Pass Condition**:
```
∃ evaluator ∈ evaluators: score(evaluator) ≥ threshold(evaluator)
```

**Example Configuration**:
```yaml
evaluate:
  evaluators:
    - name: semantic
      threshold: 0.8
    - name: criteria
      threshold: 0.75
  quality_gate: any_pass
```

**Concrete Examples**:

```gherkin
Scenario 1: First evaluator passes
  Given: Evaluators [semantic: 0.8, criteria: 0.75]
  When: Record evaluated with scores:
    | evaluator | score | threshold | passed |
    | semantic  | 0.85  | 0.8       | ✓      |
    | criteria  | 0.70  | 0.75      | ✗      |
  Then: Quality gate PASSES
    And: Record proceeds to load stage

Scenario 2: All evaluators fail
  Given: Evaluators [semantic: 0.8, criteria: 0.75]
  When: Record evaluated with scores:
    | evaluator | score | threshold | passed |
    | semantic  | 0.75  | 0.8       | ✗      |
    | criteria  | 0.70  | 0.75      | ✗      |
  Then: Quality gate FAILS
    And: Record quarantined
    And: failure_reason = "No evaluators passed threshold"
```

**Use Case**: Lenient quality gate for exploratory analysis or A/B testing where any positive signal is valuable.

---

### 4. `weighted` - Weighted Average Above Threshold

**Semantics**: Weighted average of all evaluator scores must exceed configured threshold.

**Pass Condition**:
```
weighted_avg = Σ(score(e) × weight(e)) / Σ(weight(e)) ≥ threshold
```

**Example Configuration**:
```yaml
evaluate:
  evaluators:
    - name: semantic
      type: semantic
      weight: 2.0  # High importance
    - name: criteria
      type: custom_criteria
      weight: 1.0  # Normal importance
    - name: tone
      type: tone_analysis
      weight: 0.5  # Low importance
  quality_gate:
    type: weighted
    threshold: 0.75  # Weighted average must be ≥ 0.75
```

**Concrete Examples**:

```gherkin
Scenario 1: Weighted average exceeds threshold
  Given: Weighted gate with threshold 0.75
  When: Record evaluated with scores:
    | evaluator | score | weight | weighted_score |
    | semantic  | 0.90  | 2.0    | 1.80           |
    | criteria  | 0.70  | 1.0    | 0.70           |
    | tone      | 0.60  | 0.5    | 0.30           |
  Then: weighted_avg = (1.80 + 0.70 + 0.30) / (2.0 + 1.0 + 0.5)
                     = 2.80 / 3.5
                     = 0.80
    And: Quality gate PASSES (0.80 ≥ 0.75)

Scenario 2: Weighted average below threshold
  Given: Weighted gate with threshold 0.75
  When: Record evaluated with scores:
    | evaluator | score | weight | weighted_score |
    | semantic  | 0.70  | 2.0    | 1.40           |
    | criteria  | 0.75  | 1.0    | 0.75           |
    | tone      | 0.80  | 0.5    | 0.40           |
  Then: weighted_avg = (1.40 + 0.75 + 0.40) / 3.5 = 0.729
    And: Quality gate FAILS (0.729 < 0.75)
    And: Record quarantined

Scenario 3: Default weights (all 1.0)
  Given: No weights specified
  When: Evaluators have implicit weight 1.0
  Then: weighted_avg = simple arithmetic mean
```

---

## Batch-Level Quality Gates

All quality gates operate **per-record** by default, but pipelines also track **batch-level metrics**.

### Batch Quality Metrics

```python
@dataclass
class BatchQualityResult:
    """Batch-level quality aggregation."""
    total_records: int
    passed_records: int
    failed_records: int
    pass_rate: float  # passed / total

    mean_score: float  # Average across all evaluators and records
    std_score: float   # Standard deviation
    min_score: float
    max_score: float
```

### Batch Failure Threshold

Optional configuration to fail entire pipeline if batch quality is too low:

```yaml
evaluate:
  quality_gate: all_pass
  batch_threshold: 0.95  # At least 95% of records must pass
```

**Concrete Example**:

```gherkin
Scenario: Batch threshold violation aborts pipeline
  Given: Pipeline with batch_threshold 0.95
  When: Batch of 1000 records processed
    And: 920 records pass quality gate (92%)
    And: 80 records quarantined (8%)
  Then: Batch pass_rate = 0.92
    And: Batch threshold check FAILS (0.92 < 0.95)
    And: Pipeline aborts with status "partial"
    And: Error message = "Batch quality below threshold: 92.0% < 95.0%"
    And: All 920 passed records still loaded to destination
    And: 80 quarantined records remain in quarantine table
```

**Without Batch Threshold**:
```gherkin
Scenario: No batch threshold (default behavior)
  Given: Pipeline without batch_threshold configured
  When: Any number of records fail quality gate
  Then: Pipeline continues processing all records
    And: Status = "success" if any records passed
    And: Failed records quarantined individually
    And: Passed records loaded to destination
```

---

## Implementation Reference

### Quality Gate Interface

```python
# loom/quality_gates/base.py
from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass

@dataclass
class EvaluatorResult:
    """Single evaluator result for a record."""
    evaluator_name: str
    score: float
    threshold: float
    passed: bool
    confidence: float
    explanation: str

class QualityGate(ABC):
    """Base interface for quality gates."""

    @abstractmethod
    def check_record(
        self,
        evaluations: List[EvaluatorResult]
    ) -> bool:
        """
        Check if record passes quality gate.

        Args:
            evaluations: Results from all evaluators for this record

        Returns:
            True if record passes gate, False if should be quarantined
        """
        pass

    @abstractmethod
    def get_failure_reason(
        self,
        evaluations: List[EvaluatorResult]
    ) -> str:
        """
        Generate human-readable failure reason.

        Returns:
            Explanation of why quality gate failed
        """
        pass
```

### Concrete Implementations

```python
# loom/quality_gates/all_pass.py
class AllPassQualityGate(QualityGate):
    """All evaluators must pass their thresholds."""

    def check_record(
        self,
        evaluations: List[EvaluatorResult]
    ) -> bool:
        return all(e.passed for e in evaluations)

    def get_failure_reason(
        self,
        evaluations: List[EvaluatorResult]
    ) -> str:
        failed = [e for e in evaluations if not e.passed]

        if len(failed) == 1:
            e = failed[0]
            return f"{e.evaluator_name} evaluator below threshold ({e.score:.2f} < {e.threshold})"
        else:
            reasons = [
                f"{e.evaluator_name} ({e.score:.2f} < {e.threshold})"
                for e in failed
            ]
            return f"Multiple evaluators failed: {', '.join(reasons)}"


# loom/quality_gates/majority_pass.py
class MajorityPassQualityGate(QualityGate):
    """More than 50% of evaluators must pass."""

    def check_record(
        self,
        evaluations: List[EvaluatorResult]
    ) -> bool:
        passed_count = sum(1 for e in evaluations if e.passed)
        return passed_count > len(evaluations) / 2

    def get_failure_reason(
        self,
        evaluations: List[EvaluatorResult]
    ) -> str:
        passed_count = sum(1 for e in evaluations if e.passed)
        total = len(evaluations)
        percentage = (passed_count / total * 100) if total > 0 else 0

        return f"Majority not achieved: {passed_count}/{total} passed ({percentage:.0f}%)"


# loom/quality_gates/weighted.py
class WeightedQualityGate(QualityGate):
    """Weighted average must exceed threshold."""

    def __init__(
        self,
        weights: Dict[str, float],
        threshold: float
    ):
        self.weights = weights
        self.threshold = threshold

    def check_record(
        self,
        evaluations: List[EvaluatorResult]
    ) -> bool:
        weighted_sum = sum(
            e.score * self.weights.get(e.evaluator_name, 1.0)
            for e in evaluations
        )
        total_weight = sum(
            self.weights.get(e.evaluator_name, 1.0)
            for e in evaluations
        )

        weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0.0
        return weighted_avg >= self.threshold

    def get_failure_reason(
        self,
        evaluations: List[EvaluatorResult]
    ) -> str:
        weighted_sum = sum(
            e.score * self.weights.get(e.evaluator_name, 1.0)
            for e in evaluations
        )
        total_weight = sum(
            self.weights.get(e.evaluator_name, 1.0)
            for e in evaluations
        )
        weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0.0

        return f"Weighted average below threshold ({weighted_avg:.3f} < {self.threshold})"
```

---

## Testing Strategy

### Unit Tests

```python
# tests/unit/test_quality_gates.py
import pytest
from loom.quality_gates import AllPassQualityGate, MajorityPassQualityGate
from loom.quality_gates.base import EvaluatorResult

class TestAllPassQualityGate:
    """Test all_pass gate semantics."""

    def test_all_evaluators_pass(self):
        """When all evaluators pass threshold, gate passes."""
        gate = AllPassQualityGate()
        evaluations = [
            EvaluatorResult("semantic", 0.85, 0.8, True, 0.9, "Good"),
            EvaluatorResult("criteria", 0.80, 0.75, True, 0.85, "Acceptable")
        ]

        assert gate.check_record(evaluations) is True

    def test_one_evaluator_fails(self):
        """When one evaluator fails, gate fails."""
        gate = AllPassQualityGate()
        evaluations = [
            EvaluatorResult("semantic", 0.85, 0.8, True, 0.9, "Good"),
            EvaluatorResult("criteria", 0.70, 0.75, False, 0.75, "Below threshold")
        ]

        assert gate.check_record(evaluations) is False
        assert "criteria" in gate.get_failure_reason(evaluations)

    def test_all_evaluators_fail(self):
        """When all evaluators fail, gate fails with all reasons."""
        gate = AllPassQualityGate()
        evaluations = [
            EvaluatorResult("semantic", 0.60, 0.8, False, 0.7, "Low"),
            EvaluatorResult("criteria", 0.65, 0.75, False, 0.7, "Low")
        ]

        assert gate.check_record(evaluations) is False
        reason = gate.get_failure_reason(evaluations)
        assert "semantic" in reason
        assert "criteria" in reason

class TestMajorityPassQualityGate:
    """Test majority_pass gate semantics."""

    def test_two_of_three_pass(self):
        """2/3 evaluators pass (majority)."""
        gate = MajorityPassQualityGate()
        evaluations = [
            EvaluatorResult("semantic", 0.85, 0.8, True, 0.9, "Good"),
            EvaluatorResult("criteria", 0.80, 0.75, True, 0.85, "Good"),
            EvaluatorResult("tone", 0.65, 0.7, False, 0.7, "Below")
        ]

        assert gate.check_record(evaluations) is True  # 2/3 > 50%

    def test_exactly_half_fails(self):
        """1/2 evaluators pass (not majority)."""
        gate = MajorityPassQualityGate()
        evaluations = [
            EvaluatorResult("semantic", 0.85, 0.8, True, 0.9, "Good"),
            EvaluatorResult("criteria", 0.70, 0.75, False, 0.75, "Below")
        ]

        assert gate.check_record(evaluations) is False  # 1/2 = 50%, not > 50%
```

### Integration Tests

```python
# tests/integration/test_quality_gate_pipeline.py
import pytest
from loom import Loom, Pipeline

@pytest.mark.asyncio
async def test_all_pass_gate_with_real_evaluators():
    """Test all_pass gate with real Arbiter evaluators."""
    loom = Loom()

    pipeline = Pipeline(
        name="test_quality_gate",
        evaluate=EvaluateConfig(
            evaluators=[
                {"type": "semantic", "threshold": 0.8},
                {"type": "custom_criteria", "criteria": "Accurate", "threshold": 0.75}
            ],
            quality_gate="all_pass"
        )
    )

    # Record that should pass
    good_record = Record(
        id="1",
        data={"output": "Paris is the capital of France", "reference": "Paris is France's capital"}
    )

    result = await loom.evaluate_record(pipeline, good_record)
    assert result.passed is True

    # Record that should fail
    bad_record = Record(
        id="2",
        data={"output": "London is the capital of France", "reference": "Paris is France's capital"}
    )

    result = await loom.evaluate_record(pipeline, bad_record)
    assert result.passed is False
```

---

## Monitoring and Observability

### Metrics

Track quality gate performance:

```python
# Prometheus metrics
quality_gate_pass_rate = Gauge(
    'loom_quality_gate_pass_rate',
    'Percentage of records passing quality gate',
    ['pipeline_name', 'gate_type']
)

quality_gate_failures_by_evaluator = Counter(
    'loom_quality_gate_failures_total',
    'Quality gate failures by evaluator',
    ['pipeline_name', 'evaluator_name']
)
```

### Queries

```sql
-- Quality gate pass rate by pipeline
SELECT
  pipeline_name,
  COUNT(*) as total_records,
  SUM(CASE WHEN passed THEN 1 ELSE 0 END) as passed_count,
  ROUND(AVG(CASE WHEN passed THEN 1.0 ELSE 0.0 END) * 100, 2) as pass_rate_pct
FROM record_evaluations
WHERE run_id = 'run_abc123'
GROUP BY pipeline_name;

-- Evaluators causing most failures
SELECT
  evaluator_name,
  COUNT(*) as failure_count,
  AVG(score) as avg_score,
  AVG(threshold) as avg_threshold
FROM record_evaluations
WHERE passed = false
  AND run_id = 'run_abc123'
GROUP BY evaluator_name
ORDER BY failure_count DESC;
```

---

## Migration Guide

### From Ambiguous to Precise Configuration

**Before (Ambiguous)**:
```yaml
evaluate:
  quality_gate: all_pass
```

**After (Precise)**:
```yaml
evaluate:
  evaluators:
    - name: semantic_similarity
      type: semantic
      threshold: 0.8       # EXPLICIT: 0.8 is the pass threshold
    - name: quality_check
      type: custom_criteria
      threshold: 0.75      # EXPLICIT: 0.75 is the pass threshold
  quality_gate: all_pass   # EXPLICIT: Both evaluators must pass
  batch_threshold: 0.95    # OPTIONAL: 95% of batch must pass
```

---

## Conclusion

This specification provides **unambiguous semantics** for all quality gate types with:

1. ✅ **Precise mathematical definitions** of pass conditions
2. ✅ **Concrete examples** demonstrating edge cases
3. ✅ **Executable test cases** validating implementations
4. ✅ **SQL verification queries** for production validation
5. ✅ **Clear failure messages** for debugging

**Next Steps:**
1. Implement quality gate classes following this specification
2. Add unit tests covering all scenarios
3. Update DESIGN_SPEC.md and ARCHITECTURE.md to reference this document
4. Add quality gate validation to pipeline YAML schema
