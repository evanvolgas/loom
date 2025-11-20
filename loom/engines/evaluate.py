"""Evaluate engine for quality assessment using Arbiter."""

import asyncio
import logging
from typing import List

from arbiter import evaluate as arbiter_evaluate

from loom.core.exceptions import EvaluateError
from loom.core.models import EvaluateConfig, Record
from loom.core.types import QualityGateType, RecordStatus

logger = logging.getLogger(__name__)


class EvaluateEngine:
    """Evaluate records using Arbiter."""

    def __init__(self, config: EvaluateConfig):
        """Initialize evaluate engine.

        Args:
            config: Evaluate stage configuration
        """
        self.config = config
        evaluator_names = [e.name for e in config.evaluators]
        logger.info(
            f"Evaluate engine initialized: evaluators={evaluator_names}, "
            f"quality_gate={config.quality_gate.value}, timeout={config.timeout}s"
        )

    async def evaluate_record(self, record: Record) -> Record:
        """Evaluate a single record.

        Args:
            record: Record with transformed data to evaluate

        Returns:
            Record with evaluation scores and quality gate result

        Raises:
            EvaluateError: If evaluation fails
        """
        logger.debug(f"Starting evaluation for record {record.id}")

        if not record.transformed_data:
            logger.error(f"Record {record.id} has no transformed data to evaluate")
            raise EvaluateError(
                f"Record {record.id} has no transformed data to evaluate"
            )

        try:
            # Get reference text if configured
            reference = None
            if self.config.reference_field and self.config.reference_field in record.data:
                reference = str(record.data[self.config.reference_field])

            # Prepare evaluator list for Arbiter
            evaluator_types = [eval_config.name for eval_config in self.config.evaluators]

            # Call Arbiter evaluate function
            logger.debug(f"Calling Arbiter for record {record.id} with evaluators: {evaluator_types}")
            result = await asyncio.wait_for(
                arbiter_evaluate(
                    output=record.transformed_data,
                    reference=reference,
                    evaluators=evaluator_types,
                ),
                timeout=self.config.timeout,
            )

            # Extract scores from Arbiter result
            for eval_config in self.config.evaluators:
                score_obj = next(
                    (s for s in result.scores if s.name == eval_config.name), None
                )
                if score_obj:
                    record.evaluation_scores[eval_config.name] = score_obj.value
                    logger.debug(
                        f"Record {record.id} - {eval_config.name}: {score_obj.value:.3f}"
                    )

            # Apply quality gate
            record.quality_gate_passed = self._check_quality_gate(record)
            record.status = (
                RecordStatus.PASSED
                if record.quality_gate_passed
                else RecordStatus.FAILED
            )

            gate_status = "PASSED" if record.quality_gate_passed else "FAILED"
            logger.info(
                f"Record {record.id} evaluation complete: quality_gate={gate_status}, "
                f"scores={record.evaluation_scores}"
            )

            return record

        except asyncio.TimeoutError:
            record.status = RecordStatus.ERROR
            record.error = f"Evaluation exceeded timeout ({self.config.timeout}s)"
            logger.error(f"Evaluation timeout for record {record.id}: {self.config.timeout}s")
            raise EvaluateError(record.error)
        except Exception as e:
            record.status = RecordStatus.ERROR
            record.error = str(e)
            logger.error(
                f"Evaluation failed for record {record.id}: {type(e).__name__}: {e}"
            )
            raise EvaluateError(f"Evaluation failed for record {record.id}: {e}")

    def _check_quality_gate(self, record: Record) -> bool:
        """Check if record passes quality gate.

        Args:
            record: Record with evaluation scores

        Returns:
            True if passes, False otherwise
        """
        if self.config.quality_gate == QualityGateType.ALL_PASS:
            return self._check_all_pass(record)
        elif self.config.quality_gate == QualityGateType.MAJORITY_PASS:
            return self._check_majority_pass(record)
        elif self.config.quality_gate == QualityGateType.ANY_PASS:
            return self._check_any_pass(record)
        elif self.config.quality_gate == QualityGateType.WEIGHTED:
            return self._check_weighted(record)
        else:
            raise EvaluateError(f"Unknown quality gate: {self.config.quality_gate}")

    def _check_all_pass(self, record: Record) -> bool:
        """All evaluators must pass their thresholds.

        Mathematical definition: ∀ evaluator: score ≥ threshold
        """
        for eval_config in self.config.evaluators:
            score = record.evaluation_scores.get(eval_config.name)
            if score is None or score < eval_config.threshold:
                return False
        return True

    def _check_majority_pass(self, record: Record) -> bool:
        """Majority of evaluators must pass.

        Mathematical definition: passed_count > total_count / 2
        """
        passed_count = 0
        total_count = len(self.config.evaluators)

        for eval_config in self.config.evaluators:
            score = record.evaluation_scores.get(eval_config.name)
            if score is not None and score >= eval_config.threshold:
                passed_count += 1

        return passed_count > total_count / 2

    def _check_any_pass(self, record: Record) -> bool:
        """At least one evaluator must pass.

        Mathematical definition: ∃ evaluator: score ≥ threshold
        """
        for eval_config in self.config.evaluators:
            score = record.evaluation_scores.get(eval_config.name)
            if score is not None and score >= eval_config.threshold:
                return True
        return False

    def _check_weighted(self, record: Record) -> bool:
        """Weighted average must exceed threshold.

        Mathematical definition: Σ(score × weight) / Σ(weight) ≥ threshold
        """
        if self.config.quality_gate_threshold is None:
            raise EvaluateError(
                "quality_gate_threshold required for weighted quality gate"
            )

        weighted_sum = 0.0
        weight_sum = 0.0

        for eval_config in self.config.evaluators:
            score = record.evaluation_scores.get(eval_config.name)
            if score is not None:
                weighted_sum += score * eval_config.weight
                weight_sum += eval_config.weight

        if weight_sum == 0:
            return False

        weighted_avg = weighted_sum / weight_sum
        return weighted_avg >= self.config.quality_gate_threshold

    async def evaluate_batch(self, records: List[Record]) -> List[Record]:
        """Evaluate a batch of records concurrently.

        Args:
            records: List of transformed records to evaluate

        Returns:
            List of evaluated records with quality gate results

        Raises:
            EvaluateError: If batch evaluation fails
        """
        logger.info(f"Starting batch evaluation: {len(records)} records")

        from loom.core.config import config

        semaphore = asyncio.Semaphore(config.max_concurrent_records)
        logger.debug(f"Using concurrency limit: {config.max_concurrent_records}")

        async def evaluate_with_semaphore(record: Record) -> Record:
            async with semaphore:
                try:
                    return await self.evaluate_record(record)
                except EvaluateError:
                    # Record already has error set
                    return record

        tasks = [evaluate_with_semaphore(record) for record in records]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions and filter to Records only
        evaluated_records: List[Record] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch evaluation failed: {type(result).__name__}: {result}")
                raise EvaluateError(f"Batch evaluation failed: {result}")
            elif isinstance(result, Record):
                evaluated_records.append(result)

        passed_count = sum(1 for r in evaluated_records if r.quality_gate_passed)
        failed_count = sum(1 for r in evaluated_records if not r.quality_gate_passed and r.status != RecordStatus.ERROR)
        error_count = sum(1 for r in evaluated_records if r.status == RecordStatus.ERROR)
        logger.info(
            f"Batch evaluation complete: {passed_count} passed, "
            f"{failed_count} failed quality gate, {error_count} errors out of {len(records)} total"
        )

        return evaluated_records

    def check_batch_threshold(self, records: List[Record]) -> bool:
        """Check if batch passes batch_threshold requirement.

        Args:
            records: List of evaluated records

        Returns:
            True if batch threshold passed, False otherwise
        """
        if self.config.batch_threshold is None:
            return True  # No batch threshold configured

        passed_count = sum(1 for r in records if r.quality_gate_passed)
        total_count = len(records)

        if total_count == 0:
            return True

        pass_rate = passed_count / total_count
        return pass_rate >= self.config.batch_threshold
