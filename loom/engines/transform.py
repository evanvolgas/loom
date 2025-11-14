"""Transform engine for LLM-based data transformation."""

import asyncio
import logging
from pathlib import Path
from string import Template
from typing import List, Optional

from arbiter.core.llm_client import LLMClient

from loom.core.exceptions import ConfigurationError, TransformError
from loom.core.models import Record, TransformConfig
from loom.core.types import RecordStatus
from loom.resilience import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpenError,
)

logger = logging.getLogger(__name__)


class TransformEngine:
    """Transform records using LLM."""

    def __init__(self, config: TransformConfig):
        """Initialize transform engine.

        Args:
            config: Transform stage configuration
        """
        self.config = config
        self.prompt_template = self._load_prompt_template()
        self.llm_client: Optional[LLMClient] = None
        self.circuit_breaker = CircuitBreaker(
            name="transform_llm",
            config=CircuitBreakerConfig(
                failure_threshold=5,
                timeout_seconds=60,
                success_threshold=3,
            ),
        )
        logger.info(
            f"Transform engine initialized: model={self.config.model}, "
            f"provider={self.config.provider}, timeout={self.config.timeout}s, "
            f"temperature={self.config.temperature}"
        )

    def _load_prompt_template(self) -> Template:
        """Load prompt template from file.

        Returns:
            String Template for prompt

        Raises:
            ConfigurationError: If prompt file not found or invalid
        """
        prompt_path = Path(self.config.prompt)

        if not prompt_path.exists():
            logger.error(f"Prompt template not found: {prompt_path}")
            raise ConfigurationError(f"Prompt template not found: {prompt_path}")

        try:
            with open(prompt_path, "r") as f:
                content = f.read()
            logger.debug(f"Loaded prompt template from {prompt_path} ({len(content)} chars)")
            return Template(content)
        except Exception as e:
            logger.error(f"Failed to load prompt template: {type(e).__name__}: {e}")
            raise ConfigurationError(f"Failed to load prompt template: {e}")

    async def _get_llm_client(self) -> LLMClient:
        """Get or create LLM client.

        Returns:
            Initialized LLM client
        """
        if self.llm_client is None:
            from arbiter.core.llm_client import LLMManager

            logger.debug(f"Initializing LLM client: model={self.config.model}")
            self.llm_client = await LLMManager.get_client(
                model=self.config.model,
            )
            logger.info(f"LLM client initialized: {self.config.model}")
        return self.llm_client

    def _build_prompt(self, record: Record) -> str:
        """Build prompt from template and record data.

        Args:
            record: Record to transform

        Returns:
            Formatted prompt string
        """
        # Substitute record data into template
        try:
            return self.prompt_template.substitute(**record.data)
        except KeyError as e:
            raise TransformError(
                f"Missing template variable in record data: {e}. "
                f"Available keys: {list(record.data.keys())}"
            )

    async def transform_record(self, record: Record) -> Record:
        """Transform a single record using LLM.

        Args:
            record: Record to transform

        Returns:
            Transformed record with updated status

        Raises:
            TransformError: If transformation fails
        """
        logger.debug(f"Starting transformation for record {record.id}")
        try:
            # Build prompt
            prompt = self._build_prompt(record)
            logger.debug(f"Built prompt for record {record.id} ({len(prompt)} chars)")

            # Get LLM client
            client = await self._get_llm_client()

            # Call LLM with circuit breaker protection
            async def llm_call():
                async with asyncio.timeout(self.config.timeout):
                    messages = [{"role": "user", "content": prompt}]
                    response = await client.complete(
                        messages=messages,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                    )
                    return response.content

            try:
                transformed_text = await self.circuit_breaker.call(llm_call)
                logger.debug(
                    f"LLM transformation successful for record {record.id} "
                    f"({len(transformed_text)} chars)"
                )
            except CircuitBreakerOpenError as e:
                # Circuit breaker is open - use fallback or fail gracefully
                logger.error(f"LLM circuit breaker open for record {record.id}: {e}")
                raise TransformError(f"LLM circuit breaker open: {e}")

            # Update record
            record.transformed_data = transformed_text
            record.status = RecordStatus.TRANSFORMED
            logger.info(f"Successfully transformed record {record.id}")

            return record

        except asyncio.TimeoutError:
            record.status = RecordStatus.ERROR
            record.error = f"LLM call exceeded timeout ({self.config.timeout}s)"
            logger.error(f"Timeout transforming record {record.id}: {self.config.timeout}s")
            raise TransformError(record.error)
        except Exception as e:
            record.status = RecordStatus.ERROR
            record.error = str(e)
            logger.error(
                f"Transform failed for record {record.id}: {type(e).__name__}: {e}"
            )
            raise TransformError(f"Transform failed for record {record.id}: {e}")

    async def transform_batch(self, records: List[Record]) -> List[Record]:
        """Transform a batch of records concurrently.

        Args:
            records: List of records to transform

        Returns:
            List of transformed records

        Raises:
            TransformError: If batch transformation fails
        """
        logger.info(f"Starting batch transformation: {len(records)} records")

        # Transform records concurrently with semaphore to limit concurrency
        from loom.core.config import config

        semaphore = asyncio.Semaphore(config.max_concurrent_records)
        logger.debug(f"Using concurrency limit: {config.max_concurrent_records}")

        async def transform_with_semaphore(record: Record) -> Record:
            async with semaphore:
                try:
                    return await self.transform_record(record)
                except TransformError:
                    # Record already has error set
                    return record

        tasks = [transform_with_semaphore(record) for record in records]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions
        transformed_records = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Batch transform failed: {type(result).__name__}: {result}")
                raise TransformError(f"Batch transform failed: {result}")
            transformed_records.append(result)

        success_count = sum(1 for r in transformed_records if r.status == RecordStatus.TRANSFORMED)
        error_count = sum(1 for r in transformed_records if r.status == RecordStatus.ERROR)
        logger.info(
            f"Batch transformation complete: {success_count} successful, "
            f"{error_count} errors out of {len(records)} total"
        )

        return transformed_records

    async def close(self) -> None:
        """Clean up resources."""
        if self.llm_client:
            logger.debug("Closing LLM client")
            from arbiter.core.llm_client import LLMManager
            await LLMManager.close()
            logger.info("LLM client closed")
