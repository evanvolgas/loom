"""Extract engine for reading data from sources."""

import asyncio
import logging
from pathlib import Path
from typing import List, Optional

import polars as pl

from loom.core.exceptions import ConfigurationError, ExtractError
from loom.core.models import ExtractConfig, Record
from loom.core.types import RecordStatus, SourceType

logger = logging.getLogger(__name__)


class ExtractEngine:
    """Extract records from configured data source."""

    def __init__(self, config: ExtractConfig):
        """Initialize extract engine.

        Args:
            config: Extract stage configuration
        """
        self.config = config
        self.source_type = self._determine_source_type()
        logger.info(
            f"Extract engine initialized: source={self.config.source}, "
            f"type={self.source_type.value}, batch_size={self.config.batch_size}"
        )

    def _determine_source_type(self) -> SourceType:
        """Determine source type from config or file extension.

        Returns:
            SourceType enum value

        Raises:
            ConfigurationError: If source type cannot be determined
        """
        if self.config.source_type:
            return self.config.source_type

        # Auto-detect from file extension
        path = Path(self.config.source)
        suffix = path.suffix.lower()

        type_map = {
            ".csv": SourceType.CSV,
            ".json": SourceType.JSON,
            ".jsonl": SourceType.JSONL,
            ".parquet": SourceType.PARQUET,
        }

        if suffix not in type_map:
            raise ConfigurationError(
                f"Cannot determine source type from {suffix}. "
                f"Supported: {list(type_map.keys())}"
            )

        return type_map[suffix]

    async def extract(self) -> List[Record]:
        """Extract all records from source.

        Returns:
            List of Record objects

        Raises:
            ExtractError: If extraction fails
        """
        logger.info(f"Starting extraction from {self.config.source}")
        try:
            # Run synchronous extraction in thread pool
            records = await asyncio.to_thread(self._extract_sync)
            logger.info(
                f"Successfully extracted {len(records)} records from {self.config.source}"
            )
            return records
        except Exception as e:
            logger.error(
                f"Extraction failed from {self.config.source}: {type(e).__name__}: {e}"
            )
            raise ExtractError(f"Failed to extract from {self.config.source}: {e}")

    def _extract_sync(self) -> List[Record]:
        """Synchronous extraction logic.

        Returns:
            List of Record objects
        """
        path = Path(self.config.source)

        if not path.exists():
            logger.error(f"Source file not found: {path}")
            raise ExtractError(f"Source file not found: {path}")

        logger.debug(f"Reading {self.source_type.value} file: {path}")

        # Read data based on source type
        if self.source_type == SourceType.CSV:
            df = pl.read_csv(path)
        elif self.source_type == SourceType.JSON:
            df = pl.read_json(path)
        elif self.source_type == SourceType.JSONL:
            df = pl.read_ndjson(path)
        elif self.source_type == SourceType.PARQUET:
            df = pl.read_parquet(path)
        else:
            logger.error(f"Unsupported source type: {self.source_type}")
            raise ExtractError(f"Unsupported source type: {self.source_type}")

        logger.debug(f"DataFrame loaded: {len(df)} rows, {len(df.columns)} columns")

        # Convert DataFrame to Record objects
        records = []
        for idx, row in enumerate(df.iter_rows(named=True)):
            record = Record(
                id=str(idx),  # Use row index as ID (can be customized)
                data=row,
                metadata={
                    "source": str(path),
                    "source_type": self.source_type.value,
                    "row_index": idx,
                },
                status=RecordStatus.EXTRACTED,
            )
            records.append(record)

        return records

    async def extract_batch(
        self, offset: int = 0, limit: Optional[int] = None
    ) -> List[Record]:
        """Extract records in batches.

        Args:
            offset: Starting row index
            limit: Maximum number of records to extract

        Returns:
            List of Record objects

        Raises:
            ExtractError: If extraction fails
        """
        logger.info(f"Extracting batch: offset={offset}, limit={limit}")
        try:
            all_records = await self.extract()

            end = offset + limit if limit else None
            batch = all_records[offset:end]
            logger.info(f"Batch extracted: {len(batch)} records (offset={offset}, limit={limit})")
            return batch
        except Exception as e:
            logger.error(f"Batch extraction failed: {type(e).__name__}: {e}")
            raise ExtractError(f"Failed to extract batch: {e}")
