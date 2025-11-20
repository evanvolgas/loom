"""Load engine for writing data to destinations."""

import asyncio
import logging
from pathlib import Path
from typing import List

import polars as pl

from loom.core.exceptions import ConfigurationError, LoadError
from loom.core.models import LoadConfig, Record
from loom.core.types import DestinationType, RecordStatus

logger = logging.getLogger(__name__)


class LoadEngine:
    """Load records to configured destination."""

    def __init__(self, config: LoadConfig):
        """Initialize load engine.

        Args:
            config: Load stage configuration
        """
        self.config = config
        self.destination_type = self._determine_destination_type()
        logger.info(
            f"Load engine initialized: destination={self.config.destination}, "
            f"type={self.destination_type.value}, mode={self.config.mode}"
        )

    def _determine_destination_type(self) -> DestinationType:
        """Determine destination type from config or file extension.

        Returns:
            DestinationType enum value

        Raises:
            ConfigurationError: If destination type cannot be determined
        """
        if self.config.destination_type:
            return self.config.destination_type

        # Auto-detect from file extension
        path = Path(self.config.destination)
        suffix = path.suffix.lower()

        type_map = {
            ".csv": DestinationType.CSV,
            ".json": DestinationType.JSON,
            ".jsonl": DestinationType.JSONL,
            ".parquet": DestinationType.PARQUET,
        }

        if suffix not in type_map:
            raise ConfigurationError(
                f"Cannot determine destination type from {suffix}. "
                f"Supported: {list(type_map.keys())}"
            )

        return type_map[suffix]

    async def load(self, records: List[Record]) -> int:
        """Load all records to destination.

        Args:
            records: List of records to load

        Returns:
            Number of successfully loaded records

        Raises:
            LoadError: If loading fails
        """
        logger.info(f"Starting load to {self.config.destination}: {len(records)} total records")
        try:
            # Filter to only successfully evaluated records
            records_to_load = [
                r
                for r in records
                if r.status in (RecordStatus.PASSED, RecordStatus.EVALUATED)
            ]

            filtered_out = len(records) - len(records_to_load)
            if filtered_out > 0:
                logger.info(
                    f"Filtered out {filtered_out} records (status not PASSED or EVALUATED)"
                )

            if not records_to_load:
                logger.warning("No records to load after filtering")
                return 0

            logger.info(f"Loading {len(records_to_load)} records to {self.config.destination}")

            # Run synchronous load in thread pool
            loaded_count = await asyncio.to_thread(
                self._load_sync, records_to_load
            )

            logger.info(
                f"Successfully loaded {loaded_count} records to {self.config.destination}"
            )
            return loaded_count
        except Exception as e:
            logger.error(
                f"Load failed to {self.config.destination}: {type(e).__name__}: {e}"
            )
            raise LoadError(f"Failed to load to {self.config.destination}: {e}")

    def _load_sync(self, records: List[Record]) -> int:
        """Synchronous load logic.

        Args:
            records: Records to load

        Returns:
            Number of loaded records
        """
        path = Path(self.config.destination)

        # Ensure parent directory exists
        if not path.parent.exists():
            logger.debug(f"Creating parent directory: {path.parent}")
        path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare data for loading
        logger.debug(f"Preparing {len(records)} records for loading")
        data_rows = []
        for record in records:
            # Combine original data with evaluation results
            row = {
                **record.data,
                "loom_record_id": record.id,
                "loom_transformed_data": record.transformed_data,
                "loom_quality_gate_passed": record.quality_gate_passed,
                **{f"loom_score_{k}": v for k, v in record.evaluation_scores.items()},
            }
            data_rows.append(row)

        # Convert to DataFrame
        df = pl.DataFrame(data_rows)
        logger.debug(f"Created DataFrame: {len(df)} rows, {len(df.columns)} columns")

        # Write based on destination type and mode
        if self.config.mode == "overwrite" or not path.exists():
            logger.debug(f"Writing new file: {path} (mode={self.config.mode})")
            self._write_new(df, path)
        elif self.config.mode == "append":
            logger.debug(f"Appending to existing file: {path}")
            self._append(df, path)
        else:
            logger.error(f"Unsupported load mode: {self.config.mode}")
            raise LoadError(f"Unsupported load mode: {self.config.mode}")

        # Update record status
        for record in records:
            record.status = RecordStatus.LOADED

        logger.debug(f"Marked {len(records)} records as LOADED")
        return len(records)

    def _write_new(self, df: pl.DataFrame, path: Path) -> None:
        """Write new file (overwrite mode).

        Args:
            df: DataFrame to write
            path: Destination path
        """
        if self.destination_type == DestinationType.CSV:
            df.write_csv(path)
        elif self.destination_type == DestinationType.JSON:
            df.write_json(path)
        elif self.destination_type == DestinationType.JSONL:
            df.write_ndjson(path)
        elif self.destination_type == DestinationType.PARQUET:
            df.write_parquet(path)
        else:
            raise LoadError(f"Unsupported destination type: {self.destination_type}")

    def _append(self, df: pl.DataFrame, path: Path) -> None:
        """Append to existing file.

        Args:
            df: DataFrame to append
            path: Destination path
        """
        # Read existing data
        if self.destination_type == DestinationType.CSV:
            existing_df = pl.read_csv(path)
            combined_df = pl.concat([existing_df, df])
            combined_df.write_csv(path)
        elif self.destination_type == DestinationType.JSON:
            # JSON append: read as array, append, write
            existing_df = pl.read_json(path)
            combined_df = pl.concat([existing_df, df])
            combined_df.write_json(path)
        elif self.destination_type == DestinationType.JSONL:
            # JSONL append: read existing, concat, write
            # Note: Polars write_ndjson doesn't support append parameter
            existing_df = pl.read_ndjson(path)
            combined_df = pl.concat([existing_df, df])
            combined_df.write_ndjson(path)
        elif self.destination_type == DestinationType.PARQUET:
            existing_df = pl.read_parquet(path)
            combined_df = pl.concat([existing_df, df])
            combined_df.write_parquet(path)
        else:
            raise LoadError(f"Unsupported destination type: {self.destination_type}")
