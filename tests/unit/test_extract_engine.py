"""Unit tests for ExtractEngine."""

import pytest
from pathlib import Path

from loom.core.models import ExtractConfig, Record
from loom.core.types import SourceType, RecordStatus
from loom.core.exceptions import ExtractError, ConfigurationError
from loom.engines.extract import ExtractEngine


class TestExtractEngine:
    """Test ExtractEngine functionality."""

    def test_init_with_explicit_source_type(self):
        """Test engine initialization with explicit source type."""
        config = ExtractConfig(
            source="tests/fixtures/data/sample.csv",
            source_type=SourceType.CSV,
        )
        engine = ExtractEngine(config)

        assert engine.config == config
        assert engine.source_type == SourceType.CSV

    def test_init_with_auto_detect_csv(self):
        """Test engine initialization with CSV auto-detection."""
        config = ExtractConfig(source="tests/fixtures/data/sample.csv")
        engine = ExtractEngine(config)

        assert engine.source_type == SourceType.CSV

    def test_init_with_auto_detect_json(self):
        """Test engine initialization with JSON auto-detection."""
        config = ExtractConfig(source="tests/fixtures/data/sample.json")
        engine = ExtractEngine(config)

        assert engine.source_type == SourceType.JSON

    def test_init_with_auto_detect_jsonl(self):
        """Test engine initialization with JSONL auto-detection."""
        config = ExtractConfig(source="tests/fixtures/data/sample.jsonl")
        engine = ExtractEngine(config)

        assert engine.source_type == SourceType.JSONL

    def test_init_with_auto_detect_parquet(self):
        """Test engine initialization with Parquet auto-detection."""
        config = ExtractConfig(source="tests/fixtures/data/sample.parquet")
        engine = ExtractEngine(config)

        assert engine.source_type == SourceType.PARQUET

    def test_init_with_unsupported_extension(self):
        """Test engine initialization with unsupported file extension."""
        config = ExtractConfig(source="data.txt")

        with pytest.raises(ConfigurationError) as exc_info:
            ExtractEngine(config)

        assert "Cannot determine source type" in str(exc_info.value)
        assert ".txt" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_extract_csv(self):
        """Test extracting records from CSV file."""
        config = ExtractConfig(source="tests/fixtures/data/sample.csv")
        engine = ExtractEngine(config)

        records = await engine.extract()

        assert len(records) == 5
        assert all(isinstance(r, Record) for r in records)
        assert records[0].data["id"] == 1  # Polars infers numeric types
        assert records[0].data["text"] == "This product is amazing! Best purchase ever."
        assert records[0].data["category"] == "product_review"
        assert records[0].data["expected_sentiment"] == "positive"
        assert records[0].status == RecordStatus.EXTRACTED
        assert records[0].metadata["source_type"] == "csv"

    @pytest.mark.asyncio
    async def test_extract_json(self):
        """Test extracting records from JSON file."""
        config = ExtractConfig(source="tests/fixtures/data/sample.json")
        engine = ExtractEngine(config)

        records = await engine.extract()

        assert len(records) == 5
        assert records[0].data["id"] == "1"
        assert records[0].data["text"] == "This product is amazing! Best purchase ever."
        assert records[0].status == RecordStatus.EXTRACTED
        assert records[0].metadata["source_type"] == "json"

    @pytest.mark.asyncio
    async def test_extract_jsonl(self):
        """Test extracting records from JSONL file."""
        config = ExtractConfig(source="tests/fixtures/data/sample.jsonl")
        engine = ExtractEngine(config)

        records = await engine.extract()

        assert len(records) == 5
        assert records[0].data["id"] == "1"
        assert records[0].data["expected_sentiment"] == "positive"
        assert records[0].status == RecordStatus.EXTRACTED
        assert records[0].metadata["source_type"] == "jsonl"

    @pytest.mark.asyncio
    async def test_extract_parquet(self):
        """Test extracting records from Parquet file."""
        config = ExtractConfig(source="tests/fixtures/data/sample.parquet")
        engine = ExtractEngine(config)

        records = await engine.extract()

        assert len(records) == 5
        assert records[0].data["id"] == "1"
        assert records[0].data["category"] == "product_review"
        assert records[0].status == RecordStatus.EXTRACTED
        assert records[0].metadata["source_type"] == "parquet"

    @pytest.mark.asyncio
    async def test_extract_file_not_found(self):
        """Test extraction fails when file doesn't exist."""
        config = ExtractConfig(source="tests/fixtures/data/nonexistent.csv")
        engine = ExtractEngine(config)

        with pytest.raises(ExtractError) as exc_info:
            await engine.extract()

        assert "Source file not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_extract_record_ids(self):
        """Test that records have proper IDs (row indices)."""
        config = ExtractConfig(source="tests/fixtures/data/sample.csv")
        engine = ExtractEngine(config)

        records = await engine.extract()

        # IDs should be string row indices
        assert records[0].id == "0"
        assert records[1].id == "1"
        assert records[2].id == "2"
        assert records[3].id == "3"
        assert records[4].id == "4"

    @pytest.mark.asyncio
    async def test_extract_record_metadata(self):
        """Test that records have proper metadata."""
        config = ExtractConfig(source="tests/fixtures/data/sample.csv")
        engine = ExtractEngine(config)

        records = await engine.extract()

        for idx, record in enumerate(records):
            assert "source" in record.metadata
            assert "sample.csv" in record.metadata["source"]
            assert record.metadata["source_type"] == "csv"
            assert record.metadata["row_index"] == idx

    @pytest.mark.asyncio
    async def test_extract_batch_first_two(self):
        """Test extracting first 2 records as batch."""
        config = ExtractConfig(source="tests/fixtures/data/sample.csv")
        engine = ExtractEngine(config)

        batch = await engine.extract_batch(offset=0, limit=2)

        assert len(batch) == 2
        assert batch[0].id == "0"
        assert batch[1].id == "1"

    @pytest.mark.asyncio
    async def test_extract_batch_with_offset(self):
        """Test extracting batch with offset."""
        config = ExtractConfig(source="tests/fixtures/data/sample.csv")
        engine = ExtractEngine(config)

        batch = await engine.extract_batch(offset=2, limit=2)

        assert len(batch) == 2
        assert batch[0].id == "2"
        assert batch[1].id == "3"

    @pytest.mark.asyncio
    async def test_extract_batch_no_limit(self):
        """Test extracting batch without limit (all from offset)."""
        config = ExtractConfig(source="tests/fixtures/data/sample.csv")
        engine = ExtractEngine(config)

        batch = await engine.extract_batch(offset=3)

        assert len(batch) == 2  # Records 3 and 4
        assert batch[0].id == "3"
        assert batch[1].id == "4"

    @pytest.mark.asyncio
    async def test_extract_batch_beyond_range(self):
        """Test extracting batch beyond available records."""
        config = ExtractConfig(source="tests/fixtures/data/sample.csv")
        engine = ExtractEngine(config)

        batch = await engine.extract_batch(offset=10, limit=5)

        assert len(batch) == 0  # No records available

    @pytest.mark.asyncio
    async def test_extract_batch_partial(self):
        """Test extracting batch that's partially available."""
        config = ExtractConfig(source="tests/fixtures/data/sample.csv")
        engine = ExtractEngine(config)

        batch = await engine.extract_batch(offset=3, limit=10)

        assert len(batch) == 2  # Only 2 records available from offset 3

    @pytest.mark.asyncio
    async def test_extract_all_records_have_data(self):
        """Test that all extracted records have non-empty data."""
        config = ExtractConfig(source="tests/fixtures/data/sample.csv")
        engine = ExtractEngine(config)

        records = await engine.extract()

        for record in records:
            assert record.data is not None
            assert len(record.data) > 0
            assert "text" in record.data
            assert record.data["text"] != ""

    @pytest.mark.asyncio
    async def test_extract_preserves_data_types(self):
        """Test that data types are inferred by Polars during extraction."""
        config = ExtractConfig(source="tests/fixtures/data/sample.csv")
        engine = ExtractEngine(config)

        records = await engine.extract()

        # Polars infers types from CSV data
        for record in records:
            assert isinstance(record.data["id"], int)  # Numeric IDs inferred as int
            assert isinstance(record.data["text"], str)
            assert isinstance(record.data["category"], str)
            assert isinstance(record.data["expected_sentiment"], str)

    @pytest.mark.asyncio
    async def test_extract_different_formats_same_data(self):
        """Test that different formats yield same logical data."""
        csv_config = ExtractConfig(source="tests/fixtures/data/sample.csv")
        json_config = ExtractConfig(source="tests/fixtures/data/sample.json")
        jsonl_config = ExtractConfig(source="tests/fixtures/data/sample.jsonl")
        parquet_config = ExtractConfig(source="tests/fixtures/data/sample.parquet")

        csv_engine = ExtractEngine(csv_config)
        json_engine = ExtractEngine(json_config)
        jsonl_engine = ExtractEngine(jsonl_config)
        parquet_engine = ExtractEngine(parquet_config)

        csv_records = await csv_engine.extract()
        json_records = await json_engine.extract()
        jsonl_records = await jsonl_engine.extract()
        parquet_records = await parquet_engine.extract()

        # All should have same number of records
        assert len(csv_records) == len(json_records) == len(jsonl_records) == len(parquet_records) == 5

        # First record should have same text across all formats
        assert csv_records[0].data["text"] == json_records[0].data["text"]
        assert csv_records[0].data["text"] == jsonl_records[0].data["text"]
        assert csv_records[0].data["text"] == parquet_records[0].data["text"]
