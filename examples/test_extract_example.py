"""Example demonstrating data extraction."""

import asyncio
from loom.core.models import ExtractConfig
from loom.core.types import SourceType
from loom.engines.extract import ExtractEngine


async def main():
    print("📂 Testing ExtractEngine\n")

    # Example 1: Extract CSV data
    print("Example 1: Extracting CSV data")
    config = ExtractConfig(
        source="examples/data/sample.csv",
        source_type=SourceType.CSV,
    )

    engine = ExtractEngine(config)
    records = await engine.extract()

    print(f"  ✅ Extracted {len(records)} records")
    for i, record in enumerate(records[:3], 1):
        print(f"  Record {i}: {record.data}")
    if len(records) > 3:
        print(f"  ... and {len(records) - 3} more records\n")
    else:
        print()

    # Example 2: Extract with batching
    print("Example 2: Extract first 2 records only")
    records_batch = await engine.extract_batch(limit=2)
    print(f"  ✅ Extracted {len(records_batch)} records (limit=2)")
    for i, record in enumerate(records_batch, 1):
        print(f"  Record {i}: {record.data}")
    print()

    # Example 3: Extract with offset
    print("Example 3: Extract records 3-4 (offset=2, limit=2)")
    records_offset = await engine.extract_batch(offset=2, limit=2)
    print(f"  ✅ Extracted {len(records_offset)} records")
    for i, record in enumerate(records_offset, 1):
        print(f"  Record {i}: {record.data}")
    print()

    # Verify record structure
    print("Example 4: Verify record structure")
    first_record = records[0]
    print(f"  ✅ Record ID: {first_record.id}")
    print(f"  ✅ Record Status: {first_record.status.value}")
    print(f"  ✅ Record Data Keys: {list(first_record.data.keys())}")
    print(f"  ✅ Has Metadata: {first_record.metadata is not None}")
    print()

    print("✅ All extraction examples completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
