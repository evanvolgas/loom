# Loom Examples

This directory contains example pipelines demonstrating Loom's capabilities.

## Customer Sentiment Classification

**Pipeline:** `pipelines/customer_sentiment.yaml`

This example demonstrates a complete AI(E)TL pipeline:
- **Extract**: Read customer reviews from CSV
- **Transform**: Classify sentiment using GPT-4o-mini
- **Evaluate**: Check semantic similarity against expected sentiment
- **Load**: Write results back to CSV

### Prerequisites

1. **Install Loom**:
   ```bash
   cd /path/to/loom
   uv pip install -e ".[dev]"
   ```

2. **Set OpenAI API Key**:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```

3. **Install Arbiter** (Loom's hard dependency):
   ```bash
   pip install arbiter
   ```

### Run the Example

1. **Validate the pipeline**:
   ```bash
   loom validate examples/pipelines/customer_sentiment.yaml
   ```

2. **Run the pipeline**:
   ```bash
   loom run examples/pipelines/customer_sentiment.yaml
   ```

3. **Check the results**:
   ```bash
   cat data/sentiment_results.csv
   ```

   The output will include:
   - Original review data
   - `loom_transformed_data`: LLM-generated sentiment classification
   - `loom_quality_gate_passed`: Whether the record passed quality gates
   - `loom_score_semantic`: Semantic similarity score

### Understanding the Pipeline

#### Extract Stage
```yaml
extract:
  source: data/customer_reviews.csv
  batch_size: 100
```
Reads customer reviews from CSV file.

#### Transform Stage
```yaml
transform:
  prompt: prompts/classify_sentiment.txt
  model: gpt-4o-mini
  provider: openai
  batch_size: 5
  temperature: 0.3
```
Uses GPT-4o-mini to classify sentiment. The prompt template uses `$review_text` variable substitution.

#### Evaluate Stage
```yaml
evaluate:
  evaluators:
    - name: semantic
      type: semantic
      threshold: 0.7
  quality_gate: all_pass
  reference_field: expected_sentiment
```
Evaluates LLM output against the `expected_sentiment` field using semantic similarity.

#### Load Stage
```yaml
load:
  destination: data/sentiment_results.csv
  mode: overwrite
```
Writes results to CSV with Loom metadata columns.

### Quality Gates

This example uses `all_pass` quality gate:
- **Pass Condition**: All evaluators must score ≥ threshold
- **Semantic evaluator**: Checks if LLM output is semantically similar to expected sentiment
- **Threshold**: 0.7 (70% similarity required)

See `QUALITY_GATES.md` for detailed quality gate semantics.

### Customization

#### Change LLM Model
```yaml
transform:
  model: claude-3-5-sonnet
  provider: anthropic
```

#### Add More Evaluators
```yaml
evaluate:
  evaluators:
    - name: semantic
      type: semantic
      threshold: 0.7
    - name: quality_check
      type: custom_criteria
      criteria: "Classification must be POSITIVE, NEGATIVE, or NEUTRAL"
      threshold: 0.8
  quality_gate: all_pass
```

#### Use Different Quality Gate
```yaml
evaluate:
  quality_gate: majority_pass  # >50% of evaluators must pass
```

## Creating Your Own Pipeline

1. **Create data source** (CSV, JSON, JSONL, or Parquet)
2. **Create prompt template** in `prompts/` directory
3. **Create pipeline YAML** in `examples/pipelines/`
4. **Run**: `loom run examples/pipelines/your_pipeline.yaml`

See `DESIGN_SPEC.md` for full pipeline specification.
