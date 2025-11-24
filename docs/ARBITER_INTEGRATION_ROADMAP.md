# Arbiter Integration Roadmap

**Version:** 1.0
**Last Updated:** 2025-11-15
**Status:** Active Planning

---

## Overview

This document tracks Loom's integration with [Arbiter](https://github.com/ashita-ai/arbiter), the production-grade LLM evaluation framework. Arbiter serves as Loom's evaluation engine, providing quality gates for AI pipelines.

**Key Relationship:**
- **Arbiter**: Evaluates individual LLM outputs (what to evaluate)
- **Loom**: Orchestrates pipelines with evaluation gates (when/how to evaluate)

---

## Current Arbiter Status

### Version: v0.1.0-alpha
**Phase:** Phase 2.5 Complete (Nov 2025) → Phase 3 Starting Dec 15, 2025

### Currently Available Evaluators ✅

| Evaluator | Purpose | Status | Accuracy |
|-----------|---------|--------|----------|
| **SemanticEvaluator** | Similarity scoring between output and reference | ✅ Implemented | N/A |
| **CustomCriteriaEvaluator** | Domain-specific evaluation with custom criteria | ✅ Implemented | N/A |
| **PairwiseComparisonEvaluator** | A/B testing and model comparison | ✅ Implemented | N/A |

**Loom Integration:** All 3 evaluators are ready for use in Loom quality gates

**Example:**
```yaml
evaluate:
  evaluators:
    - type: semantic
      threshold: 0.8
    - type: custom_criteria
      criteria: "Accurate, helpful, no hallucination"
      threshold: 0.75
  quality_gate: all_pass
```

---

## Upcoming Arbiter Evaluators

### Phase 3: Core Evaluators (Dec 15, 2025 - Jan 5, 2026)

#### 1. FactualityEvaluator (Week 1) 🚧
**Status:** Implementation starts Dec 15, 2025
**Purpose:** Hallucination detection and fact verification
**Accuracy:** 70-80% (LLM-based only)

**Capabilities:**
- Extract atomic claims from outputs
- Classify claims (factual, subjective, opinion)
- Verify claims against reference or LLM knowledge
- Score based on factual accuracy

**Loom Integration Example:**
```yaml
evaluate:
  evaluators:
    - type: factuality
      threshold: 0.85
      config:
        reference_field: source_document  # Optional reference
  quality_gate: all_pass
```

**Output:**
```python
{
  "factuality_score": 0.78,
  "verified_claims": [
    "The Eiffel Tower is 300 meters tall",
    "It was completed in 1889"
  ],
  "non_factual_claims": [
    "The tower weighs 100,000 tons"  # Actual: 10,100 tons
  ],
  "uncertain_claims": []
}
```

#### 2. GroundednessEvaluator (Week 2) 🚧
**Status:** Planned for Dec 22-29, 2025
**Purpose:** RAG system validation (source attribution)

**Capabilities:**
- Identify statements requiring sources
- Map statements to source documents
- Detect ungrounded claims
- Track citation accuracy

**Loom Integration Example:**
```yaml
extract:
  source: postgres://questions

transform:
  type: rag
  prompt: prompts/answer_question.txt
  context: {{ retrieved_documents }}

evaluate:
  evaluators:
    - type: groundedness
      threshold: 0.9
      config:
        source_documents_field: retrieved_documents
  quality_gate: all_pass
```

**Use Case:** Prevent hallucinated answers in RAG pipelines

#### 3. RelevanceEvaluator (Week 3) 🚧
**Status:** Planned for Dec 29, 2025 - Jan 5, 2026
**Purpose:** Query-output alignment assessment

**Capabilities:**
- Analyze query requirements
- Identify addressed vs missing points
- Detect off-topic content
- Score relevance to query

**Loom Integration Example:**
```yaml
evaluate:
  evaluators:
    - type: relevance
      threshold: 0.8
      config:
        query_field: user_question
        expected_topics_field: required_topics
```

---

## Phase 5: Enhanced Factuality (Feb-Mar 2026)

### 🎯 Major Accuracy Improvement: 70-80% → 95-98%

**Timeline:** 6 weeks (Feb-Mar 2026)

### 5.1: Plugin Infrastructure + Tavily (Weeks 1-2)

**External Verification Tools:**

| Plugin | Type | Cost | Purpose |
|--------|------|------|---------|
| **TavilyPlugin** | Web search | Free tier: 1000 req/month | Real-time fact verification |
| **WikipediaPlugin** | Encyclopedia | Free | Well-known facts, historical data |
| **WikidataPlugin** | Knowledge graph | Free | Structured knowledge |
| **ArxivPlugin** | Scientific papers | Free | Scientific claims |
| **PubMedPlugin** | Medical literature | Free | Medical claims |

**Loom Integration Example:**
```yaml
evaluate:
  evaluators:
    - type: factuality
      threshold: 0.90  # Higher threshold with plugins
      config:
        plugins:
          - name: tavily
            api_key: ${TAVILY_API_KEY}
            max_results: 5
            priority: 1
          - name: wikipedia
            max_results: 3
            priority: 2
          - name: arxiv
            max_results: 3
            priority: 3
        use_cache: true
        cache_ttl: 86400  # 24 hours
```

**Enhanced Output:**
```python
{
  "factuality_score": 0.95,  # Higher accuracy with external verification
  "verified_claims": [
    {
      "claim": "The Eiffel Tower is 300 meters tall",
      "is_factual": true,
      "confidence": 0.98,
      "sources": [
        {
          "url": "https://en.wikipedia.org/wiki/Eiffel_Tower",
          "title": "Eiffel Tower - Wikipedia",
          "snippet": "The tower is 300 metres (984 ft) tall..."
        },
        {
          "url": "https://www.toureiffel.paris/en",
          "title": "Official Eiffel Tower Website",
          "snippet": "The Tower is 300 metres high..."
        }
      ]
    }
  ],
  "sources_consulted": 8,
  "verification_tools_used": ["tavily", "wikipedia"]
}
```

### 5.2: Vector Cache with Milvus (Weeks 3-4)

**Performance Optimization:**
- 30x speedup on cache hits (1500ms → 50ms)
- 90% cost reduction on cached claims
- TTL-based expiration (default 7 days)
- Cosine similarity threshold >0.95

**Loom Benefit:** Dramatically faster evaluation for repeated claims

### 5.3: Atomic Claim Decomposition (Week 5)

**Granular Verification:**
- LLM extracts individual verifiable claims
- Each claim verified independently
- Per-claim source attribution
- Claim-level hallucination detection

**Example:**
```
Input: "The Eiffel Tower is 330 meters tall and was built in 1889."

Claims Extracted:
1. "The Eiffel Tower is 330 meters tall" → ✗ False (actual: 300m)
2. "The Eiffel Tower was built in 1889" → ✓ True

Result: Partial hallucination detected
```

---

## Loom Quality Gates with Arbiter

### Current Implementation (Phase 2.5)

**Available Quality Gates:**
1. **all_pass** - All evaluators must pass
2. **majority_pass** - More than 50% must pass
3. **any_pass** - At least one must pass
4. **weighted** - Weighted combination of scores

**Example Pipeline:**
```yaml
name: customer_qa_generation
version: 1.0.0

extract:
  source: postgres://customers/questions

transform:
  type: ai
  prompt: prompts/answer_question.txt
  model: gpt-4o-mini

evaluate:
  evaluators:
    - type: semantic
      threshold: 0.8
    - type: custom_criteria
      criteria: "Helpful, accurate, no hallucination"
      threshold: 0.75
  quality_gate: all_pass

load:
  destination: postgres://customer_service/qa_responses
  on_failure: quarantine
```

### Enhanced with Phase 3 Evaluators (Dec 2025 - Jan 2026)

```yaml
evaluate:
  evaluators:
    - type: semantic
      threshold: 0.8
    - type: factuality  # NEW - Basic LLM-based
      threshold: 0.85
    - type: groundedness  # NEW - For RAG systems
      threshold: 0.9
      config:
        source_documents_field: retrieved_docs
    - type: relevance  # NEW - Query alignment
      threshold: 0.8
  quality_gate: all_pass
```

### Enhanced with Phase 5 Plugins (Feb-Mar 2026)

```yaml
evaluate:
  evaluators:
    - type: factuality  # ENHANCED - External verification
      threshold: 0.90  # Higher threshold now achievable
      config:
        plugins:
          - name: tavily
            max_results: 5
          - name: wikipedia
            max_results: 3
        use_cache: true
  quality_gate: all_pass
```

---

## Accuracy Comparison

### Before Arbiter Phase 5 (Current - Jan 2026)
**FactualityEvaluator (LLM-only):**
- Accuracy: 70-80%
- No source citations
- Knowledge cutoff limitations
- Overconfident on incorrect facts

**Loom Impact:**
- Moderate hallucination detection
- Some false positives/negatives
- Limited trust in production systems

### After Arbiter Phase 5 (Mar 2026)
**FactualityEvaluator (with plugins):**
- Accuracy: 95-98% (multi-source verification)
- Complete source citations
- Real-time web search (Tavily)
- Evidence-based confidence scoring

**Loom Impact:**
- High-confidence hallucination detection
- Production-grade quality gates
- Trusted AI pipeline outputs
- Comprehensive audit trails with citations

---

## Integration Timeline

### Now (Nov 2025) ✅
- SemanticEvaluator available
- CustomCriteriaEvaluator available
- PairwiseComparisonEvaluator available

### Dec 15, 2025 - Jan 5, 2026 🚧
- FactualityEvaluator (basic, 70-80%)
- GroundednessEvaluator
- RelevanceEvaluator

### Feb-Mar 2026 ⏳
- FactualityEvaluator enhanced with Tavily plugin
- 95-98% accuracy with external verification
- Vector cache for performance
- Atomic claim decomposition

### April 2026 ⏳
- Arbiter v1.0 release
- Loom v1.0 release with complete evaluator suite

---

## Cost Analysis

### Phase 3: Basic Factuality (LLM-only)

**Per Evaluation:**
- LLM calls: ~$0.0004 (claim extraction + verification)
- Total: ~$0.0004

**10,000 evaluations/month:** ~$4

### Phase 5: Enhanced Factuality (with plugins)

**Per Evaluation:**
- LLM calls: ~$0.0004
- Tavily API (after free tier): ~$0.003
- Total: ~$0.0034

**10,000 evaluations/month:**
- Free tier coverage: 1,000 evals (Tavily free)
- Paid evals: 9,000 × $0.0034 = ~$31
- **Total: ~$31/month** (vs $4 LLM-only)

**ROI Calculation:**
- Accuracy improvement: 70-80% → 95-98%
- False positive reduction: ~60-80%
- Production confidence: High enough for customer-facing systems
- **Cost increase: 8x | Accuracy improvement: 20-30%**

---

## Developer Guidelines

### Using Arbiter Evaluators in Loom Pipelines

**1. Check Evaluator Availability**
- Consult this document for current evaluator status
- Phase 3 evaluators: Available Dec 15, 2025+
- Phase 5 enhancements: Available Feb 2026+

**2. Configure Appropriately**
```yaml
# Basic configuration (Phase 3)
evaluate:
  evaluators:
    - type: factuality
      threshold: 0.85

# Advanced configuration (Phase 5)
evaluate:
  evaluators:
    - type: factuality
      threshold: 0.90
      config:
        plugins:
          - name: tavily
            api_key: ${TAVILY_API_KEY}
        use_cache: true
```

**3. Set Realistic Thresholds**
- **Phase 3 (LLM-only):** threshold: 0.75-0.85
- **Phase 5 (with plugins):** threshold: 0.85-0.95

**4. Monitor Performance**
```bash
# Check evaluation metrics
loom metrics <pipeline_name> --evaluator factuality

# View quarantined records
loom quarantine list --pipeline <pipeline_name>
```

---

## Migration Guide

### Migrating to Enhanced Factuality (Phase 5)

**Before (Phase 3):**
```yaml
evaluate:
  evaluators:
    - type: factuality
      threshold: 0.80  # Lower threshold for LLM-only
```

**After (Phase 5):**
```yaml
evaluate:
  evaluators:
    - type: factuality
      threshold: 0.90  # Higher threshold with plugins
      config:
        plugins:
          - name: tavily
            api_key: ${TAVILY_API_KEY}
            max_results: 5
          - name: wikipedia
            max_results: 3
        use_cache: true
        cache_ttl: 86400
```

**Steps:**
1. Add Tavily API key to environment: `export TAVILY_API_KEY=tvly-xxxxx`
2. Update pipeline YAML with plugin config
3. Increase quality gate threshold (0.80 → 0.90)
4. Monitor evaluation metrics for 1 week
5. Adjust thresholds based on observed accuracy

---

## Related Documentation

- **Arbiter Repository:** https://github.com/ashita-ai/arbiter
- **Arbiter DESIGN_SPEC.md:** Vision and architecture
- **Arbiter ROADMAP.md:** Full development timeline
- **Arbiter Tools Architecture:** `/arbiter/docs/TOOLS_PLUGIN_ARCHITECTURE.md`
- **Loom QUALITY_GATES.md:** Quality gate specifications
- **Loom ARCHITECTURE.md:** System architecture

---

## Questions & Feedback

**For Arbiter Issues:**
- Repository: https://github.com/ashita-ai/arbiter/issues

**For Loom Integration:**
- Repository: https://github.com/evanvolgas/loom/issues

---

**Last Updated:** 2025-11-15 | **Next Review:** Dec 15, 2025 (Phase 3 launch)
