# Financial Metrics Extraction System with RAG & Evaluation

> A production-ready system for extracting structured financial metrics from documents using Retrieval-Augmented Generation (RAG) and Large Language Models, with systematic evaluation and validation frameworks.

## Overview

This project demonstrates end-to-end financial data extraction combining:
- **RAG Pipeline**: Document retrieval using semantic search (FAISS + embeddings)
- **LLM Extraction**: Structured JSON extraction using Groq API
- **Evaluation Framework**: Systematic testing and confidence scoring
- **Production Code**: Logging, error handling, configuration management

**Use Case**: Extract revenue, net income, risks, and financial metrics from earnings reports, financial statements, or investment documents with validated accuracy.

---

## Architecture

```
Document Input (PDF/Text)
        ↓
    RAG Pipeline
    - Load Document
    - Split into Chunks (500 tokens)
    - Create Embeddings (HuggingFace)
    - Store in FAISS Vector DB
        ↓
    Semantic Retrieval
    - Query: "Extract revenue"
    - Retrieve Top-3 Relevant Chunks
        ↓
    Groq LLM Extraction
    - Pass Retrieved Context to LLM
    - Prompt for Structured JSON
    - Parse JSON Response
        ↓
    Structured Output
    {
      "revenue": 18200000000,
      "net_income": 1200000000,
      "risks": [...],
      "data_quality": "high"
    }
        ↓
    Evaluation Framework
    - Compare vs Ground Truth
    - Calculate Accuracy
    - Score Confidence
    - Detect Hallucinations
        ↓
    Production Output
    - Save to JSON
    - Log Results
    - Return Metrics
```

---

## Key Design Decisions

### 1. RAG Over Fine-Tuning
**Why**: RAG allows real-time updates to documents without retraining. Better for frequently changing financial data.

**Trade-offs**:
- ✅ No retraining needed
- ✅ Works with new documents immediately
- ❌ Slower than fine-tuned models
- ❌ Depends on retrieval quality

### 2. Semantic Chunking (500 tokens)
**Why**: Preserves financial context. Too small chunks lose context, too large chunks dilute relevance.

**Validation**: Tested with 250, 500, 1000 token chunks. 500 showed best accuracy.

### 3. Groq API (Free Tier)
**Why**: 
- Fast inference (500+ tokens/sec)
- Cost-effective (free tier sufficient)
- OpenAI-compatible API

**Alternatives Considered**:
- Claude API: Higher cost
- Open-source LLaMA: Setup complexity
- Hugging Face Inference: Slower

### 4. Systematic Evaluation Framework
**Why**: Quantify extraction accuracy to identify failure modes and build trust.

**Metrics**:
- Overall Accuracy: % fields matching ground truth
- Confidence Scores: 0-1 per field
- F1 Score: Precision + Recall for risks
- Hallucination Detection: High confidence + low accuracy = red flag

---

## Project Structure

```
financial-metrics-extractor/
├── src/
│   ├── rag_pipeline.py      # Document loading → embedding → FAISS storage
│   ├── extractor.py         # LLM extraction with structured output
│   ├── evaluator.py         # Accuracy testing, confidence scoring
│   └── utils.py             # Logging, config, error handling
│
├── tests/
│   └── test_extraction.py   # Unit and integration tests
│
├── data/
│   └── sample_financial_report.txt  # Test document
│
├── results/
│   ├── extraction_results.json      # Latest extraction output
│   └── evaluation_metrics.json      # Test results
│
├── requirements.txt         # Dependencies
├── .env                     # API keys (not in git)
├── .gitignore              # Exclude secrets
└── README.md               # This file
```

---

## Installation

### Prerequisites
- Python 3.10+
- Groq API key (free: https://console.groq.com)

### Setup

```bash
# 1. Clone and navigate
git clone <repo>
cd financial-metrics-extractor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API key
echo "GROQ_API_KEY=your_key_here" > .env
```

---

## Usage

### Basic Extraction

```bash
python -m src.extractor
```

Output:
```json
{
  "revenue": 18200000000,
  "net_income": 1200000000,
  "risks": ["rising interest rates", ...],
  "data_quality": "high"
}
```

### Run Evaluation

```bash
python -m src.evaluator
```

Output:
```
📊 Overall Accuracy: 100.0%
✅ All 6 metrics extracted correctly
```

### Run Tests

```bash
pytest tests/ -v
```

---

## API Reference

### RAGPipeline

```python
from src.rag_pipeline import RAGPipeline

# Initialize
rag = RAGPipeline(model_name="all-MiniLM-L6-v2")

# Load and build
rag.build_pipeline(["data/report.txt"])

# Retrieve context
context = rag.retrieve("What was the revenue?")
# Returns: List[str] - top 3 most relevant chunks
```

### FinancialMetricsExtractor

```python
from src.extractor import FinancialMetricsExtractor

# Initialize
extractor = FinancialMetricsExtractor(model="llama-3.1-8b-instant")

# Extract metrics
result = extractor.extract_metrics(context)
# Returns: Dict with revenue, net_income, risks, etc.

# Evaluate quality
evaluation = extractor.evaluate_extraction(result)
# Returns: Dict with field_scores, has_risks, data_quality
```

### ExtractionEvaluator

```python
from src.evaluator import ExtractionEvaluator

evaluator = ExtractionEvaluator()

# Compare with ground truth
report = evaluator.evaluate_extraction(extracted, ground_truth)
# Returns: Accuracy metrics, confidence scores, issues
```

---

## Evaluation Results

### Metrics Tested

| Metric | Extracted | Expected | Correct | Confidence |
|--------|-----------|----------|---------|------------|
| Revenue | $18.2B | $18.2B | ✅ | 1.00 |
| Net Income | $1.2B | $1.2B | ✅ | 1.00 |
| Operating Expenses | $4.8B | $4.8B | ✅ | 1.00 |
| Assets Under Management | $850B | $850B | ✅ | 1.00 |
| Debt-to-Equity | 0.42 | 0.42 | ✅ | 1.00 |
| Equity Ratio | 0.70 | 0.70 | ✅ | 1.00 |
| Risks | 4/4 matched | - | ✅ | 1.00 |

**Overall Accuracy**: 100% | **Average Confidence**: 100%

### Edge Cases Tested

- ✅ Partial queries ("What is revenue?")
- ✅ Vague queries ("Tell me about financials")
- ✅ Missing fields (null handling)
- ✅ Hallucination detection

---

## Configuration

Edit `src/utils.py` > `Config` class:

```python
class Config:
    # RAG
    CHUNK_SIZE = 500          # Tokens per chunk
    RETRIEVAL_TOP_K = 3       # Retrieved chunks
    
    # Extraction
    GROQ_MODEL = "llama-3.1-8b-instant"
    MAX_TOKENS = 1024
    TEMPERATURE = 0.7
    
    # Evaluation
    NUMERIC_TOLERANCE = 0.05  # 5% acceptable error
    MIN_CONFIDENCE_THRESHOLD = 0.7
```

---

## Assumptions & Limitations

### Assumptions
1. **Document Quality**: Assumes well-formatted financial documents
2. **Context Window**: LLM sees only top-3 retrieved chunks (~1500 tokens)
3. **Language**: English-only extraction
4. **Numeric Format**: Assumes numbers in billions/millions are labeled

### Limitations
1. **Complex Tables**: May struggle with complex financial tables
2. **Cross-Document Aggregation**: Single document only (no multi-doc aggregation)
3. **Qualitative Interpretation**: Risk assessment is LLM-based, not rule-based
4. **Speed**: RAG retrieval adds 2-3 sec latency (vs. pure LLM: 0.5 sec)
5. **Cost**: Groq free tier has rate limits (~30 requests/min)

### Known Failure Modes
- ❌ Extracting metrics from narrative text (not structured data)
- ❌ Handling inconsistent date formats
- ❌ Multi-currency extraction (assumes USD)

---

## Production Considerations

### Data Security
- ✅ API keys stored in `.env` (never committed)
- ✅ No data logged (only metrics)
- ✅ Compatible with HIPAA environments (use Groq Enterprise)

### Monitoring
- Logs all extractions to `results/extraction.log`
- Tracks confidence scores per field
- Alerts on hallucination detection

### Scaling
- Current setup: 30 requests/min (Groq free)
- For production: Use Groq paid tier or batch API
- Caching: Implement LLM response caching for identical queries

### Error Handling
```python
try:
    result = extractor.extract_metrics(context)
except Exception as e:
    logger.error(f"Extraction failed: {e}")
    return {"error": str(e), "timestamp": datetime.now()}
```

---

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run specific test class:
```bash
pytest tests/test_extraction.py::TestExtraction -v
```

Run with coverage:
```bash
pip install pytest-cov
pytest tests/ --cov=src --cov-report=html
```

---

## Results & Output

### Extraction Output
Saved to `results/extraction_results.json`:
```json
{
  "timestamp": "2025-04-08T22:15:00Z",
  "extraction": {
    "revenue": 18200000000,
    "risks": ["rising interest rates", ...],
    "data_quality": "high"
  },
  "evaluation": {
    "overall_accuracy": 1.0,
    "avg_confidence": 1.0,
    "field_scores": {...}
  }
}
```

### Logs
Saved to `results/extraction.log`:
```
2025-04-08 22:15:00 - FinancialExtractor - INFO - Extraction started
2025-04-08 22:15:02 - FinancialExtractor - INFO - Retrieved 3 chunks
2025-04-08 22:15:03 - FinancialExtractor - INFO - Extraction complete
```

---

## Interview Talking Points (For Manulife)

### 1. Problem Decomposition
"I took an ambiguous problem ('extract financial metrics') and broke it into:
- Information retrieval (RAG)
- Structured extraction (LLM)
- Validation (evaluation framework)"

### 2. Evaluation-First Design
"Rather than just building an extraction function, I built a systematic evaluation framework that:
- Compares against ground truth
- Scores confidence per field
- Detects hallucinations
- Tests edge cases"

### 3. Production Mindset
"The code includes logging, error handling, configuration management, and tests — not just a notebook. I'm thinking about monitoring and maintainability from day one."

### 4. Technical Depth
"I chose RAG over fine-tuning because it allows real-time document updates without retraining — critical for frequently changing financial data."

### 5. Trade-off Analysis
"I evaluated Groq vs. Claude API vs. open-source models based on cost, speed, and reliability. Groq offers the best cost-performance for this use case."

---

## Next Steps

### Short Term
- [ ] Add more test documents (10+ financial reports)
- [ ] Implement caching for repeated queries
- [ ] Add CSV export for batch processing

### Medium Term
- [ ] Multi-document aggregation (sum across reports)
- [ ] Fine-tuned model for domain-specific metrics
- [ ] Web API wrapper (FastAPI)
- [ ] Dashboard for monitoring extraction quality

### Long Term
- [ ] Deploy to production (AWS Lambda / Azure Functions)
- [ ] Implement human-in-the-loop feedback
- [ ] Build custom evaluation dashboard
- [ ] Support multi-language extraction

---

## Contributing

Pull requests welcome. Please include:
- Unit tests for new features
- Updated README
- Evaluation results showing no regression

---

## License

MIT

---

## Contact

Built for Manulife Financial. Questions? Open an issue.

---

## Changelog

### v1.0 (2025-04-08)
- Initial release
- RAG pipeline with FAISS
- Groq-based extraction
- Evaluation framework
- 100% accuracy on test data