#!/usr/bin/env python3
"""Unit and integration tests for Financial Metrics Extraction System."""

import pytest
import json
from pathlib import Path
from src.rag_pipeline import RAGPipeline
from src.extractor import FinancialMetricsExtractor
from src.evaluator import ExtractionEvaluator, GROUND_TRUTH
from src.utils import Config

class TestRAGPipeline:
    """Test RAG pipeline functionality."""
    
    @pytest.fixture
    def rag(self):
        """Initialize RAG pipeline."""
        rag = RAGPipeline()
        rag.build_pipeline(["data/sample_financial_report.txt"])
        return rag
    
    def test_pipeline_initialization(self, rag):
        """Test that RAG pipeline initializes correctly."""
        assert rag.vector_store is not None
        assert rag.retriever is not None
    
    def test_document_retrieval(self, rag):
        """Test that retrieval works."""
        query = "What was the revenue?"
        results = rag.retrieve(query)
        
        assert len(results) > 0
        assert all(isinstance(r, str) for r in results)
    
    def test_retrieval_relevance(self, rag):
        """Test that retrieved content is relevant."""
        query = "revenue Q3"
        results = rag.retrieve(query)
        
        # At least one result should contain revenue information
        combined = " ".join(results).lower()
        assert "revenue" in combined or "billion" in combined


class TestExtraction:
    """Test extraction functionality."""
    
    @pytest.fixture
    def setup(self):
        """Setup extraction test."""
        rag = RAGPipeline()
        rag.build_pipeline(["data/sample_financial_report.txt"])
        extractor = FinancialMetricsExtractor()
        return rag, extractor
    
    def test_extractor_initialization(self, setup):
        """Test that extractor initializes correctly."""
        rag, extractor = setup
        assert extractor.client is not None
        assert extractor.model == "llama-3.1-8b-instant"
    
    def test_extraction_returns_json(self, setup):
        """Test that extraction returns valid JSON."""
        rag, extractor = setup
        context = rag.retrieve("Extract all financial metrics")
        result = extractor.extract_metrics(context)
        
        assert isinstance(result, dict)
        assert not ("error" in result and len(result) == 1)
    
    def test_extraction_structure(self, setup):
        """Test that extracted JSON has required structure."""
        rag, extractor = setup
        context = rag.retrieve("Extract all financial metrics")
        result = extractor.extract_metrics(context)
        
        required_fields = ["revenue", "net_income", "risks", "data_quality"]
        for field in required_fields:
            assert field in result or "error" in result
    
    def test_extraction_numeric_fields(self, setup):
        """Test that numeric fields are numeric."""
        rag, extractor = setup
        context = rag.retrieve("Extract all financial metrics")
        result = extractor.extract_metrics(context)
        
        if "error" not in result:
            numeric_fields = ["revenue", "net_income", "operating_expenses"]
            for field in numeric_fields:
                if field in result and result[field] is not None:
                    assert isinstance(result[field], (int, float))


class TestEvaluation:
    """Test evaluation framework."""
    
    @pytest.fixture
    def evaluator(self):
        """Initialize evaluator."""
        return ExtractionEvaluator()
    
    def test_numeric_comparison(self, evaluator):
        """Test numeric value comparison."""
        is_correct, confidence = evaluator.compare_numeric(100, 100)
        assert is_correct is True
        assert confidence == 1.0
    
    def test_numeric_comparison_with_tolerance(self, evaluator):
        """Test numeric comparison with tolerance."""
        # 105 vs 100 = 5% error, should be within tolerance
        is_correct, confidence = evaluator.compare_numeric(105, 100, tolerance=0.05)
        assert is_correct is True
        assert confidence >= 0.95
    
    def test_numeric_comparison_outside_tolerance(self, evaluator):
        """Test numeric comparison outside tolerance."""
        # 110 vs 100 = 10% error, outside 5% tolerance
        is_correct, confidence = evaluator.compare_numeric(110, 100, tolerance=0.05)
        assert is_correct is False
        assert confidence < 0.95
    
    def test_list_comparison(self, evaluator):
        """Test list comparison."""
        extracted = ["risk1", "risk2", "risk3"]
        expected = ["risk1", "risk2", "risk3"]
        
        precision, recall = evaluator.compare_list(extracted, expected)
        assert precision == 1.0
        assert recall == 1.0
    
    def test_list_comparison_partial_match(self, evaluator):
        """Test list comparison with partial match."""
        extracted = ["risk1", "risk2"]
        expected = ["risk1", "risk2", "risk3"]
        
        precision, recall = evaluator.compare_list(extracted, expected)
        assert 0 < precision <= 1.0
        assert 0 < recall < 1.0


class TestConfiguration:
    """Test configuration management."""
    
    def test_config_exists(self):
        """Test that config is accessible."""
        config = Config.to_dict()
        assert isinstance(config, dict)
    
    def test_config_has_required_fields(self):
        """Test that config has required fields."""
        config = Config.to_dict()
        required = ["embedding_model", "groq_model", "chunk_size", "max_tokens"]
        
        for field in required:
            assert field in config


class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test complete extraction pipeline."""
        # Setup
        rag = RAGPipeline()
        rag.build_pipeline(["data/sample_financial_report.txt"])
        extractor = FinancialMetricsExtractor()
        evaluator = ExtractionEvaluator()
        
        # Extract
        context = rag.retrieve("Extract all financial metrics")
        extracted = extractor.extract_metrics(context)
        
        # Evaluate
        evaluation = evaluator.evaluate_extraction(extracted, GROUND_TRUTH)
        
        # Assertions
        assert extracted is not None
        assert evaluation is not None
        assert "overall_accuracy" in evaluation
        assert evaluation["overall_accuracy"] >= 0.5  # At least 50% accuracy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])