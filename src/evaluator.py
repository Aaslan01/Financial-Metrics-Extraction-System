#!/usr/bin/env python3
"""Evaluation Framework for Financial Metrics Extraction."""

import json
from typing import Dict, List, Tuple, Optional
from src.extractor import FinancialMetricsExtractor
from src.rag_pipeline import RAGPipeline


class ExtractionEvaluator:
    """Evaluate accuracy, confidence, and quality of financial metric extractions."""
    
    def __init__(self):
        """Initialize evaluator."""
        self.results = []
        print("✅ Initialized ExtractionEvaluator")
    
    def compare_numeric(self, extracted: Optional[float], expected: Optional[float], 
                       tolerance: float = 0.05) -> Tuple[bool, float]:
        """
        Compare extracted vs expected numeric values with tolerance.
        
        Args:
            extracted: Extracted value
            expected: Expected/ground truth value
            tolerance: Tolerance as percentage (0.05 = 5%)
            
        Returns:
            (is_correct, confidence_score)
        """
        if extracted is None or expected is None:
            return (extracted == expected, 0.0 if extracted != expected else 0.9)
        
        if expected == 0:
            return (extracted == expected, 1.0 if extracted == expected else 0.0)
        
        error_pct = abs(extracted - expected) / abs(expected)
        is_correct = error_pct <= tolerance
        confidence = max(0, 1 - error_pct)
        
        return (is_correct, confidence)
    
    def compare_list(self, extracted: List[str], expected: List[str]) -> Tuple[float, float]:
        """
        Compare extracted vs expected list (e.g., risks).
        
        Args:
            extracted: Extracted list
            expected: Expected list
            
        Returns:
            (precision, recall)
        """
        if not extracted and not expected:
            return (1.0, 1.0)
        
        if not extracted or not expected:
            return (0.0, 0.0)
        
        # Simple string matching
        extracted_lower = [str(x).lower().strip() for x in extracted]
        expected_lower = [str(x).lower().strip() for x in expected]
        
        matches = sum(1 for e in extracted_lower if any(ex in e for ex in expected_lower))
        
        precision = matches / len(extracted_lower) if extracted_lower else 0
        recall = matches / len(expected_lower) if expected_lower else 0
        
        return (precision, recall)
    
    def evaluate_extraction(self, extracted: Dict, ground_truth: Dict) -> Dict:
        """
        Comprehensive evaluation of extraction accuracy.
        
        Args:
            extracted: Extracted metrics
            ground_truth: Ground truth values
            
        Returns:
            Evaluation report
        """
        report = {
            "metrics_evaluation": {},
            "overall_accuracy": 0.0,
            "field_scores": {},
            "issues": []
        }
        
        numeric_metrics = ["revenue", "net_income", "operating_expenses", 
                          "assets_under_management", "debt_to_equity", "equity_ratio"]
        
        correct_count = 0
        confidence_scores = {}
        
        # Evaluate numeric metrics
        for metric in numeric_metrics:
            extracted_val = extracted.get(metric)
            expected_val = ground_truth.get(metric)
            
            is_correct, confidence = self.compare_numeric(extracted_val, expected_val)
            
            report["metrics_evaluation"][metric] = {
                "extracted": extracted_val,
                "expected": expected_val,
                "correct": is_correct,
                "confidence": round(confidence, 3)
            }
            
            confidence_scores[metric] = confidence
            if is_correct:
                correct_count += 1
        
        # Evaluate lists (risks)
        if "risks" in ground_truth:
            extracted_risks = extracted.get("risks", [])
            expected_risks = ground_truth.get("risks", [])
            precision, recall = self.compare_list(extracted_risks, expected_risks)
            
            report["metrics_evaluation"]["risks"] = {
                "extracted": extracted_risks,
                "expected": expected_risks,
                "precision": round(precision, 3),
                "recall": round(recall, 3),
                "f1_score": round(2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0, 3)
            }
            
            confidence_scores["risks"] = (precision + recall) / 2
        
        # Calculate overall metrics
        report["field_scores"] = confidence_scores
        report["overall_accuracy"] = round(correct_count / len(numeric_metrics), 3)
        report["avg_confidence"] = round(sum(confidence_scores.values()) / len(confidence_scores), 3)
        
        # Check for hallucinations
        hallucination_check = self._check_hallucinations(extracted, ground_truth)
        if hallucination_check:
            report["issues"].append(hallucination_check)
        
        return report
    
    def _check_hallucinations(self, extracted: Dict, ground_truth: Dict) -> Optional[str]:
        """
        Check for obvious hallucinations (high confidence, low accuracy).
        
        Args:
            extracted: Extracted metrics
            ground_truth: Ground truth
            
        Returns:
            Hallucination warning or None
        """
        data_quality = extracted.get("data_quality", "").lower()
        
        # If model claims high quality but accuracy is low, that's suspicious
        if data_quality == "high":
            metric_accuracy = sum([
                1 for metric in ["revenue", "net_income", "operating_expenses"]
                if extracted.get(metric) == ground_truth.get(metric)
            ]) / 3
            
            if metric_accuracy < 0.5:
                return "⚠️  HIGH HALLUCINATION RISK: Model claims 'high' data_quality but accuracy is low"
        
        return None
    
    def test_edge_cases(self, rag: RAGPipeline, extractor: FinancialMetricsExtractor) -> Dict:
        """
        Test extraction on edge cases.
        
        Args:
            rag: RAG pipeline
            extractor: Extractor instance
            
        Returns:
            Edge case test results
        """
        results = {
            "empty_query": None,
            "partial_data": None,
            "vague_query": None
        }
        
        test_queries = [
            ("What is the revenue?", "partial_data"),
            ("Tell me about financial metrics", "vague_query"),
            ("Extract everything", "vague_query")
        ]
        
        for query, case_type in test_queries:
            try:
                context = rag.retrieve(query)
                extracted = extractor.extract_metrics(context, query)
                
                has_error = "error" in extracted
                has_data = len([v for v in extracted.values() if v is not None]) > 0
                
                results[case_type] = {
                    "query": query,
                    "has_error": has_error,
                    "has_data": has_data,
                    "result": extracted
                }
            except Exception as e:
                results[case_type] = {
                    "query": query,
                    "error": str(e)
                }
        
        return results
    
    def generate_report(self, evaluation: Dict) -> str:
        """
        Generate human-readable evaluation report.
        
        Args:
            evaluation: Evaluation results
            
        Returns:
            Formatted report string
        """
        report = "\n" + "="*70 + "\n"
        report += "EXTRACTION EVALUATION REPORT\n"
        report += "="*70 + "\n\n"
        
        # Overall metrics
        report += "📊 OVERALL METRICS:\n"
        report += f"  Overall Accuracy: {evaluation['overall_accuracy']*100:.1f}%\n"
        report += f"  Average Confidence: {evaluation['avg_confidence']*100:.1f}%\n\n"
        
        # Field-by-field accuracy
        report += "🔍 FIELD-BY-FIELD EVALUATION:\n"
        for field, evaluation_data in evaluation['metrics_evaluation'].items():
            if field == "risks":
                f1 = evaluation_data.get("f1_score", 0)
                report += f"  {field.upper()}: F1={f1:.2f}\n"
            else:
                correct = "✅" if evaluation_data['correct'] else "❌"
                conf = evaluation_data['confidence']
                report += f"  {field}: {correct} (Confidence: {conf:.2f})\n"
        
        # Issues
        if evaluation['issues']:
            report += "\n⚠️  ISSUES DETECTED:\n"
            for issue in evaluation['issues']:
                report += f"  • {issue}\n"
        
        report += "\n" + "="*70 + "\n"
        
        return report


# Ground truth for testing
GROUND_TRUTH = {
    "revenue": 18200000000,
    "net_income": 1200000000,
    "operating_expenses": 4800000000,
    "assets_under_management": 850000000000,
    "debt_to_equity": 0.42,
    "equity_ratio": 0.70,
    "risks": [
        "rising interest rates",
        "regulatory pressures",
        "geopolitical uncertainties",
        "competitive pressure"
    ]
}


# Example usage
if __name__ == "__main__":
    # Initialize components
    rag = RAGPipeline()
    rag.build_pipeline(["data/sample_financial_report.txt"])
    
    extractor = FinancialMetricsExtractor()
    evaluator = ExtractionEvaluator()
    
    # Run extraction
    print("\n📊 Running Extraction & Evaluation...\n")
    context = rag.retrieve("Extract all financial metrics")
    extracted = extractor.extract_metrics(context)
    
    # Evaluate
    evaluation = evaluator.evaluate_extraction(extracted, GROUND_TRUTH)
    
    # Generate report
    report = evaluator.generate_report(evaluation)
    print(report)
    
    # Save results
    results = {
        "extracted_metrics": extracted,
        "ground_truth": GROUND_TRUTH,
        "evaluation": evaluation
    }
    
    with open("results/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("✅ Results saved to results/evaluation_results.json\n")