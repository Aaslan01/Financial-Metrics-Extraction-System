#!/usr/bin/env python3
"""Financial Metrics Extraction using Groq API."""

import json
import os
from typing import Dict, List, Optional
from groq import Groq
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline


# Load environment variables
load_dotenv()


class FinancialMetricsExtractor:
    """Extract financial metrics from documents using Groq API with RAG context."""
    
    def __init__(self, model: str =  "llama-3.1-8b-instant"):
        """
        Initialize extractor with Groq client.
        
        Args:
            model: Groq model to use (free tier options)
        """
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in .env file")
        
        self.client = Groq(api_key=api_key)
        self.model = model
        print(f"✅ Initialized Groq extractor with model: {model}")
    
    def extract_metrics(self, context: List[str], query: str = "Extract all financial metrics") -> Dict:
        """
        Extract financial metrics from retrieved context using Groq.
        
        Args:
            context: List of relevant document chunks
            query: Extraction query
            
        Returns:
            Dictionary with extracted metrics
        """
        # Join context chunks
        context_text = "\n\n".join(context)
        
        # Prompt for structured extraction
        prompt = f"""You are a financial analyst. Extract key financial metrics from the following document context.

DOCUMENT CONTEXT:
{context_text}

EXTRACTION TASK:
Extract and return ONLY valid JSON with the following structure:
{{
    "revenue": <number or null>,
    "net_income": <number or null>,
    "operating_expenses": <number or null>,
    "assets_under_management": <number or null>,
    "debt_to_equity": <number or null>,
    "equity_ratio": <number or null>,
    "risks": [<list of identified risks as strings>],
    "segment_performance": {{<segment_name: performance_description>}},
    "key_insights": "<brief summary of financial health>",
    "data_quality": "<high/medium/low - confidence in extracted data>"
}}

RULES:
1. Extract numbers only (no currency symbols)
2. Convert billions to actual numbers (18.2 billion = 18200000000)
3. If data not found, use null
4. Return ONLY valid JSON, no other text
5. If you cannot parse as JSON, return an error field

Extract now:"""
        
        try:
            message = self.client.chat.completions.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = message.choices[0].message.content.strip()
            
            # Parse JSON response
            try:
                extracted = json.loads(response_text)
            except json.JSONDecodeError:
                # Try to extract JSON from response if wrapped in markdown
                if "```json" in response_text:
                    json_str = response_text.split("```json")[1].split("```")[0].strip()
                    extracted = json.loads(json_str)
                elif "```" in response_text:
                    json_str = response_text.split("```")[1].split("```")[0].strip()
                    extracted = json.loads(json_str)
                else:
                    extracted = {"error": "Failed to parse response", "raw": response_text}
            
            return extracted
            
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            return {"error": str(e)}
    
    def identify_risks(self, context: List[str]) -> List[str]:
        """
        Identify financial risks from context.
        
        Args:
            context: List of relevant document chunks
            
        Returns:
            List of identified risks
        """
        context_text = "\n\n".join(context)
        
        prompt = f"""From the following financial document, identify all key risks mentioned.

DOCUMENT:
{context_text}

Return ONLY a JSON array of risk strings, like:
["risk1", "risk2", "risk3"]

Extract risks now:"""
        
        try:
            message = self.client.chat.completions.create(
                model=self.model,
                max_tokens=512,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = message.choices[0].message.content.strip()
            
            # Parse JSON array
            try:
                risks = json.loads(response_text)
            except json.JSONDecodeError:
                if "[" in response_text:
                    json_str = response_text[response_text.index("["):response_text.rindex("]")+1]
                    risks = json.loads(json_str)
                else:
                    risks = []
            
            return risks
            
        except Exception as e:
            print(f"❌ Risk identification error: {e}")
            return []
    
    def evaluate_extraction(self, extracted: Dict) -> Dict:
        """
        Evaluate quality of extraction.
        
        Args:
            extracted: Extracted metrics dictionary
            
        Returns:
            Evaluation results
        """
        evaluation = {
            "fields_extracted": 0,
            "null_fields": 0,
            "has_risks": False,
            "has_insights": False,
            "data_quality": "unknown"
        }
        
        if "error" in extracted:
            evaluation["error"] = extracted["error"]
            return evaluation
        
        # Count extracted fields
        metric_fields = ["revenue", "net_income", "operating_expenses", 
                        "assets_under_management", "debt_to_equity", "equity_ratio"]
        
        for field in metric_fields:
            if field in extracted:
                if extracted[field] is not None:
                    evaluation["fields_extracted"] += 1
                else:
                    evaluation["null_fields"] += 1
        
        evaluation["has_risks"] = bool(extracted.get("risks"))
        evaluation["has_insights"] = bool(extracted.get("key_insights"))
        evaluation["data_quality"] = extracted.get("data_quality", "unknown")
        
        return evaluation


# Example usage
if __name__ == "__main__":
    # Initialize RAG and Extractor
    rag = RAGPipeline()
    rag.build_pipeline(["data/sample_financial_report.txt"])
    
    extractor = FinancialMetricsExtractor()
    
    # Test extraction
    print("\n" + "="*60)
    print("FINANCIAL METRICS EXTRACTION")
    print("="*60 + "\n")
    
    query = "Extract all financial metrics from this document"
    context = rag.retrieve(query)
    
    print("📊 Extracting metrics...\n")
    result = extractor.extract_metrics(context, query)
    
    print("📋 EXTRACTED METRICS:")
    print(json.dumps(result, indent=2))
    
    print("\n📊 EXTRACTION QUALITY EVALUATION:")
    evaluation = extractor.evaluate_extraction(result)
    print(json.dumps(evaluation, indent=2))