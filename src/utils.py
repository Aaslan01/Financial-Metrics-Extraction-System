#!/usr/bin/env python3
"""Utility functions for logging, configuration, and error handling."""

import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


def setup_logging(log_file: str = "results/extraction.log") -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_file: Path to log file
        
    Returns:
        Configured logger instance
    """
    # Create results directory if needed
    Path("results").mkdir(exist_ok=True)
    
    logger = logging.getLogger("FinancialExtractor")
    logger.setLevel(logging.DEBUG)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def save_results(extraction_result: Dict, evaluation_result: Dict, 
                output_file: str = "results/extraction_results.json") -> None:
    """
    Save extraction and evaluation results to JSON.
    
    Args:
        extraction_result: Extracted metrics
        evaluation_result: Evaluation metrics
        output_file: Output file path
    """
    Path("results").mkdir(exist_ok=True)
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "extraction": extraction_result,
        "evaluation": evaluation_result,
        "metadata": {
            "model": "llama-3.1-8b-instant",
            "embedding_model": "all-MiniLM-L6-v2",
            "vector_db": "FAISS"
        }
    }
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to {output_file}")


def validate_json_structure(data: Dict) -> bool:
    """
    Validate that extracted data has required structure.
    
    Args:
        data: Data to validate
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ["revenue", "net_income", "debt_to_equity", "risks", "data_quality"]
    
    for field in required_fields:
        if field not in data:
            return False
    
    return True


def handle_extraction_error(error: Exception, logger: logging.Logger) -> Dict:
    """
    Handle extraction errors gracefully.
    
    Args:
        error: Exception that occurred
        logger: Logger instance
        
    Returns:
        Error response dictionary
    """
    logger.error(f"Extraction error: {str(error)}", exc_info=True)
    
    return {
        "error": str(error),
        "timestamp": datetime.now().isoformat(),
        "type": type(error).__name__
    }


class Config:
    """Configuration management."""
    
    # RAG Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 100
    RETRIEVAL_TOP_K = 3
    
    # Extraction Configuration
    GROQ_MODEL = "llama-3.1-8b-instant"
    MAX_TOKENS = 1024
    TEMPERATURE = 0.7
    
    # Evaluation Configuration
    NUMERIC_TOLERANCE = 0.05  # 5% tolerance
    MIN_CONFIDENCE_THRESHOLD = 0.7
    
    # Logging Configuration
    LOG_FILE = "results/extraction.log"
    RESULTS_FILE = "results/extraction_results.json"
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "embedding_model": cls.EMBEDDING_MODEL,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "retrieval_top_k": cls.RETRIEVAL_TOP_K,
            "groq_model": cls.GROQ_MODEL,
            "max_tokens": cls.MAX_TOKENS,
            "temperature": cls.TEMPERATURE,
            "numeric_tolerance": cls.NUMERIC_TOLERANCE,
            "min_confidence_threshold": cls.MIN_CONFIDENCE_THRESHOLD
        }


if __name__ == "__main__":
    # Test configuration
    print("Configuration:")
    print(json.dumps(Config.to_dict(), indent=2))