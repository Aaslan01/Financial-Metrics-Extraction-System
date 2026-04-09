#!/usr/bin/env python3
"""Quick test to verify all imports work."""

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from groq import Groq
    from dotenv import load_dotenv
    import os
    
    print("✅ All imports successful!")
    
    # Test Groq connection
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    if api_key:
        client = Groq(api_key=api_key)
        print("✅ Groq API key loaded!")
    else:
        print("❌ Groq API key not found in .env")
        
except Exception as e:
    print(f"❌ Import failed: {e}")