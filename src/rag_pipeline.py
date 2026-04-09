#!/usr/bin/env python3
"""RAG Pipeline for Financial Document Processing."""

import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline for financial documents."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize RAG pipeline with embeddings model.
        
        Args:
            model_name: HuggingFace embedding model name
        """
        self.embeddings = HuggingFaceEmbeddings(model_name=model_name)
        self.vector_store = None
        self.retriever = None
        
        print(f"✅ Initialized embeddings with model: {model_name}")
    
    def load_documents(self, file_paths: List[str]):
        """
        Load documents from file paths.
        
        Args:
            file_paths: List of file paths to load
            
        Returns:
            List of loaded documents
        """
        documents = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                print(f"⚠️  File not found: {file_path}")
                continue
                
            try:
                loader = TextLoader(file_path)
                docs = loader.load()
                documents.extend(docs)
                print(f"✅ Loaded: {file_path} ({len(docs)} documents)")
            except Exception as e:
                print(f"❌ Error loading {file_path}: {e}")
        
        return documents
    
    def split_documents(self, documents, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Split documents into chunks for embedding.
        
        Args:
            documents: List of documents to split
            chunk_size: Size of each chunk
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of chunked documents
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = splitter.split_documents(documents)
        print(f"✅ Split into {len(chunks)} chunks")
        
        return chunks
    
    def create_vector_store(self, chunks):
        """
        Create FAISS vector store from chunks.
        
        Args:
            chunks: List of document chunks
        """
        try:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}  # Retrieve top-3 most relevant chunks
            )
            print(f"✅ Created FAISS vector store")
        except Exception as e:
            print(f"❌ Error creating vector store: {e}")
            raise
    
    def retrieve(self, query: str) -> List[str]:
        """
        Retrieve relevant document chunks for a query.
        
        Args:
            query: Search query
            
        Returns:
            List of relevant document chunks
        """
        if not self.retriever:
            raise ValueError("Vector store not initialized. Call create_vector_store() first.")
        
        results = self.retriever.invoke(query)
        
        retrieved_text = [doc.page_content for doc in results]
        
        print(f"\n📄 Retrieved {len(retrieved_text)} relevant chunks for: '{query}'")
        
        return retrieved_text
    
    def build_pipeline(self, file_paths: List[str], chunk_size: int = 500, chunk_overlap: int = 100):
        """
        End-to-end RAG pipeline: load → split → embed → store.
        
        Args:
            file_paths: List of document paths
            chunk_size: Size of chunks
            chunk_overlap: Overlap between chunks
        """
        print("\n🚀 Starting RAG Pipeline Build...\n")
        
        # Load documents
        documents = self.load_documents(file_paths)
        if not documents:
            raise ValueError("No documents loaded")
        
        # Split into chunks
        chunks = self.split_documents(documents, chunk_size, chunk_overlap)
        
        # Create vector store
        self.create_vector_store(chunks)
        
        print("\n✅ RAG Pipeline Ready!\n")


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = RAGPipeline()
    
    # Build pipeline with sample data
    sample_files = ["data/sample_financial_report.txt"]
    pipeline.build_pipeline(sample_files)
    
    # Test retrieval
    queries = [
        "What was the total revenue in Q3?",
        "What are the main risks facing the company?",
        "How did Asia division perform?"
    ]
    
    for query in queries:
        context = pipeline.retrieve(query)
        print(f"📌 Query: {query}")
        for i, chunk in enumerate(context, 1):
            print(f"   [{i}] {chunk[:200]}...\n")