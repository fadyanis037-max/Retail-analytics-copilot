"""
BM25-based document retrieval system for RAG.
Handles chunking, indexing, and search with citation tracking.
"""

import os
from typing import List, Dict, Tuple
from dataclasses import dataclass
import re
from rank_bm25 import BM25Okapi


@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    id: str  # Format: "{filename}::chunk{N}"
    content: str
    source: str  # Filename
    score: float = 0.0


class DocumentRetriever:
    """BM25-based retriever for local documents."""
    
    def __init__(self, docs_dir: str):
        """
        Initialize retriever with document directory.
        
        Args:
            docs_dir: Path to directory containing markdown files
        """
        self.docs_dir = docs_dir
        self.chunks: List[Chunk] = []
        self.bm25 = None
        self.tokenized_chunks = []
        
    def load_and_index(self):
        """Load all documents, chunk them, and build BM25 index."""
        # Load all markdown files
        for filename in os.listdir(self.docs_dir):
            if filename.endswith('.md'):
                filepath = os.path.join(self.docs_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Chunk by paragraphs (split on double newlines)
                paragraphs = re.split(r'\n\n+', content.strip())
                
                for idx, para in enumerate(paragraphs):
                    if para.strip():  # Skip empty paragraphs
                        chunk_id = f"{filename.replace('.md', '')}::chunk{idx}"
                        chunk = Chunk(
                            id=chunk_id,
                            content=para.strip(),
                            source=filename
                        )
                        self.chunks.append(chunk)
        
        # Tokenize chunks for BM25 (simple whitespace tokenization)
        self.tokenized_chunks = [
            chunk.content.lower().split() 
            for chunk in self.chunks
        ]
        
        # Build BM25 index
        if self.tokenized_chunks:
            self.bm25 = BM25Okapi(self.tokenized_chunks)
    
    def search(self, query: str, top_k: int = 3) -> List[Chunk]:
        """
        Search for relevant chunks using BM25.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            List of Chunk objects with scores
        """
        if not self.bm25 or not self.chunks:
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(
            range(len(scores)), 
            key=lambda i: scores[i], 
            reverse=True
        )[:top_k]
        
        # Build result chunks with scores
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            chunk.score = float(scores[idx])
            results.append(chunk)
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Chunk:
        """Get a specific chunk by its ID."""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None
