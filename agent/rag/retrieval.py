import os
import glob
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class LocalRetriever:
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = docs_dir
        self.chunks: List[Dict] = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self._load_and_index()

    def _load_and_index(self):
        """Loads markdown files, chunks them, and builds TF-IDF index."""
        file_paths = glob.glob(os.path.join(self.docs_dir, "*.md"))
        
        chunk_id_counter = 0
        for file_path in file_paths:
            filename = os.path.basename(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Simple paragraph splitting
            # Split by double newline to get paragraphs/sections
            raw_chunks = content.split("\n\n")
            
            for i, raw_chunk in enumerate(raw_chunks):
                clean_chunk = raw_chunk.strip()
                if not clean_chunk:
                    continue
                
                # Further split if lines start with # (headers) to keep context?
                # For simplicity, we just treat paragraphs/sections as chunks.
                # We might want to prepend the filename or header to the chunk for context.
                
                self.chunks.append({
                    "id": f"{filename}::chunk{i}",
                    "content": clean_chunk,
                    "source": filename,
                    "full_id": f"{filename}::chunk{i}"
                })
                chunk_id_counter += 1
        
        if self.chunks:
            corpus = [chunk["content"] for chunk in self.chunks]
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieves top-k relevant chunks for the query."""
        if not self.chunks or self.vectorizer is None:
            return []
        
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        
        # Get top k indices
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        results = []
        for idx in top_k_indices:
            if similarities[idx] > 0: # Only return positive matches
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(similarities[idx])
                results.append(chunk)
                
        return results
