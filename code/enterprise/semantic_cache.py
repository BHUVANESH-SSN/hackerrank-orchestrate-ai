"""
Semantic Caching Layer Blueprint.

Intercepts identical or highly similar user queries before they hit the expensive LLM.
Uses local SentenceTransformers to compute cosine similarity against previously answered questions.
Reduces API costs by 80%+ on recurring Support queries.
"""
import numpy as np
from typing import Optional, Dict
from sentence_transformers import SentenceTransformer

class SemanticCache:
    def __init__(self, similarity_threshold: float = 0.95):
        # We reuse the lightweight all-MiniLM-L6-v2 model you already downloaded for ChromaDB!
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.threshold = similarity_threshold
        
        # In production, this data lives in Redis Vector Store
        self.cache_keys = []      # List of embedded vectors
        self.cache_values = {}    # Maps index to the generated AgentState/Response
        
    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def get_cached_response(self, user_query: str) -> Optional[Dict]:
        """Check if an extremely similar question was already asked."""
        if not self.cache_keys:
            return None
            
        query_vector = self.encoder.encode(user_query)
        
        best_score = 0.0
        best_idx = -1
        
        for i, cached_vec in enumerate(self.cache_keys):
            score = self._cosine_similarity(query_vector, cached_vec)
            if score > best_score:
                best_score = score
                best_idx = i
                
        # If the user's question is 95%+ similar to an old one, return the cached result!
        if best_score >= self.threshold:
            print(f"💰 CACHE HIT! Bypassing LangGraph processing. (Similarity: {best_score:.2f})")
            return self.cache_values[best_idx]
            
        return None

    def cache_new_response(self, user_query: str, pipeline_output: Dict):
        """Save a new LLM generation to the cache."""
        query_vector = self.encoder.encode(user_query)
        idx = len(self.cache_keys)
        
        self.cache_keys.append(query_vector)
        self.cache_values[idx] = pipeline_output
        print("💾 New query routed through LangGraph and cached for future users.")

# Example Usage
if __name__ == "__main__":
    cache = SemanticCache(similarity_threshold=0.90)
    
    # 1. User A asks a brand new question
    query_a = "How do I reset my HackerRank password?"
    result_a = cache.get_cached_response(query_a)
    if not result_a:
        # Normally this runs the full expensive Graph pipeline
        generated_answer = {"response": "Go to settings > reset password.", "tokens_used": 450}
        cache.cache_new_response(query_a, generated_answer)
        
    print("-" * 50)
    
    # 2. User B asks the exact same semantic question with different wording
    query_b = "I forgot my password for HackerRank, how can I change it?"
    result_b = cache.get_cached_response(query_b)
    if result_b:
        print(f"Immediate Output: {result_b['response']}")
        print("Token API Cost: 0")
