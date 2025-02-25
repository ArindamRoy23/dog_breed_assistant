from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class PipelineSelector:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define example queries for each pipeline
        self.nlu_examples = [
            "What breeds are good with children?",
            "I'm looking for a dog that's good in apartments",
            "Which breeds are easy to train?",
            "Tell me about breeds that don't shed much",
            "What's the best breed for first-time owners?"
        ]
        
        self.analytics_examples = [
            "What are the top 5 heaviest breeds?",
            "Show me breeds sorted by lifespan",
            "Which breeds have the highest energy levels?",
            "List breeds by popularity",
            "Compare the sizes of different breeds"
        ]
        
        # Pre-compute embeddings
        self.nlu_embeddings = self.model.encode(self.nlu_examples)
        self.analytics_embeddings = self.model.encode(self.analytics_examples)
        
    def select_pipeline(self, query: str) -> str:
        # Get query embedding
        query_embedding = self.model.encode([query])
        
        # Calculate average similarity to each pipeline's examples
        nlu_similarity = cosine_similarity(query_embedding, self.nlu_embeddings).mean()
        analytics_similarity = cosine_similarity(query_embedding, self.analytics_embeddings).mean()
        
        # Return the pipeline with higher similarity
        return "nlu" if nlu_similarity > analytics_similarity else "analytics" 