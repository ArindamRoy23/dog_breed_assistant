import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama
from typing import List, Dict

class NLUPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.llm = ollama
        
        # Load and prepare the dataset
        try:
            self.df = pd.read_csv('src/data/dog_breeds.csv')
            # Create breed descriptions for embedding
            self.breed_descriptions = self._create_breed_descriptions()
            # Pre-compute embeddings
            self.breed_embeddings = self.embedding_model.encode(self.breed_descriptions)
        except FileNotFoundError:
            print("Warning: Dataset not found. Please ensure dog_breeds.csv is in the data directory.")
            self.df = None
            self.breed_descriptions = []
            self.breed_embeddings = None
    
    def _create_breed_descriptions(self) -> List[str]:
        """Create comprehensive descriptions for each breed from the dataset."""
        descriptions = []
        for _, row in self.df.iterrows():
            desc = f"The {row['breed']} is a {row['size']} sized dog. "
            desc += f"It has {row['coat_type']} fur and typically lives {row['life_expectancy']} years. "
            desc += f"This breed is known for being {row['temperament']}. "
            desc += f"They require {row['energy_level']} exercise and are {row['trainability']} to train. "
            desc += f"Their grooming needs are {row['grooming']}."
            descriptions.append(desc)
        return descriptions
    
    def _get_relevant_breeds(self, query: str, top_k: int = 3) -> List[Dict]:
        """Retrieve the most relevant breeds for the query using embedding similarity."""
        query_embedding = self.embedding_model.encode([query])
        similarities = cosine_similarity(query_embedding, self.breed_embeddings)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        relevant_breeds = []
        for idx in top_indices:
            breed_data = self.df.iloc[idx].to_dict()
            breed_data['description'] = self.breed_descriptions[idx]
            breed_data['similarity'] = similarities[idx]
            relevant_breeds.append(breed_data)
            
        return relevant_breeds
    
    def process(self, query: str) -> str:
        """Process a natural language query about dog breeds."""
        if self.df is None:
            return "Error: Dataset not loaded. Please ensure dog_breeds.csv is available."
        
        # Get relevant breeds
        relevant_breeds = self._get_relevant_breeds(query)
        
        # Prepare context for the LLM
        context = "Based on the following dog breed information:\n\n"
        for breed in relevant_breeds:
            context += f"- {breed['description']}\n"
        
        # Prepare the prompt
        prompt = f"{context}\n\nUser Question: {query}\n\nPlease provide a helpful and informative answer based on the above information. Focus on directly addressing the user's question while incorporating specific details about the relevant breeds."
        
        # Get response from Ollama
        try:
            response = self.llm.chat(model='mistral', messages=[
                {
                    'role': 'system',
                    'content': 'You are a helpful dog breed expert. Provide clear, concise, and accurate information about dog breeds.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            
            return response['message']['content']
        except Exception as e:
            return f"Error processing query: {str(e)}" 