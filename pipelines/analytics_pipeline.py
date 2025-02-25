import pandas as pd
import ollama
from typing import Tuple, Any

class AnalyticsPipeline:
    def __init__(self):
        self.llm = ollama
        try:
            self.df = pd.read_csv('src/data/dog_breeds.csv')
        except FileNotFoundError:
            print("Warning: Dataset not found. Please ensure dog_breeds.csv is in the data directory.")
            self.df = None
    
    def _generate_pandas_query(self, user_query: str) -> str:
        """Convert natural language query to pandas operations using LLM."""
        prompt = f"""Given a pandas DataFrame 'df' with columns representing dog breed information, 
        generate Python code using pandas to answer this question: "{user_query}"
        
        The DataFrame has these columns:
        {', '.join(self.df.columns if self.df is not None else [])}
        
        Return ONLY the pandas code, nothing else. The code should be a single line or expression 
        that can be evaluated to get the answer. Do not include print statements or variable assignments."""
        
        try:
            response = self.llm.chat(model='mistral', messages=[
                {
                    'role': 'system',
                    'content': 'You are a data analysis expert. Generate only pandas code, no explanations.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            
            return response['message']['content'].strip()
        except Exception as e:
            return str(e)
    
    def _format_results(self, results: Any, user_query: str) -> str:
        """Format the pandas results into a natural language response."""
        # Convert results to string format for LLM
        results_str = str(results)
        
        prompt = f"""Given these analysis results:
        {results_str}
        
        And the original question:
        {user_query}
        
        Please provide a clear, natural language response that explains the findings.
        Focus on the key insights and present the information in a way that's easy to understand."""
        
        try:
            response = self.llm.chat(model='mistral', messages=[
                {
                    'role': 'system',
                    'content': 'You are a helpful dog breed expert. Explain data analysis results in clear, natural language.'
                },
                {
                    'role': 'user',
                    'content': prompt
                }
            ])
            
            return response['message']['content']
        except Exception as e:
            return f"Error formatting results: {str(e)}"
    
    def process(self, query: str) -> str:
        """Process a data analytics query about dog breeds."""
        if self.df is None:
            return "Error: Dataset not loaded. Please ensure dog_breeds.csv is available."
        
        try:
            # Generate pandas query
            pandas_query = self._generate_pandas_query(query)
            
            # Execute the query
            # Using eval with locals() to execute the pandas query in the current context
            local_vars = {'df': self.df, 'pd': pd}
            results = eval(pandas_query, {"__builtins__": {}}, local_vars)
            
            # Format results
            response = self._format_results(results, query)
            
            return response
        except Exception as e:
            return f"Error processing analytics query: {str(e)}" 