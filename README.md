# Intelligent Dog Breed Assistant

An AI-powered question-answering system that helps users find information about dog breeds. The system combines natural language understanding and data analytics capabilities to provide comprehensive answers about dog breeds.

## Features

- Natural Language Understanding (NLU) Pipeline for contextual queries
- Data Analytics Pipeline for numerical and statistical analysis
- Interactive Streamlit web interface
- Containerized deployment using Docker
- Intelligent pipeline selection based on query type

## Prerequisites

- Python 3.10 or higher
- Docker (optional, for containerized deployment)
- Ollama (for local LLM support)

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd dog-breed-assistant
```

2. Install Ollama:
Visit [Ollama's website](https://ollama.ai/) and follow the installation instructions for your platform.

3. Pull the required Ollama model:
```bash
ollama pull mistral
```

4. Download the dataset:
- Download the dog breeds dataset from [Kaggle](https://www.kaggle.com/datasets/mexwell/dog-breeds-dataset)
- Place the downloaded CSV file in `src/data/dog_breeds.csv`

5. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

### Local Development
```bash
streamlit run app.py
```

### Using Docker
1. Build the Docker image:
```bash
docker build -t dog-breed-assistant .
```

2. Run the container:
```bash
docker run -p 8501:8501 dog-breed-assistant
```

The application will be available at `http://localhost:8501`

## Testing

### Running Tests
To run the test suite:
```bash
pytest
```

To run tests with coverage report:
```bash
pytest --cov=src tests/
```

### Test Structure
- `tests/conftest.py`: Common test fixtures and configurations
- `tests/test_pipeline_selector.py`: Tests for pipeline selection logic
- `tests/test_nlu_pipeline.py`: Tests for natural language understanding pipeline
- `tests/test_analytics_pipeline.py`: Tests for data analytics pipeline

### Manual Testing
1. Start the application:
```bash
streamlit run app.py
```

2. Try example queries:
   - Natural Language: "What breeds are good with children?"
   - Analytics: "Show me the top 5 longest living breeds"

3. Verify responses for:
   - Accuracy and relevance
   - Pipeline selection correctness
   - Error handling
   - Response formatting

## Usage Examples

### Natural Language Queries
- "What breeds are good with children?"
- "I'm looking for a dog that's good in apartments"
- "Tell me about breeds that don't shed much"

### Data Analysis Queries
- "What are the top 5 heaviest breeds?"
- "Show me breeds sorted by lifespan"
- "Which breeds have the highest energy levels?"

## Project Structure

```
dog-breed-assistant/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── README.md            # Project documentation
├── tests/               # Test suite
│   ├── conftest.py      # Test configurations
│   ├── test_pipeline_selector.py
│   ├── test_nlu_pipeline.py
│   └── test_analytics_pipeline.py
└── src/
    ├── data/            # Dataset directory
    ├── pipelines/       # Pipeline implementations
    │   ├── nlu_pipeline.py
    │   └── analytics_pipeline.py
    └── utils/           # Utility functions
        └── pipeline_selector.py
```

## Technical Details

### NLU Pipeline
- Uses BERT embeddings for semantic search
- Implements RAG (Retrieval-Augmented Generation) with Ollama
- Provides contextual answers based on relevant breed information

### Analytics Pipeline
- Converts natural language to pandas operations
- Executes data analysis queries
- Formats results in natural language

### Pipeline Selection
- Uses BERT embeddings to classify query type
- Automatically routes queries to appropriate pipeline
- Maintains example queries for accurate classification

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 