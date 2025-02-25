import pytest
from unittest.mock import patch, MagicMock
from src.pipelines.analytics_pipeline import AnalyticsPipeline

def test_analytics_pipeline_initialization(sample_dog_data):
    """Test analytics pipeline initialization with sample data."""
    with patch('pandas.read_csv', return_value=sample_dog_data):
        pipeline = AnalyticsPipeline()
        assert pipeline.df is not None
        assert len(pipeline.df) == len(sample_dog_data)

@pytest.mark.parametrize("query,expected_code", [
    ("What are the top 5 heaviest breeds?", "df.nlargest(5, 'weight')"),
    ("Show me breeds sorted by lifespan", "df.sort_values('life_expectancy')"),
    ("Which breeds have the highest energy levels?", "df.nlargest(5, 'energy_level')")
])
@patch('ollama.chat')
def test_generate_pandas_query(mock_chat, query, expected_code, sample_dog_data):
    """Test generation of pandas queries from natural language."""
    mock_chat.return_value = {'message': {'content': expected_code}}
    
    with patch('pandas.read_csv', return_value=sample_dog_data):
        pipeline = AnalyticsPipeline()
        result = pipeline._generate_pandas_query(query)
        
        assert isinstance(result, str)
        assert len(result) > 0
        assert mock_chat.called

@patch('ollama.chat')
def test_format_results(mock_chat, sample_dog_data, mock_ollama_response):
    """Test formatting of analysis results."""
    mock_chat.return_value = mock_ollama_response
    
    with patch('pandas.read_csv', return_value=sample_dog_data):
        pipeline = AnalyticsPipeline()
        results = sample_dog_data.head()
        response = pipeline._format_results(results, "Show me the top breeds")
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert mock_chat.called

def test_process_query_error_handling():
    """Test error handling in query processing."""
    with patch('pandas.read_csv', side_effect=FileNotFoundError):
        pipeline = AnalyticsPipeline()
        response = pipeline.process("Show me the heaviest breeds")
        assert "Error" in response
        assert "dataset" in response.lower()

@patch('ollama.chat')
def test_full_query_processing(mock_chat, sample_dog_data, mock_ollama_response):
    """Test end-to-end query processing."""
    # Mock both LLM calls (query generation and result formatting)
    mock_chat.side_effect = [
        {'message': {'content': "df.nlargest(5, 'weight')"}},
        mock_ollama_response
    ]
    
    with patch('pandas.read_csv', return_value=sample_dog_data):
        pipeline = AnalyticsPipeline()
        response = pipeline.process("What are the heaviest breeds?")
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert mock_chat.call_count == 2  # Called for query generation and formatting 