import pytest
from unittest.mock import patch, MagicMock
from src.pipelines.nlu_pipeline import NLUPipeline

def test_nlu_pipeline_initialization(sample_dog_data, test_data_path):
    """Test NLU pipeline initialization with sample data."""
    # Save sample data to test directory
    sample_dog_data.to_csv(test_data_path / "dog_breeds.csv", index=False)
    
    with patch('pandas.read_csv', return_value=sample_dog_data):
        pipeline = NLUPipeline()
        assert pipeline.df is not None
        assert len(pipeline.breed_descriptions) == len(sample_dog_data)
        assert pipeline.breed_embeddings is not None

def test_create_breed_descriptions(sample_dog_data):
    """Test breed description creation."""
    with patch('pandas.read_csv', return_value=sample_dog_data):
        pipeline = NLUPipeline()
        descriptions = pipeline._create_breed_descriptions()
        
        assert len(descriptions) == len(sample_dog_data)
        for desc in descriptions:
            assert isinstance(desc, str)
            assert "is a" in desc
            assert "sized dog" in desc

@pytest.mark.parametrize("query", [
    "What breeds are good with children?",
    "I need a low maintenance dog",
    "Tell me about active breeds"
])
def test_get_relevant_breeds(query, sample_dog_data):
    """Test retrieval of relevant breeds."""
    with patch('pandas.read_csv', return_value=sample_dog_data):
        pipeline = NLUPipeline()
        relevant_breeds = pipeline._get_relevant_breeds(query, top_k=2)
        
        assert len(relevant_breeds) <= 2
        for breed in relevant_breeds:
            assert 'breed' in breed
            assert 'description' in breed
            assert 'similarity' in breed
            assert isinstance(breed['similarity'], float)
            assert 0 <= breed['similarity'] <= 1

@patch('ollama.chat')
def test_process_query(mock_chat, sample_dog_data, mock_ollama_response):
    """Test processing of user queries."""
    mock_chat.return_value = mock_ollama_response
    
    with patch('pandas.read_csv', return_value=sample_dog_data):
        pipeline = NLUPipeline()
        response = pipeline.process("What breeds are good with children?")
        
        assert isinstance(response, str)
        assert mock_chat.called
        assert len(response) > 0

def test_pipeline_error_handling():
    """Test error handling when dataset is not found."""
    with patch('pandas.read_csv', side_effect=FileNotFoundError):
        pipeline = NLUPipeline()
        response = pipeline.process("What breeds are good with children?")
        assert "Error" in response
        assert "dataset" in response.lower() 