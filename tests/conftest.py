import pytest
import pandas as pd
import os
from pathlib import Path

@pytest.fixture
def sample_dog_data():
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        'breed': ['Labrador Retriever', 'German Shepherd', 'Golden Retriever'],
        'size': ['Large', 'Large', 'Large'],
        'coat_type': ['Double', 'Double', 'Double'],
        'life_expectancy': ['10-12 years', '9-13 years', '10-12 years'],
        'temperament': ['Friendly, Active', 'Loyal, Confident', 'Friendly, Intelligent'],
        'energy_level': ['High', 'High', 'High'],
        'trainability': ['Easy', 'Easy', 'Easy'],
        'grooming': ['Moderate', 'High', 'High']
    })

@pytest.fixture
def mock_ollama_response():
    """Mock Ollama response for testing."""
    return {
        'message': {
            'content': 'This is a mock response from the LLM.'
        }
    }

@pytest.fixture
def test_data_path(tmp_path):
    """Create a temporary data directory with test CSV."""
    data_dir = tmp_path / "src" / "data"
    data_dir.mkdir(parents=True)
    return data_dir 