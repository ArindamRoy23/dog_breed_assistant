import pytest
from src.utils.pipeline_selector import PipelineSelector

def test_pipeline_selector_initialization():
    """Test that the PipelineSelector initializes correctly."""
    selector = PipelineSelector()
    assert hasattr(selector, 'model')
    assert hasattr(selector, 'nlu_examples')
    assert hasattr(selector, 'analytics_examples')
    assert hasattr(selector, 'nlu_embeddings')
    assert hasattr(selector, 'analytics_embeddings')

@pytest.mark.parametrize("query,expected_pipeline", [
    ("What breeds are good with children?", "nlu"),
    ("I need a dog that doesn't shed much", "nlu"),
    ("Tell me about friendly breeds", "nlu"),
    ("What are the top 5 heaviest breeds?", "analytics"),
    ("Show me breeds sorted by lifespan", "analytics"),
    ("Compare the sizes of different breeds", "analytics"),
])
def test_pipeline_selection(query, expected_pipeline):
    """Test that queries are routed to the correct pipeline."""
    selector = PipelineSelector()
    result = selector.select_pipeline(query)
    assert result == expected_pipeline 