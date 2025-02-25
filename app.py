import streamlit as st
import pandas as pd
from pathlib import Path
from src.pipelines.nlu_pipeline import NLUPipeline
from src.pipelines.analytics_pipeline import AnalyticsPipeline
from src.utils.pipeline_selector import PipelineSelector
import uuid

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())

# Page config
st.set_page_config(
    page_title="Dog Breed Assistant",
    page_icon="🐕",
    layout="wide"
)

# Title
st.title("🐕 Intelligent Dog Breed Assistant")

# Load pipelines
@st.cache_resource
def load_pipelines():
    nlu = NLUPipeline()
    analytics = AnalyticsPipeline()
    selector = PipelineSelector()
    return nlu, analytics, selector

nlu_pipeline, analytics_pipeline, pipeline_selector = load_pipelines()

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask me anything about dog breeds!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Determine which pipeline to use
    pipeline = pipeline_selector.select_pipeline(prompt)
    
    # Process the query
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if pipeline == "nlu":
                response = nlu_pipeline.process(prompt)
            else:
                response = analytics_pipeline.process(prompt)
            
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This is an intelligent question-answering system that helps users find information about dog breeds.
    
    You can ask questions like:
    - Natural language queries about breeds
    - Data analysis questions about breeds
    
    Examples:
    - "Which breeds are good with children?"
    - "Show me the top 5 longest living breeds"
    """) 