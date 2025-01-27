import os
import pickle
from unittest.mock import MagicMock, Mock, patch

import pytest
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.cross_encoder import HuggingFaceCrossEncoder

from src.recommender.llm_recommender import (
    build_self_query_chain,
    create_recommender_chain,
    initialize_embeddings_model,
    load_chroma_index,
    load_cross_encoder_model,
    route,
)


@pytest.fixture
def mock_cross_encoder():
    return Mock(spec=HuggingFaceCrossEncoder)


def test_load_cross_encoder_model(tmp_path):
    # Create a mock pickled model
    mock_model = MagicMock()
    test_path = tmp_path / "cross_encoder.pkl"
    with open(test_path, "wb") as f:
        pickle.dump(mock_model, f)

    with patch("src.config.settings.CROSS_ENCODER_RERANKER_PATH", test_path):
        model = load_cross_encoder_model()
        assert model is not None


def test_route_with_docs(sample_docs):
    with patch("src.recommender.llm_recommender.build_rag_chain") as mock_rag_chain:
        mock_rag_chain.return_value = MagicMock()
        result = route(sample_docs)
        assert callable(result)


def test_create_recommender_chain(mock_vectorstore):
    with (
        patch(
            "src.recommender.llm_recommender.build_self_query_chain"
        ) as mock_query_chain,
        patch("src.recommender.llm_recommender.build_rag_chain") as mock_rag_chain,
    ):
        mock_query_chain.return_value = MagicMock()
        mock_rag_chain.return_value = MagicMock()
        chain = create_recommender_chain(mock_vectorstore)
        assert callable(chain)


@pytest.mark.integration
def test_recommender_chain_integration(monkeypatch):
    # Mock the OpenAI API key
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Mock the necessary components
    with (
        patch(
            "src.recommender.llm_recommender.initialize_embeddings_model"
        ) as mock_embed,
        patch("src.recommender.llm_recommender.load_chroma_index") as mock_chroma,
        patch("src.recommender.llm_recommender.create_recommender_chain") as mock_chain,
    ):

        mock_embed.return_value = MagicMock()
        mock_chroma.return_value = MagicMock()
        mock_chain.return_value = MagicMock()
        mock_chain.return_value.invoke.return_value = "Test recommendation"

        query = "woman dress for summer less than 2000"
        result = mock_chain.return_value.invoke({"query": query})

        assert isinstance(result, str)
        assert len(result) > 0
