from unittest.mock import Mock, patch

import pytest
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from src.recommender.llm_recommender import (
    build_self_query_chain,
    create_recommender_chain,
    initialize_embeddings_model,
    load_chroma_index,
    load_cross_encoder_model,
    route,
)


@pytest.fixture
def mock_embeddings():
    return Mock(spec=HuggingFaceEmbeddings)


@pytest.fixture
def mock_vectorstore():
    return Mock(spec=Chroma)


@pytest.fixture
def sample_docs():
    return [
        Document(
            page_content="Summer dress",
            metadata={"price": 1500, "size": "M", "category": "dress"},
        ),
        Document(
            page_content="Winter coat",
            metadata={"price": 3000, "size": "L", "category": "coat"},
        ),
    ]


def test_initialize_embeddings_model():
    with patch("langchain_huggingface.HuggingFaceEmbeddings") as mock_embeddings:
        embeddings = initialize_embeddings_model()
        assert isinstance(embeddings, HuggingFaceEmbeddings)


def test_load_chroma_index(mock_embeddings):
    with patch("langchain_chroma.Chroma") as mock_chroma:
        vectorstore = load_chroma_index(mock_embeddings)
        assert isinstance(vectorstore, Chroma)


def test_build_self_query_chain(mock_vectorstore):
    chain = build_self_query_chain(mock_vectorstore)
    assert callable(chain)


def test_route_with_empty_docs():
    result = route([])
    assert callable(result)


def test_route_with_docs(sample_docs):
    result = route(sample_docs)
    assert callable(result)


def test_create_recommender_chain(mock_vectorstore):
    chain = create_recommender_chain(mock_vectorstore)
    assert callable(chain)


@pytest.mark.integration
def test_recommender_chain_integration():
    query = "woman dress for summer less than 2000"
    embeddings = initialize_embeddings_model()
    vectorstore = load_chroma_index(embeddings)
    chain = create_recommender_chain(vectorstore)

    result = chain.invoke({"query": query})
    assert isinstance(result, str)
    assert len(result) > 0
