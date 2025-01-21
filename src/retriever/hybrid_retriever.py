"""
This module implements a chatbot for assisting parents with their questions using
a retrieval-based approach.
"""

import os
import pickle
import sys

from langchain.globals import set_debug
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import settings


def load_faiss_index():
    """
    Load the FAISS index.
    """
    try:
        embeddings_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDINGS_MODEL_NAME
        )
        vector_store = FAISS.load_local(
            settings.FAISS_INDEX_PATH,
            embeddings_model,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        logger.exception("Failed to load FAISS index.")
        raise e
    embedding_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.FAISS_TOP_K},
    )
    return embedding_retriever


def load_bm25_index():
    """
    Load the BM25 index.
    """
    try:
        with open(settings.BM25_INDEX_PATH, "rb") as file:
            bm25_retriever = pickle.load(file)
    except Exception as e:
        logger.exception("Failed to load BM25 index.")
        raise e
    return bm25_retriever


def create_ensemble_retriever(retrievers: list):
    """
    Create an ensemble retriever.
    """
    ensemble_retriever = EnsembleRetriever(
        retrievers=retrievers,
        weights=settings.RETRIEVER_WEIGHTS,
        top_k=settings.RETTRIEVER_TOP_K,
    )
    return ensemble_retriever


def create_cross_encoder_reranker(ensemble_retriever):
    """
    Create a cross encoder reranker.
    """
    model_name = settings.CROSS_ENCODER_MODEL_NAME
    model = HuggingFaceCrossEncoder(model_name=model_name)
    compressor = CrossEncoderReranker(model=model, top_n=3)
    cross_encoder_reranker = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )
    return cross_encoder_reranker


def save_cross_encoder_reranker(cross_encoder_reranker):
    """
    Save the cross encoder reranker.
    """
    pickle.dump(
        cross_encoder_reranker,
        open(settings.CROSS_ENCODER_RERANKER_PATH, "wb"),
    )


def retriever_flow():
    """
    Run the retriever flow.
    """
    try:
        faiss_retriever = load_faiss_index()
        bm25_retriever = load_bm25_index()
        ensemble_retriever = create_ensemble_retriever(
            [faiss_retriever, bm25_retriever]
        )
        cross_encoder_reranker = create_cross_encoder_reranker(ensemble_retriever)
        save_cross_encoder_reranker(cross_encoder_reranker)
    except Exception as e:
        logger.exception("Failed to run retriever flow.")
        raise e
