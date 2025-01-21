"""
This module implements a hybrid retriever using FAISS (vector search) and BM25 (lexical search).
It also applies cross-encoder reranking to improve retrieval quality.
"""

import os
import pickle
import sys
from typing import List

from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

# Append project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import settings


def load_faiss_index() -> FAISS:
    """
    Load the FAISS index.

    Returns:
        FAISS retriever object.
    """
    try:
        logger.info("Loading FAISS index...")
        embeddings_model = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDINGS_MODEL_NAME
        )
        vector_store = FAISS.load_local(
            settings.FAISS_INDEX_PATH,
            embeddings_model,
            allow_dangerous_deserialization=False,  # Security Improvement
        )
    except Exception as e:
        logger.exception("Failed to load FAISS index.")
        raise e

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.FAISS_TOP_K},
    )


def load_bm25_index() -> object:
    """
    Load the BM25 index.

    Returns:
        BM25 retriever object.
    """
    try:
        logger.info("Loading BM25 index...")
        with open(settings.BM25_INDEX_PATH, "rb") as file:
            bm25_retriever = pickle.load(file)
        return bm25_retriever
    except FileNotFoundError:
        logger.warning("BM25 index not found. Proceeding with FAISS-only retrieval.")
        return None
    except Exception as e:
        logger.exception("Failed to load BM25 index.")
        raise e


def create_ensemble_retriever(retrievers: List[object]) -> EnsembleRetriever:
    """
    Create an ensemble retriever from multiple retrieval sources.

    Args:
        retrievers: A list of retriever objects.

    Returns:
        EnsembleRetriever instance.
    """
    logger.info("Creating ensemble retriever...")
    return EnsembleRetriever(
        retrievers=retrievers,
        weights=settings.RETRIEVER_WEIGHTS,
        top_k=settings.RETRIEVER_TOP_K,  # Fixed Typo
    )


def create_cross_encoder_reranker(
    ensemble_retriever: EnsembleRetriever,
) -> ContextualCompressionRetriever:
    """
    Create a cross encoder reranker.

    Args:
        ensemble_retriever: The ensemble retriever to rerank results.

    Returns:
        ContextualCompressionRetriever instance.
    """
    logger.info("Creating cross encoder reranker...")
    model_name = settings.CROSS_ENCODER_MODEL_NAME
    model = HuggingFaceCrossEncoder(model_name=model_name)
    compressor = CrossEncoderReranker(model=model, top_n=3)

    return ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )


def save_cross_encoder_reranker(
    cross_encoder_reranker: ContextualCompressionRetriever,
) -> None:
    """
    Save the cross encoder reranker.

    Args:
        cross_encoder_reranker: The reranker object to save.
    """
    try:
        with open(settings.CROSS_ENCODER_RERANKER_PATH, "wb") as file:
            pickle.dump(cross_encoder_reranker, file)
        logger.info("Successfully saved the cross encoder reranker.")
    except Exception as e:
        logger.exception("Failed to save cross encoder reranker.")
        raise e


def retriever_flow() -> None:
    """
    Run the hybrid retriever flow.
    """
    try:
        logger.info("Starting retriever flow...")

        faiss_retriever = load_faiss_index()
        bm25_retriever = load_bm25_index()

        retrievers = [faiss_retriever]
        if bm25_retriever:  # Only add BM25 if it was successfully loaded
            retrievers.append(bm25_retriever)

        ensemble_retriever = create_ensemble_retriever(retrievers)
        cross_encoder_reranker = create_cross_encoder_reranker(ensemble_retriever)

        save_cross_encoder_reranker(cross_encoder_reranker)

        logger.info("Retriever flow completed successfully.")
    except Exception as e:
        logger.exception("Failed to run retriever flow.")
        raise e


if __name__ == "__main__":
    retriever_flow()
