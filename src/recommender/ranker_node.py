"""
This module implements an LLM-based product recommender.
"""

import os
import pickle
import sys
from typing import List

from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from loguru import logger

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings


def initialize_embeddings_model() -> HuggingFaceEmbeddings:
    """Initializes the HuggingFace embeddings model."""
    try:
        model_name = settings.EMBEDDINGS_MODEL_NAME
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logger.info(f"Successfully initialized embeddings model: {model_name}")
        return embeddings
    except Exception as e:
        logger.exception("Failed to initialize embeddings model.")
        raise e


def load_cross_encoder_model() -> HuggingFaceEmbeddings:
    """Load pickle locally saved cross-encoder model."""
    try:
        with open(settings.CROSS_ENCODER_RERANKER_PATH, "rb") as f:
            cross_encoder = pickle.load(f)
        logger.info("Cross-encoder model loaded.")
        return cross_encoder
    except Exception as e:
        logger.exception("Failed to load cross-encoder model.")
        raise e


def build_ranker(query: str):
    """
    cross encoder retriever.
    """
    cross_encoder = load_cross_encoder_model()

    def format_docs(docs: List[Document]):
        return "\n\n".join([f"- {doc.page_content}" for doc in docs])

    product_docs = cross_encoder.invoke(query)
    logger.info(f"Retrieved {len(product_docs)} documents.")

    product_list = format_docs(product_docs)
    return product_list


if __name__ == "__main__":
    embeddings = initialize_embeddings_model()
    cross_encoder = load_cross_encoder_model()
    query = "woman dress summer"
    product_list = build_ranker(query)
    print(product_list)
