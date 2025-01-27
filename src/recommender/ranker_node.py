"""
This module implements an cross encoder ranker node for product recommender.
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
from src.recommender.state import RecState


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

    products = format_docs(product_docs)
    return products


def ranker_node(state: RecState) -> RecState:
    """
    Ranker node.
    """
    query = state["query"]
    product_list = build_ranker(query)
    state["products"] = product_list
    return state


if __name__ == "__main__":
    query = "What is the best laptop for gaming?"
    state = {"query": query}
    state = ranker_node(state)
    print(state)
