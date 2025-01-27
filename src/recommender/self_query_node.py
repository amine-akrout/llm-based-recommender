"""
This module contains the self-query retriever node, which retrieves products using the self-query retriever.
"""

import os
import sys
from functools import lru_cache
from typing import List

from langchain.chains.query_constructor.base import load_query_constructor_runnable
from langchain.retrievers import SelfQueryRetriever
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_core.runnables import RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_fixed

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings
from src.recommender.state import RecState
from src.recommender.utils import CustomChromaTranslator, get_metadata_info


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
@lru_cache(maxsize=1)
def initialize_embeddings_model() -> HuggingFaceEmbeddings:
    """Initializes the HuggingFace embeddings model with retries and caching."""
    try:
        model_name = settings.EMBEDDINGS_MODEL_NAME
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logger.info(f"Successfully initialized embeddings model: {model_name}")
        return embeddings
    except Exception as e:
        logger.exception("Failed to initialize embeddings model.")
        raise e


def load_chroma_index(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Load the chroma index with caching.
    """
    try:
        logger.info("Loading the chroma index...")
        vectorstore = Chroma(
            collection_name="product_collection",
            embedding_function=embeddings,
            persist_directory=settings.CHROMA_INDEX_PATH,
        )
        logger.info("Chroma index loaded.")
        logger.info(
            f"Number of documents in Chroma index: {vectorstore._collection.count()}"
        )
        return vectorstore
    except Exception as e:
        logger.exception("Failed to load the chroma index.")
        raise e


def build_self_query_chain(vectorstore: Chroma) -> RunnableLambda:
    """
    Returns a chain (RunnableLambda) that, given {"query": ...}, uses a SelfQueryRetriever
    to fetch documents with advanced filtering. If no docs are found, it will return an empty list.
    """
    llm = ChatOpenAI(
        model=settings.LLM_MODEL_NAME,
        temperature=settings.LLM_TEMPERATURE,
    )

    attribute_info, doc_contents = get_metadata_info()

    # Build the query-constructor chain
    query_constructor = load_query_constructor_runnable(
        llm=llm,
        document_contents=doc_contents,
        attribute_info=attribute_info,
    )

    # Create a SelfQueryRetriever
    retriever = SelfQueryRetriever(
        query_constructor=query_constructor,
        vectorstore=vectorstore,
        verbose=True,
        structured_query_translator=CustomChromaTranslator(),
    )
    self_query_chain = RunnableLambda(lambda inputs: retriever.invoke(inputs["query"]))
    return self_query_chain


def self_query_retrieve(state: RecState) -> RecState:
    """
    Given a RecState, retrieve products using the self-query retriever.
    """
    embeddings = initialize_embeddings_model()
    chroma_index = load_chroma_index(embeddings)
    self_query_chain = build_self_query_chain(chroma_index)

    def format_docs(docs: List[Document]):
        return "\n\n".join([f"- {doc.page_content}" for doc in docs])

    query = state["query"]
    logger.info(f"Processing query: {query}")

    # Retrieve products
    results = self_query_chain.invoke({"query": query})
    logger.info(f"Retrieved {len(results)} products for query: {query}")
    if len(results) == 0:
        logger.warning("No products found for the query.")
        state["self_query_state"] = "empty"
    else:
        state["self_query_state"] = "success"
        state["products"] = format_docs(results)
    return state
