"""
This module implements an LLM-based product recommender.
"""

import os
import sys

from langchain.chains.query_constructor.base import (
    get_query_constructor_prompt,
    load_query_constructor_runnable,
)
from langchain.globals import set_debug
from langchain.retrievers import SelfQueryRetriever
from langchain_chroma import Chroma
from langchain_core.runnables import chain
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from loguru import logger

# Append project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import settings

from .utils import CustomChromaTranslator, get_metadata_info


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


def load_chroma_index(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """
    Load the chroma index.
    """
    try:
        logger.info("Loading the chroma index...")
        vectorstore = Chroma(
            collection_name="product_collection",
            persist_directory=settings.CHROMA_INDEX_PATH,
            embedding_function=embeddings,
        )
        logger.info("Chroma index loaded.")
        logger.info(
            f"Number of documents in Chroma index: {vectorstore._collection.count()}"
        )
        return vectorstore
    except Exception as e:
        logger.exception("Failed to load the chroma index.")
        raise e


def self_query_retriever(vector_store: Chroma) -> SelfQueryRetriever:
    """
    Recommends products based on the given query.
    """

    llm_model = ChatOpenAI(
        model=settings.LLM_MODEL_NAME, temperature=settings.LLM_TEMPERATURE
    )

    attribute_info, doc_contents = get_metadata_info()

    chain = load_query_constructor_runnable(
        llm=llm_model, document_contents=doc_contents, attribute_info=attribute_info
    )

    retriever = SelfQueryRetriever(
        query_constructor=chain,
        vectorstore=vector_store,
        verbose=True,
        structured_query_translator=CustomChromaTranslator(),
    )
    return retriever


def self_query_recommender(retriever: SelfQueryRetriever, query: str):
    response = retriever.invoke(query)
    return response


if __name__ == "__main__":
    embeddings = initialize_embeddings_model()
    vectorstore = load_chroma_index(embeddings)
    retriever = self_query_retriever(vectorstore)

    query1 = "woman dress for summer less than 2000"
    query2 = "woman dress for summer less than 2000 and size xl"
    set_debug(True)
    response = retriever.invoke(query1)
    response2 = retriever.invoke(query2)

    for res in response:
        print(res.page_content)
        print("\n" + "-" * 20 + "\n")
