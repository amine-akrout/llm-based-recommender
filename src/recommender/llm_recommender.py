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


llm_model = ChatOpenAI(
    model=settings.LLM_MODEL_NAME, temperature=settings.LLM_TEMPERATURE
)


embeddings = initialize_embeddings_model()
vector_store = load_chroma_index(embeddings)


attribute_info = [
    {
        "name": "Product Details",
        "description": "Details about the product",
    },
    {
        "name": "Brand Name",
        "description": "Name of the brand",
    },
    {
        "name": "Available Sizes",
        "description": "Sizes available for the product",
    },
    {
        "name": "Product Price",
        "description": "Price of the product",
    },
]

doc_contents = "A detailed description of an e-commerce product, including its features, benefits, and specifications."


chain = load_query_constructor_runnable(
    llm=llm_model, document_contents=doc_contents, attribute_info=attribute_info
)

retriever = SelfQueryRetriever(
    query_constructor=chain, vectorstore=vector_store, verbose=True
)

retriever = SelfQueryRetriever.from_llm(
    llm=llm_model,
    vectorstore=vector_store,
    verbose=True,
    document_contents=doc_contents,
    metadata_field_info=attribute_info,
)

query = "woman dress for summer with size XL"
set_debug(True)
response = retriever.invoke(query)


for res in response:
    print(res.page_content)
    print("\n" + "-" * 20 + "\n")
