"""
This module implements a RAG (Retrieval-Augmented Generation) pipeline for an LLM-based product recommender.
"""

import os
import sys

from langchain.globals import set_llm_cache
from langchain.schema.output_parser import StrOutputParser
from langchain_community.cache import InMemoryCache
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_ollama import ChatOllama
from loguru import logger

# Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings
from src.recommender.state import RecState
from src.recommender.utils import create_rag_template


def build_rag_chain():
    """
    Builds and returns a RAG chain for product recommendations.
    """
    # Set up in-memory caching for the LLM
    set_llm_cache(InMemoryCache())

    # Initialize the LLM
    llm = ChatOllama(
        model=settings.OLLAMA_MODEL_NAME,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        cache=True,
    )

    # Create the RAG prompt template
    prompt = create_rag_template()

    # Initialize the output parser
    parser = StrOutputParser()

    # Define the RAG chain
    rag_chain = (
        RunnableParallel(
            {
                "docs": RunnableLambda(lambda x: x["docs"]),
                "query": RunnableLambda(lambda x: x["query"]),
            }
        )
        | prompt
        | llm
        | parser
    )

    return rag_chain


def rag_recommender(state: RecState) -> RecState:
    """
    RAG recommender node.
    """
    rag_chain = build_rag_chain()
    query = state["query"]
    docs = state["products"]

    output = rag_chain.invoke({"docs": docs, "query": query})
    state["output"] = output
    return state


if __name__ == "__main__":
    # Example documents and query
    docs = """- {\n  "Product Details": "floral polyester square neck womens regular dress - white",
                \n  "Brand Name": "and",\n  "Available Sizes": "8, 10, 14, 16, 18",\n 
                "Product Price": 2319.0\n}"""
    query = "What are the available sizes for the floral polyester square neck womens regular dress - white?"

    # Build the RAG chain
    rag_chain = build_rag_chain()

    # Invoke the chain with the input
    input_data = {"docs": docs, "query": query}
    output = rag_chain.invoke(input_data)
