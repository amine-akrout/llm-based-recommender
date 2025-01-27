"""
This module implements an LLM-based product recommender.
"""

import os
import pickle
import sys
from typing import List

from langchain.globals import set_debug, set_llm_cache
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain_community.cache import InMemoryCache
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from loguru import logger

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings
from src.recommender.utils import create_rag_template


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


def build_rag_chain():
    """
    RAG retriever.
    """
    set_llm_cache(InMemoryCache())
    prompt = create_rag_template()

    llm = ChatOllama(
        model=settings.OLLAMA_MODEL_NAME,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        cache=True,
    )
    parser = StrOutputParser()
    cross_encoder = load_cross_encoder_model()

    def format_docs(docs: List[Document]):
        return "\n\n".join([f"- {doc.page_content}" for doc in docs])

    # define the chain
    rag_chain = (
        RunnableParallel(
            {
                "docs": (
                    RunnableLambda(lambda input_dict: input_dict["query"])
                    | RunnableLambda(lambda query_str: cross_encoder.invoke(query_str))
                    | RunnableLambda(format_docs)
                ),
                "query": RunnableLambda(lambda input_dict: input_dict["query"]),
            }
        )
        | prompt
        | llm
        | parser
    )
    return rag_chain


if __name__ == "__main__":
    embeddings = initialize_embeddings_model()
    vectorstore = load_chroma_index(embeddings)
    cross_encoder = load_cross_encoder_model()
    recommender_chain = create_recommender_chain(vectorstore)

    # recommender_chain.get_graph().draw_mermaid_png(output_file_path="recommender_chain.png")

    query1 = "woman dress for summer less than 2000"
    query2 = "woman dress for summer less than 500 and size xl"
    set_debug(True)
    response = recommender_chain.invoke({"query": query1})
    response2 = recommender_chain.invoke({"query": query2})

    print(response)

    print(response2)
