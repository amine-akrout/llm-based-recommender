"""
This module implements an LLM-based product recommender.
"""

import os
import pickle
import sys
from typing import List

from langchain.chains.query_constructor.base import load_query_constructor_runnable
from langchain.globals import set_debug, set_llm_cache
from langchain.prompts import PromptTemplate
from langchain.retrievers import SelfQueryRetriever
from langchain.schema import Document
from langchain.schema.output_parser import StrOutputParser
from langchain_chroma import Chroma
from langchain_community.cache import InMemoryCache
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnableParallel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from loguru import logger

# local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import settings
from src.recommender.utils import (
    CustomChromaTranslator,
    create_rag_template,
    get_metadata_info,
)


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


def build_self_query_chain(vectorstore: Chroma) -> RunnableLambda:
    """
    Returns a chain (RunnableLambda) that, given {"query": ...}, uses a SelfQueryRetriever
    to fetch documents with advanced filtering. If no docs are found, it will return an empty list.
    """
    set_llm_cache(InMemoryCache())

    llm = ChatOpenAI(
        model=settings.LLM_MODEL_NAME,
        temperature=settings.LLM_TEMPERATURE,
        cache=True,
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


def build_rag_chain():
    """
    RAG retriever.
    """
    set_llm_cache(InMemoryCache())
    prompt = create_rag_template()

    llm = ChatOpenAI(
        model=settings.LLM_MODEL_NAME,
        temperature=settings.LLM_TEMPERATURE,
        max_tokens=settings.LLM_MAX_TOKENS,
        cache=True,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
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


def build_self_query_summary_chain():
    """
    A chain that takes docs + query and returns a quick summary via LLM.
    If you just want to return raw docs, skip the LLM.
    """
    llm = ChatOpenAI(
        model=settings.LLM_MODEL_NAME,
        temperature=settings.LLM_TEMPERATURE,
        cache=True,
    )
    prompt_template = """You found the following documents via Self-Query:

    {docs}

    User Query: {query}

    Provide a brief recommendation summary:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["docs", "query"])
    parser = StrOutputParser()

    summary_chain = prompt | llm | parser

    return summary_chain


def route(docs: List[Document]) -> RunnableBranch:
    """
    Route based on whether docs are provided.
    """
    if len(docs) == 0:
        return build_self_query_summary_chain()
    return build_rag_chain()


def create_recommender_chain(
    vectorstore: Chroma,
) -> RunnableBranch:
    """
    Returns a chain that, given {"query": ...}, fetches documents and generates a recommendation.
    """
    self_query_chain = build_self_query_chain(vectorstore)
    recommender_chain = {
        "docs": self_query_chain,
        "query": lambda x: x["query"],
    } | RunnableLambda(route)
    return recommender_chain


if __name__ == "__main__":
    embeddings = initialize_embeddings_model()
    vectorstore = load_chroma_index(embeddings)
    cross_encoder = load_cross_encoder_model()
    recommender_chain = create_recommender_chain(vectorstore)

    query1 = "woman dress for summer less than 2000"
    query2 = "woman dress for summer less than 500 and size xl"
    set_debug(True)
    response = recommender_chain.invoke({"query": query1})
    response2 = recommender_chain.invoke({"query": query2})

    print(response)

    print(response2)
