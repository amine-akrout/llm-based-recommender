"""
This module processes the e-commerce dataset, generates embeddings, 
and indexes them using FAISS (vector search) and BM25 (lexical search).
"""

import os
import pickle
import sys
from typing import Optional

import pandas as pd
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from loguru import logger

# Append project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import settings


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Renames columns to be more descriptive."""
    rename_dict = {
        "BrandName": "Brand Name",
        "Sizes": "Available Sizes",
        "SellPrice": "Product Price",
        "Deatils": "Product Details",  # Assuming this is a typo of "Details"
    }
    return df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})


def load_and_preprocess_data(n_samples: Optional[int] = 100) -> pd.DataFrame:
    """Loads the dataset, preprocesses it, and saves the processed version."""
    if not os.path.exists(settings.RAW_DATA_PATH):
        logger.error(
            f"Dataset not found at {settings.RAW_DATA_PATH}. Run data_loader.py first."
        )
        raise FileNotFoundError(f"Dataset not found at {settings.RAW_DATA_PATH}")

    df = pd.read_csv(settings.RAW_DATA_PATH)
    logger.info(f"Loaded dataset with {len(df)} records.")

    df = clean_column_names(df)

    # Ensure only existing columns are kept to avoid KeyError
    valid_columns = [
        "Product Details",
        "Brand Name",
        "Available Sizes",
        "Product Price",
    ]
    df = df[[col for col in valid_columns if col in df.columns]]

    df.dropna(subset=["Product Details"], inplace=True)

    if n_samples and n_samples < len(df):
        df = df.sample(n_samples, random_state=42)

    # Save processed data
    df.to_csv(settings.PROCESSED_DATA_PATH, index=False)
    logger.info(f"Processed dataset saved to {settings.PROCESSED_DATA_PATH}")

    return df


def generate_documents() -> list:
    """Converts CSV data into LangChain Document objects."""
    try:
        loader = CSVLoader(settings.PROCESSED_DATA_PATH, encoding="utf-8")
        documents = loader.load()
        logger.info(f"Generated {len(documents)} documents.")
        return documents
    except Exception as e:
        logger.exception("Failed to generate documents.")
        raise e


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


def create_faiss_index(embeddings: HuggingFaceEmbeddings, documents: list) -> None:
    """Creates and saves a FAISS index."""
    try:
        logger.info("Creating FAISS index...")
        faiss_index = FAISS.from_documents(documents, embeddings)
        faiss_index.save_local(settings.FAISS_INDEX_PATH)
        logger.info(f"FAISS index saved at {settings.FAISS_INDEX_PATH}")
    except Exception as e:
        logger.exception("Failed to create FAISS index.")
        raise e


def create_bm25_index(documents: list) -> None:
    """Creates and saves a BM25 index."""
    try:
        os.makedirs(os.path.dirname(settings.BM25_INDEX_PATH), exist_ok=True)

        logger.info("Creating BM25 index...")
        bm25_index = BM25Retriever.from_documents(documents)

        with open(settings.BM25_INDEX_PATH, "wb") as f:
            pickle.dump(bm25_index, f)

        logger.info(f"BM25 index saved at {settings.BM25_INDEX_PATH}")
    except Exception as e:
        logger.exception("Failed to create BM25 index.")
        raise e


def embedding_pipeline(n_samples: Optional[int] = 100) -> None:
    """Runs the entire embedding pipeline."""
    try:
        df = load_and_preprocess_data(n_samples)
        documents = generate_documents()
        embeddings = initialize_embeddings_model()

        create_faiss_index(embeddings, documents)
        create_bm25_index(documents)

        logger.info("Embedding pipeline completed successfully.")
    except Exception as e:
        logger.exception("Failed to run embedding pipeline.")
        raise e


if __name__ == "__main__":
    embedding_pipeline()
