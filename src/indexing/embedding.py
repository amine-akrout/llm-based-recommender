"""
This module processes the e-commerce dataset, generates embeddings, 
and indexes them using FAISS (vector search) and BM25 (lexical search).
"""

import json
import os
import pickle
import sys
from typing import Optional

import pandas as pd
from langchain_chroma import Chroma
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
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
        "Deatils": "Product Details",
    }
    return df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})


def load_and_preprocess_data(n_samples: Optional[int] = 2000) -> pd.DataFrame:
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

    df.dropna(inplace=True)

    if n_samples and n_samples < len(df):
        df = df.sample(n_samples, random_state=42)

    # Save processed data
    df.to_csv(settings.PROCESSED_DATA_PATH, index=False)
    logger.info(f"Processed dataset saved to {settings.PROCESSED_DATA_PATH}")

    return df


def generate_documents(use_csv_loader: bool = False) -> list:
    """Converts CSV data into LangChain Document objects."""
    if use_csv_loader:
        try:
            loader = CSVLoader(settings.PROCESSED_DATA_PATH, encoding="utf-8")
            documents = loader.load()
            logger.info(f"Generated {len(documents)} documents.")
            return documents
        except Exception as e:
            logger.exception("Failed to generate documents.")
            raise e

    def convert_price(value):
        try:
            return float(str(value).replace(",", "").strip())
        except ValueError:
            return 0.0

    def convert_sizes(value):
        """Convert sizes from string to a comma-separated string."""
        if pd.isna(value) or not isinstance(value, str):
            return ""
        return ", ".join(
            size.strip().lower() for size in value.replace("Size:", "").split(",")
        )

    df = pd.read_csv(settings.PROCESSED_DATA_PATH)
    # Convert "Available Sizes"
    if "Available Sizes" in df.columns:
        df["Available Sizes"] = df["Available Sizes"].apply(convert_sizes)

    # Convert "Product Price" to float
    if "Product Price" in df.columns:
        df["Product Price"] = df["Product Price"].apply(convert_price)

    documents = [
        Document(
            page_content=json.dumps(row.to_dict(), indent=2),
            metadata=row.to_dict(),
            id=row.name,
        )
        for _, row in df.iterrows()
    ]
    logger.info(f"Generated {len(documents)} documents.")
    return documents


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


def create_chroma_index(embeddings: HuggingFaceEmbeddings, documents: list) -> None:
    """Creates and saves a Chroma index."""
    try:
        logger.info("Creating Chroma index...")
        vector_store = Chroma(
            collection_name="product_collection",
            embedding_function=embeddings,
            persist_directory=settings.CHROMA_INDEX_PATH,
        )
        vector_store.add_documents(documents)
        logger.info(f"Chroma index saved at {settings.CHROMA_INDEX_PATH}")
        logger.info(f"Number of documents in Chroma index: {len(documents)}")
    except Exception as e:
        logger.exception("Failed to create Chroma index.")
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


def embedding_pipeline(n_samples: Optional[int] = 2000) -> None:
    """Runs the entire embedding pipeline."""
    try:
        df = load_and_preprocess_data(n_samples)
        documents = generate_documents()
        embeddings = initialize_embeddings_model()

        create_faiss_index(embeddings, documents)
        create_bm25_index(documents)
        create_chroma_index(embeddings, documents)

        logger.info("Embedding pipeline completed successfully.")
    except Exception as e:
        logger.exception("Failed to run embedding pipeline.")
        raise e


if __name__ == "__main__":
    embedding_pipeline()
