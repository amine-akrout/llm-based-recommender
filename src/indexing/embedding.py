import os
import pickle
import sys

import pandas as pd
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import settings


def clean_column_names(data):
    """Renames columns to be more descriptive."""
    rename_dict = {
        "BrandName": "Brand Name",
        "Sizes": "Available Sizes",
        "SellPrice": "Product Price",
        "Deatils": "Product Details",
    }

    data = data.rename(
        columns={k: v for k, v in rename_dict.items() if k in data.columns}
    )
    return data


def load_data_preprocess_data(n_samples=100):
    """Loads the e-commerce dataset."""
    if not os.path.exists(settings.RAW_DATA_PATH):
        logger.error(
            f"Dataset not found at {settings.RAW_DATA_PATH}. Run data_loader.py first."
        )
        raise FileNotFoundError(f"Dataset not found at {settings.RAW_DATA_PATH}")

    fashion_df = pd.read_csv(settings.RAW_DATA_PATH)
    logger.info(f"Loaded dataset with {len(fashion_df)} records.")
    fashion_df = fashion_df[["Deatils", "BrandName", "Sizes", "SellPrice"]]
    fashion_df = clean_column_names(fashion_df)
    fashion_df = fashion_df.dropna(subset=["Product Details"])
    if n_samples:
        fashion_df = fashion_df.sample(n_samples)
    # save the processed data
    fashion_df.to_csv(settings.PROCESSED_DATA_PATH, index=False)
    return fashion_df


def generate_documents():
    """Converts DataFrame into a list of LangChain Document objects."""
    try:
        loader = CSVLoader(settings.PROCESSED_DATA_PATH, encoding="utf-8")
        documents = loader.load()
        return documents
    except Exception as e:
        logger.exception("Failed to generate documents.")
        raise e


def initialize_embeddings_model():
    """
    Initialize the HuggingFaceEmbeddings.
    """
    try:
        model_name = settings.EMBEDDINGS_MODEL_NAME
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logger.info(f"Successfully initialized embeddings model: {model_name}")
        return embeddings
    except Exception as e:
        logger.exception("Failed to initialize embeddings model.")
        raise e


def create_faiss_index(embeddings, documents):
    """
    Create a FAISS index.
    """
    try:
        logger.info("Creating FAISS index.")
        faiss_index = FAISS.from_documents(documents, embeddings)
        faiss_index.save_local(settings.FAISS_INDEX_PATH)
        logger.info(f"Successfully created FAISS index at {settings.FAISS_INDEX_PATH}")
    except Exception as e:
        logger.exception("Failed to create FAISS index.")
        raise e


def create_bm25_index(documents):
    """
    Create a BM25 index.
    """
    try:
        if os.path.exists(settings.BM25_INDEX_PATH):
            os.makedirs(settings.INDEX_DIR, exist_ok=True)
        logger.info("Creating BM25 index.")
        bm25_index = BM25Retriever.from_documents(documents)
        with open(settings.BM25_INDEX_PATH, "wb") as f:
            pickle.dump(bm25_index, f)
        logger.info(f"Successfully created BM25 index at {settings.BM25_INDEX_PATH}")
    except Exception as e:
        logger.exception("Failed to create BM25 index.")
        raise e


def embedding_pipeline(n_samples=100):
    """
    Run the embedding pipeline.
    """
    fashion_df = load_data_preprocess_data(n_samples=n_samples)
    documents = generate_documents()
    embeddings = initialize_embeddings_model()
    create_faiss_index(embeddings, documents)
    create_bm25_index(documents)
