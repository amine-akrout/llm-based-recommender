"""Configuration settings for the project."""

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class Settings(BaseSettings):
    """Project settings."""

    model_config = SettingsConfigDict(
        env_file="./.env", env_file_encoding="utf-8", extra="allow"
    )

    # Base paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    INDEX_DIR: Path = DATA_DIR / "indexes"

    # Data settings
    DATASET: str = "mukuldeshantri/ecommerce-fashion-dataset"
    RAW_DATA_PATH: str = str(DATA_DIR / "FashionDataset.csv")
    PROCESSED_DATA_PATH: str = str(DATA_DIR / "processed_data.csv")

    # Kaggle settings
    KAGGLE_USERNAME: str = os.environ.get("KAGGLE_USERNAME", "")
    KAGGLE_KEY: SecretStr = SecretStr(os.environ.get("KAGGLE_KEY", ""))

    # Embeddings settings
    # EMBEDDINGS_MODEL_NAME: str = "BAAI/llm-embedder"
    EMBEDDINGS_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    # CROSS_ENCODER_MODEL_NAME: str = "BAAI/bge-reranker-base"
    CROSS_ENCODER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # LLM settings
    LLM_MODEL_NAME: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0
    LLM_MAX_TOKENS: int = 100
    # OLLAMA_MODEL_NAME: str = "llama3.2:3b"
    OLLAMA_MODEL_NAME: str = "qwen2.5"
    # OLLAMA_MODEL_NAME: str = "deepseek-r1"

    FAISS_INDEX_PATH: str = str(INDEX_DIR / "faiss_index.faiss")
    BM25_INDEX_PATH: str = str(INDEX_DIR / "bm25_index.pkl")
    CROSS_ENCODER_RERANKER_PATH: str = str(INDEX_DIR / "cross_encoder_reranker.pkl")
    CHROMA_INDEX_PATH: str = str(INDEX_DIR / "chroma_index")

    # Guadrail settings
    GUARDRAIL_SETTINGS_DIR: str = str(BASE_DIR / "src" / "core" / "guardrail")

    FAISS_TOP_K: int = 3
    BM25_TOP_K: int = 3

    RETRIEVER_TOP_K: int = 5
    RETRIEVER_WEIGHTS: list[float] = [0.5, 0.5]

    COMPRESSOR_TOP_K: int = 2

    # Open AI API settings
    OPENAI_API_KEY: SecretStr | None = None

    # Lanfuse settings
    LANGFUSE_SECRET_KEY: SecretStr | None = None
    LANGFUSE_PUBLIC_KEY: SecretStr | None = None
    LANGFUSE_HOST: str = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")

    # Logging settings
    LOGGING_LEVEL: str = "INFO"
    LOGGING_FILE: str = str(BASE_DIR / "logs" / "preprocessing.log")

    # Ensure that the data directory exists
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.INDEX_DIR, exist_ok=True)
        os.makedirs(self.BASE_DIR / "logs", exist_ok=True)


settings = Settings()
