import os
import sys

from kaggle.api.kaggle_api_extended import KaggleApi
from loguru import logger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.config import settings


def download_data():
    """Downloads dataset from Kaggle and saves it in the /data folder."""
    if os.path.exists(settings.RAW_DATA_PATH):
        logger.info("Dataset already downloaded. Skipping download.")
    os.makedirs(settings.DATA_DIR, exist_ok=True)
    logger.info(f"Downloading dataset: {settings.DATASET}")
    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            settings.DATASET, path=settings.DATA_DIR, quiet=False, unzip=True
        )
        logger.info(f"Dataset downloaded successfully in '{settings.DATA_DIR}'")
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")


if __name__ == "__main__":
    download_data()
