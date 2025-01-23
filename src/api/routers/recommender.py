"""
Chatbot API Router.
"""

import warnings

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

warnings.filterwarnings("ignore")

from src.recommender.llm_recommender import (
    create_recommender_chain,
    initialize_embeddings_model,
    load_chroma_index,
    load_cross_encoder_model,
)

router = APIRouter(prefix="/recommend", tags=["Recommender"])

# Load retriever and chain at startup
embeddings = None
vectorstore = None
cross_encoder = None
recommender_chain = None


@router.on_event("startup")
async def startup_event():
    """
    Load retriever and chatbot chain at startup.
    """
    global embeddings, vectorstore, cross_encoder, recommender_chain
    embeddings = initialize_embeddings_model()
    vectorstore = load_chroma_index(embeddings)
    cross_encoder = load_cross_encoder_model()
    recommender_chain = create_recommender_chain(vectorstore)


class QuestionRequest(BaseModel):
    """
    Request model for a question.
    """

    question: str


@router.post("/", response_model=dict)
def get_chat_response(request: QuestionRequest):
    """
    Get a recommendation to a query from the chatbot.
    """
    try:
        response = recommender_chain.invoke({"query": request.question})
        return JSONResponse(content={"question": request.question, "answer": response})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
