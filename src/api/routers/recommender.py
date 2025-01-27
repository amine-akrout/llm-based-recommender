"""
Chatbot API Router.
"""

import warnings

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

warnings.filterwarnings("ignore")

from src.recommender.graph import create_recommendaer_graph

router = APIRouter(prefix="/recommend", tags=["Recommender"])

# create graph app at startup
graph_app = None


@router.on_event("startup")
async def startup_event():
    """
    Load retriever and chatbot chain at startup.
    """
    global graph_app
    graph_app = create_recommendaer_graph()


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
        response = graph_app.invoke({"query": request.question})
        recommendation = response.get(
            "recommendation", "No recommendation found for your request."
        )
        return JSONResponse(
            content={"question": request.question, "answer": recommendation}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
