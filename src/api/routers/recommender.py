"""
Chatbot API Router.
"""

import warnings
from typing import Optional
from uuid import uuid4

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
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


def get_or_create_thread_id(
    request: Request,
    thread_id: Optional[str] = Cookie(default=None),
) -> str:
    """
    Get or create a thread ID for the user session.
    """
    if thread_id is None:
        logger.info("Creating new thread ID for the user session.")
        new_id = str(uuid4())
        request.state.new_thread_id = new_id  # mark for setting cookie
        return new_id
    return thread_id


@router.post("/", response_model=dict)
def get_chat_response(
    request: Request,
    response: Response,
    body: QuestionRequest,
    thread_id: str = Depends(get_or_create_thread_id),
):
    try:
        # Set cookie if a new thread ID was generated
        if hasattr(request, "state") and hasattr(request.state, "new_thread_id"):
            response.set_cookie("thread_id", request.state.new_thread_id)

        config = {"configurable": {"thread_id": thread_id}}
        result = graph_app.invoke({"query": body.question}, config=config)

        recommendation = result.get("recommendation", "No recommendation found.")
        return {
            "question": body.question,
            "thread_id": thread_id,
            "answer": recommendation,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
