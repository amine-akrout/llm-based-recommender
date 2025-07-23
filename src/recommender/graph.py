"""
Recommender Graph Workflow"""

import os
import sys

from langchain.globals import set_debug
from langgraph.graph import END, StateGraph
from loguru import logger

# Append project root directory
# pylint: disable=wrong-import-position
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.recommender.check_topic_node import topic_classifier
from src.recommender.rag_node import rag_recommender
from src.recommender.ranker_node import ranker_node
from src.recommender.self_query_node import self_query_retrieve
from src.recommender.state import RecState

set_debug(True)


def create_recommendaer_graph():
    """
    Creates the recommender graph workflow.
    """
    workflow = StateGraph(RecState)

    workflow.add_node("self_query_retrieve", self_query_retrieve)
    workflow.add_node("rag_recommender", rag_recommender)
    workflow.add_node("ranker", ranker_node)
    workflow.add_node("check_topic", topic_classifier)

    workflow.add_edge("ranker", "rag_recommender")
    workflow.add_edge("rag_recommender", END)

    workflow.set_entry_point("check_topic")
    workflow.add_conditional_edges(
        "check_topic",
        lambda state: state["on_topic"],
        {"Yes": "self_query_retrieve", "No": END},
    )

    workflow.add_conditional_edges(
        "self_query_retrieve",
        lambda state: state["self_query_state"],
        {"success": "rag_recommender", "empty": "ranker"},
    )
    return workflow.compile()


if __name__ == "__main__":
    app = create_recommendaer_graph()
    # app.get_graph().draw_mermaid_png(output_file_path="flow.png")
    # Run the workflow
    state = {"query": "Woman dress less than 50"}
    output = app.invoke(state)

    logger.info(output)
