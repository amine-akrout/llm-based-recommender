import os
import sys

from langgraph.graph import END, StateGraph

# Append project root directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.config import settings
from src.recommender.retriever_node import self_query_retrieve
from src.recommender.state import RecState

workflow = StateGraph(RecState)

workflow.add_node("self_query", self_query_retrieve)
workflow.add_edge("self_query", END)

workflow.set_entry_point("self_query")

app = workflow.compile()

app.get_graph().draw_mermaid_png(output_file_path="flow.png")


# Run the workflow
state = {"query": "Woman dress"}
output = app.invoke(state)

print(output)
