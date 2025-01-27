from typing import List, TypedDict


class RecState(TypedDict):
    """
    Graph state.

    Attributes:
    -----------
    question: str
        The question.
    question_status: str
        The question status.
    on_topic: bool
        The topic status.
    prompt: str
        The prompt.
    llm_output: str
        The LLM generation.
    products: str
        The retrieved products.
    answer_status: str
        The answer status.
    self_query_state: str
        The self-query state.

    """

    query: str
    # question_status: str
    # on_topic: bool
    # prompt: str
    llm_output: str
    products: str
    self_query_state: str
    # answer_status: str
