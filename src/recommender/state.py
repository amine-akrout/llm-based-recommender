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
    documents: List[str]
        The retrieved documents.
    answer_status: str
        The answer status.

    """

    query: str
    # question_status: str
    # on_topic: bool
    # prompt: str
    llm_output: str
    documents: List[str]
    # answer_status: str
