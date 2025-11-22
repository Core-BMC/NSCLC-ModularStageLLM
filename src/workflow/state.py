"""Agent state definition for workflow."""

import operator
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
)

from langchain_core.agents import AgentAction
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """Agent state dictionary for workflow."""

    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str
    intermediate_steps: List[Tuple[AgentAction, str]]
    input: Dict[str, Any]

    # TNM Classifications
    t_classification: Optional[str]
    n_classification: Optional[str]
    m_classification: Optional[str]
    stage_classification: Optional[str]

    # Histology Classification
    histology_category: Optional[str]
    histology_subcategory: Optional[str]
    histology_type: Optional[str]
    histology_confidence: Optional[str]
    histology_reason: Optional[str]

