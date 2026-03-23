"""State definitions for the RCA Agent workflow."""

import operator
from collections.abc import Sequence
from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class RCAState(TypedDict):
    """State for the RCA agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]
    tool_call_iterations: int
    incident_description: str
    rca_findings: str
    raw_notes: Annotated[list[str], operator.add]


class RCAOutputState(TypedDict):
    """Output state for the RCA agent."""

    rca_findings: str
    raw_notes: Annotated[list[str], operator.add]
    messages: Annotated[Sequence[BaseMessage], add_messages]
