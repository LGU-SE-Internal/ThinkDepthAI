"""Tools for the RCA Agent."""

import json

from langchain_core.tools import tool


@tool
def think_tool(reasoning: str) -> str:
    """Reflect on findings and plan next investigation steps.

    Use this tool to analyze findings so far and decide what to investigate next.

    Args:
        reasoning: Your analysis and reasoning about findings and next steps

    Returns:
        Confirmation that reasoning was recorded
    """
    return json.dumps(
        {"status": "recorded", "reasoning": reasoning, "next_action": "Continue with planned investigation"}
    )
