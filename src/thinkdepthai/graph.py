"""RCA Agent LangGraph workflow — 3-node graph: llm_call → tool_node → compress."""

import base64
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from .prompts import PromptManager
from .state import RCAOutputState, RCAState
from .tools import think_tool
from .utils.logger import get_logger
from .utils.rate_limiter import retry_with_backoff

logger = get_logger(__name__)

_SKIP_THOUGHT_SIGNATURE_VALIDATOR_B64 = base64.b64encode(b"skip_thought_signature_validator").decode("utf-8")


@dataclass(frozen=True)
class _GraphConfig:
    """Immutable per-graph configuration captured by node closures."""

    tools_by_name: dict[str, Any] = field(default_factory=dict)
    prompt_path: str = "agents/langgraph/rca.yaml"
    retry_config: dict[str, Any] = field(
        default_factory=lambda: {"max_retries": 5, "min_wait": 1.0, "max_wait": 60.0, "jitter": 1.0}
    )

    async def invoke_with_retry(self, coro_fn) -> Any:
        return await retry_with_backoff(
            coro_fn,
            max_retries=int(self.retry_config["max_retries"]),
            min_wait=self.retry_config["min_wait"],
            max_wait=self.retry_config["max_wait"],
            jitter=self.retry_config["jitter"],
        )


def _fix_gemini_thought_signatures(messages: list) -> list:
    """Fix Gemini thought signatures to work around LangChain bug."""
    for _i, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, "additional_kwargs") and msg.additional_kwargs:
            if "__gemini_function_call_thought_signatures__" in msg.additional_kwargs:
                thought_sigs = msg.additional_kwargs["__gemini_function_call_thought_signatures__"]
                if isinstance(thought_sigs, dict):
                    for tool_call_id in thought_sigs.keys():
                        thought_sigs[tool_call_id] = _SKIP_THOUGHT_SIGNATURE_VALIDATOR_B64
            if "thought_signature" in msg.additional_kwargs:
                msg.additional_kwargs["thought_signature"] = _SKIP_THOUGHT_SIGNATURE_VALIDATOR_B64
    return messages


async def llm_call(state: RCAState, model_with_tools, cfg: _GraphConfig):
    """Analyze current state and decide on next actions."""
    from datetime import datetime

    date_str = datetime.now().strftime("%a %b %-d, %Y")
    prompts = PromptManager.get_prompts(cfg.prompt_path)

    messages = [
        SystemMessage(content=prompts["RCA_ANALYSIS_SP"].format(date=date_str)),
        HumanMessage(
            content=prompts["RCA_ANALYSIS_UP"].format(incident_description=state.get("incident_description", ""))
        ),
    ] + list(state["messages"])

    messages = _fix_gemini_thought_signatures(messages)

    async def _invoke_with_retry():
        try:
            from google.genai.errors import ServerError

            try:
                return await model_with_tools.ainvoke(messages)
            except ServerError as e:
                logger.warning(f"Gemini API server error, will retry: {e}")
                raise
        except ImportError:
            return await model_with_tools.ainvoke(messages)

    response = await cfg.invoke_with_retry(_invoke_with_retry)
    return {"messages": [response]}


async def tool_node(state: RCAState, cfg: _GraphConfig):
    """Execute all tool calls from the previous LLM response."""
    last_message = cast(AIMessage, state["messages"][-1])
    tool_calls = last_message.tool_calls

    observations = []
    for tool_call in tool_calls:
        try:
            tool = cfg.tools_by_name[tool_call["name"]]
            observation = await tool.ainvoke(tool_call["args"])
            observations.append(observation)
        except KeyError:
            error_msg = f"Tool '{tool_call['name']}' not found in available tools"
            logger.warning(error_msg)
            observations.append(f"Tool execution failed: {error_msg}")
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"Error executing tool '{tool_call['name']}': {error_msg}")
            observations.append(f"Tool execution failed: {error_msg}")

    tool_outputs = [
        ToolMessage(content=str(observation), tool_call_id=tool_call["id"])
        for observation, tool_call in zip(observations, tool_calls, strict=False)
    ]
    return {"messages": tool_outputs}


async def compress_rca_findings(state: RCAState, model, cfg: _GraphConfig):
    """Compress RCA findings into a structured report."""
    from datetime import datetime

    prompts = PromptManager.get_prompts(cfg.prompt_path)
    system_message = prompts["COMPRESS_FINDINGS_SP"].format(date=datetime.now().strftime("%a %b %-d, %Y"))
    human_message = prompts["COMPRESS_FINDINGS_UP"].format(
        incident_description=state.get("incident_description", ""),
        date=datetime.now().strftime("%a %b %-d, %Y"),
    )

    messages = (
        [SystemMessage(content=system_message)]
        + list(state.get("messages", []))
        + [HumanMessage(content=human_message)]
    )

    async def _invoke_with_retry():
        return await model.ainvoke(messages)

    response = await cfg.invoke_with_retry(_invoke_with_retry)

    from langchain_core.messages import filter_messages

    raw_notes = [str(m.content) for m in filter_messages(state["messages"], include_types=["tool", "ai"])]

    return {
        "rca_findings": str(response.content),
        "raw_notes": ["\n".join(raw_notes)],
    }


def should_continue(state: RCAState) -> Literal["tool_node", "compress_rca_findings"]:
    """Determine whether to continue investigation or compress findings."""
    messages = state["messages"]
    last_message = cast(AIMessage, messages[-1])
    if last_message.tool_calls:
        return "tool_node"
    return "compress_rca_findings"


def build_rca_agent(
    model,
    model_with_tools,
    tools=None,
    prompt_path: str | None = None,
    retry_config: dict | None = None,
):
    """Build the RCA agent workflow graph."""
    resolved_tools = tools if tools else [think_tool]
    cfg = _GraphConfig(
        tools_by_name={t.name: t for t in resolved_tools},
        prompt_path=prompt_path or "agents/langgraph/rca.yaml",
        retry_config={
            "max_retries": retry_config.get("max_retries", 5),
            "min_wait": retry_config.get("min_wait", 1.0),
            "max_wait": retry_config.get("max_wait", 60.0),
            "jitter": retry_config.get("jitter", 1.0),
        }
        if retry_config
        else {"max_retries": 5, "min_wait": 1.0, "max_wait": 60.0, "jitter": 1.0},
    )

    agent_builder = StateGraph(RCAState, output_schema=RCAOutputState)

    async def llm_call_node(state):
        return await llm_call(state, model_with_tools, cfg)

    async def tool_node_wrapper(state):
        return await tool_node(state, cfg)

    async def compress_node(state):
        return await compress_rca_findings(state, model, cfg)

    agent_builder.add_node("llm_call", llm_call_node)
    agent_builder.add_node("tool_node", tool_node_wrapper)
    agent_builder.add_node("compress_rca_findings", compress_node)

    agent_builder.add_edge(START, "llm_call")
    agent_builder.add_conditional_edges(
        "llm_call",
        should_continue,
        {"tool_node": "tool_node", "compress_rca_findings": "compress_rca_findings"},
    )
    agent_builder.add_edge("tool_node", "llm_call")
    agent_builder.add_edge("compress_rca_findings", END)

    return agent_builder.compile(name="langgraph_rca_agent")
