"""Convert LangChain messages to the rcabench-platform SDK trajectory schema."""

from __future__ import annotations

import json

from rcabench_platform.v3.sdk.llm_eval.trajectory import AgentTrajectory, Message, ToolCall, Turn


class TrajectoryConverter:
    """Converts LangChain messages to the SDK's canonical Trajectory format."""

    @staticmethod
    def from_langchain_messages(
        messages: list,
        agent_name: str = "RCA-Agent",
        system_prompt: str = "",
    ) -> AgentTrajectory:
        """Convert LangChain BaseMessage objects to an AgentTrajectory."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        flat_messages: list[Message] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                flat_messages.append(Message(role="system", content=content))
            elif isinstance(msg, HumanMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                flat_messages.append(Message(role="user", content=content))
            elif isinstance(msg, AIMessage):
                content = msg.content if isinstance(msg.content, str) else ""
                tool_calls = None
                if msg.tool_calls:
                    tool_calls = [
                        ToolCall(
                            id=tc.get("id") or "",
                            name=tc.get("name", ""),
                            arguments=json.dumps(tc.get("args", {}), ensure_ascii=False),
                        )
                        for tc in msg.tool_calls
                    ]
                flat_messages.append(Message(role="assistant", content=content, tool_calls=tool_calls))
            elif isinstance(msg, ToolMessage):
                content = msg.content if isinstance(msg.content, str) else json.dumps(msg.content)
                flat_messages.append(Message(role="tool", content=content, tool_call_id=msg.tool_call_id))
            else:
                content = str(msg.content) if hasattr(msg, "content") else str(msg)
                flat_messages.append(Message(role="assistant", content=content))

        turns = _flat_messages_to_turns(flat_messages)
        return AgentTrajectory(agent_name=agent_name, system_prompt=system_prompt, turns=turns)


def _flat_messages_to_turns(messages: list[Message]) -> list[Turn]:
    """Group a flat list of Messages into Turn objects."""
    turns: list[Turn] = []
    current_msgs: list[Message] = []

    for msg in messages:
        if msg.role in ("system", "user"):
            if current_msgs:
                turns.append(Turn(messages=current_msgs))
                current_msgs = []
            turns.append(Turn(messages=[msg]))
        elif msg.role == "assistant":
            if current_msgs:
                turns.append(Turn(messages=current_msgs))
                current_msgs = []
            current_msgs.append(msg)
        elif msg.role in ("tool", "sub_agent"):
            current_msgs.append(msg)
        else:
            current_msgs.append(msg)

    if current_msgs:
        turns.append(Turn(messages=current_msgs))

    return turns
