"""JSONL trajectory logger for debugging agent runs."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .utils.logger import get_logger

logger = get_logger(__name__)


class TrajectoryLogger:
    """Writes agent events to a JSONL file for debugging.

    Each line is a JSON object with:
      - timestamp: ISO 8601
      - type: event type (llm_start, llm_end, tool_call, tool_result, state, error, ...)
      - data: event payload
      - step: sequential step number
    """

    def __init__(
        self,
        output_dir: str | Path = "./trajectories",
        run_id: str | None = None,
        filename: str | None = None,
    ):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id or datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        # Use explicit filename if given, otherwise fall back to run_id
        stem = filename or self._run_id
        self._file_path = self._output_dir / f"{stem}.jsonl"
        self._step = 0
        self._file = open(self._file_path, "w", encoding="utf-8")
        self._start_time = time.monotonic()
        # Write _meta header (skipped by dashboard event reader)
        meta = json.dumps({"_meta": {"run_id": self._run_id}}, ensure_ascii=False)
        self._file.write(meta + "\n")
        self._file.flush()
        logger.info(f"Trajectory log: {self._file_path}")

    @property
    def file_path(self) -> Path:
        return self._file_path

    def log(self, event_type: str, data: Any = None, agent_path: list[str] | None = None) -> None:
        """Write a single event to the JSONL file.

        Args:
            event_type: Event type name (e.g. ``llm_end``, ``tool_call``).
            data: Event payload dict.
            agent_path: List of agent IDs for hierarchical display.
        """
        self._step += 1
        entry: dict[str, Any] = {
            "run_id": self._run_id,
            "seq": self._step,
            "timestamp": datetime.now(UTC).isoformat(),
            "elapsed_s": round(time.monotonic() - self._start_time, 2),
            "step": self._step,
            "event_type": event_type,
            "agent_path": agent_path or ["orchestrator"],
            "data": data,
        }
        line = json.dumps(entry, ensure_ascii=False, default=str)
        self._file.write(line + "\n")
        self._file.flush()

    def log_langchain_messages(self, messages: list) -> None:
        """Log all LangChain messages using dashboard-compatible event types."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        for msg in messages:
            if isinstance(msg, SystemMessage):
                self.log("llm_start", {
                    "messages": [{"type": "system", "content": _truncate(str(msg.content), 2000)}],
                })
            elif isinstance(msg, HumanMessage):
                self.log("llm_start", {
                    "messages": [{"type": "human", "content": _truncate(str(msg.content), 2000)}],
                })
            elif isinstance(msg, AIMessage):
                data: dict[str, Any] = {"content": _truncate(str(msg.content), 2000)}
                if msg.tool_calls:
                    data["tool_calls"] = [
                        {"id": tc.get("id", ""), "name": tc.get("name", ""), "args": tc.get("args", {})}
                        for tc in msg.tool_calls
                    ]
                if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                    data["usage"] = dict(msg.usage_metadata)
                self.log("llm_end", data)
            elif isinstance(msg, ToolMessage):
                self.log("tool_result", {
                    "tool_name": getattr(msg, "name", ""),
                    "tool_call_id": msg.tool_call_id,
                    "result": _truncate(str(msg.content), 3000),
                })
            else:
                self.log("llm_end", {"content": _truncate(str(msg), 1000)})

    def log_result(self, final_output: str, trace_id: str) -> None:
        """Log the final result."""
        self.log("result", {"trace_id": trace_id, "final_output": final_output})

    def close(self) -> None:
        """Close the JSONL file."""
        if self._file and not self._file.closed:
            elapsed = round(time.monotonic() - self._start_time, 2)
            self.log("run_complete", {"total_steps": self._step, "elapsed_s": elapsed})
            self._file.close()
            logger.info(f"Trajectory saved: {self._file_path} ({self._step} events, {elapsed}s)")


def _truncate(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"... (truncated, {len(s)} chars total)"
