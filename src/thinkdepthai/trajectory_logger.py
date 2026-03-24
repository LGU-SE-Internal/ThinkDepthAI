"""JSONL trajectory logger for debugging agent runs."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
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

    def __init__(self, output_dir: str | Path = "./trajectories", run_id: str | None = None):
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._file_path = self._output_dir / f"{self._run_id}.jsonl"
        self._step = 0
        self._file = open(self._file_path, "w", encoding="utf-8")
        self._start_time = time.monotonic()
        logger.info(f"Trajectory log: {self._file_path}")

    @property
    def file_path(self) -> Path:
        return self._file_path

    def log(self, event_type: str, data: Any = None) -> None:
        """Write a single event to the JSONL file."""
        self._step += 1
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_s": round(time.monotonic() - self._start_time, 2),
            "step": self._step,
            "type": event_type,
            "data": data,
        }
        line = json.dumps(entry, ensure_ascii=False, default=str)
        self._file.write(line + "\n")
        self._file.flush()

    def log_langchain_messages(self, messages: list) -> None:
        """Log all LangChain messages from a completed run as individual events."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

        for msg in messages:
            if isinstance(msg, SystemMessage):
                self.log("system_message", {"content": _truncate(str(msg.content), 2000)})
            elif isinstance(msg, HumanMessage):
                self.log("user_message", {"content": _truncate(str(msg.content), 2000)})
            elif isinstance(msg, AIMessage):
                data: dict[str, Any] = {"content": _truncate(str(msg.content), 2000)}
                if msg.tool_calls:
                    data["tool_calls"] = [
                        {"id": tc.get("id", ""), "name": tc.get("name", ""), "args": tc.get("args", {})}
                        for tc in msg.tool_calls
                    ]
                if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                    data["usage"] = dict(msg.usage_metadata)
                self.log("assistant_message", data)
            elif isinstance(msg, ToolMessage):
                self.log(
                    "tool_result",
                    {
                        "tool_call_id": msg.tool_call_id,
                        "content": _truncate(str(msg.content), 3000),
                    },
                )
            else:
                self.log("unknown_message", {"type": type(msg).__name__, "content": _truncate(str(msg), 1000)})

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
