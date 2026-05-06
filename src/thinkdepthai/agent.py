"""LanggraphRCAAgent — standalone Root Cause Analysis agent."""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage
from rcabench_platform.v3.sdk.llm_eval.trajectory import Trajectory

from .config import load_agent_config
from .config.schema import AgentConfig
from .converters import TrajectoryConverter
from .graph import build_rca_agent
from .llm import create_langchain_model
from .prompts import PromptManager
from .tools import think_tool
from .tools_lib import AsyncBaseToolkit, toolkit_to_langchain_tools
from .tools_lib.query_parquet_toolkit import QueryParquetFilesToolkit
from .trajectory_logger import TrajectoryLogger
from .utils.logger import get_logger

logger = get_logger(__name__)

# Toolkit registry
TOOLKIT_MAP: dict[str, type[AsyncBaseToolkit]] = {
    "query_parquet_files": QueryParquetFilesToolkit,
}


@dataclass
class AgentResult:
    """Standardized result from an agent run."""

    task: str = ""
    trace_id: str = ""
    final_output: str = ""
    trajectory: Trajectory = field(default_factory=Trajectory)
    metadata: dict[str, Any] = field(default_factory=dict)


class LanggraphRCAAgent:
    """Root Cause Analysis agent using LangGraph.

    Standalone implementation — no BaseAgent inheritance.
    """

    def __init__(
        self,
        *,
        config: AgentConfig | str | None = None,
        name: str | None = None,
        trajectory_dir: str | None = None,
        on_usage: Callable[[dict[str, Any]], None] | None = None,
    ):
        if isinstance(config, AgentConfig):
            self.config = config
        elif isinstance(config, str):
            self.config = load_agent_config(config)
        else:
            self.config = load_agent_config()

        if name:
            self.config.agent.name = name

        self._rca_agent = None
        self._toolkits: dict[str, AsyncBaseToolkit] = {}
        self._initialized = False
        self._trajectory_dir = trajectory_dir
        self._on_usage = on_usage
        self._current_traj_logger: TrajectoryLogger | None = None
        self._current_trace_id: str | None = None
        self._usage_totals: dict[str, Any] = {}

    async def __aenter__(self) -> LanggraphRCAAgent:
        await self.build()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.cleanup()

    async def build(self):
        """Build the RCA agent: load toolkits, create model, compile graph."""
        if self._initialized:
            return

        # Load toolkits from config
        for tk_name, tk_config in self.config.toolkits.items():
            name = tk_config.name or tk_name
            if name in TOOLKIT_MAP:
                toolkit = TOOLKIT_MAP[name](tk_config.config)
                self._toolkits[name] = toolkit
                logger.info(f"Loaded toolkit: {name}")

        # Get base tools
        base_tools = [think_tool]

        # Get toolkit tools via LangChain adapter
        toolkit_tools: list = []
        for toolkit in self._toolkits.values():
            toolkit_tools.extend(toolkit_to_langchain_tools(toolkit))

        all_tools = base_tools + toolkit_tools

        # Create LangChain model
        langchain_model = create_langchain_model(self.config.model)
        model_with_tools = langchain_model.bind_tools(all_tools)

        # Get prompt path
        prompt_path = self.config.agent.prompt_path or "agents/langgraph/rca.yaml"
        logger.info(f"Using prompt path: {prompt_path}")

        # Get retry config
        retry_config = {
            "max_retries": self.config.model.rate_limit.max_retries,
            "min_wait": self.config.model.rate_limit.retry_min_wait,
            "max_wait": self.config.model.rate_limit.retry_max_wait,
            "jitter": self.config.model.rate_limit.retry_jitter,
        }

        # Build the graph
        self._rca_agent = build_rca_agent(
            langchain_model,
            model_with_tools,
            tools=all_tools,
            prompt_path=prompt_path,
            retry_config=retry_config,
            usage_callback=self._handle_usage,
        )

        self._initialized = True
        logger.info("RCA Agent built successfully")

    async def cleanup(self):
        """Release resources."""
        for toolkit in self._toolkits.values():
            if hasattr(toolkit, "cleanup"):
                await toolkit.cleanup()
        self._toolkits.clear()

    async def run(
        self,
        input: str,
        trace_id: str | None = None,
        data_dir: str | None = None,
        trajectory_filename: str | None = None,
    ) -> AgentResult:
        """Run RCA analysis on an incident.

        Uses astream(stream_mode="updates") for real-time JSONL logging of each
        graph node execution, while accumulating the final state for the result.

        Args:
            input: Incident description text.
            trace_id: Optional trace ID for this run.
            data_dir: Path to the directory containing parquet data files.
                      Appended to the incident description so the LLM knows
                      where to find the telemetry data.
            trajectory_filename: Stem for the trajectory JSONL file
                (e.g. ``"sock-shop_case42"``).  Falls back to *trace_id*.
        """
        if not self._initialized:
            await self.build()

        incident_description = input

        trace_id = trace_id or str(uuid.uuid4())

        print(incident_description)
        logger.info(f"> trace_id: {trace_id}")

        # Setup trajectory logger if enabled
        traj_logger: TrajectoryLogger | None = None
        if self._trajectory_dir:
            traj_logger = TrajectoryLogger(
                output_dir=self._trajectory_dir,
                run_id=trace_id,
                filename=trajectory_filename,
            )
            traj_logger.log(
                "run_start",
                {
                    "trace_id": trace_id,
                    "model": self.config.model.model_provider.model,
                    "prompt_path": self.config.agent.prompt_path,
                    "incident_description": incident_description[:500],
                },
            )

        # Expose to _handle_usage (called from inside graph nodes)
        self._current_traj_logger = traj_logger
        self._current_trace_id = trace_id
        self._usage_totals = self._empty_totals()

        # Reconstruct system prompt and user message for trajectory & dashboard
        prompt_path = self.config.agent.prompt_path or "agents/langgraph/rca.yaml"
        prompts = PromptManager.get_prompts(prompt_path)
        date_str = datetime.now().strftime("%a %b %-d, %Y")
        system_prompt_text = prompts["RCA_ANALYSIS_SP"].format(date=date_str)
        user_message_text = prompts["RCA_ANALYSIS_UP"].format(
            incident_description=incident_description
        )

        if traj_logger:
            traj_logger.log("llm_start", {
                "messages": [
                    {"type": "system", "content": system_prompt_text[:2000]},
                    {"type": "human", "content": user_message_text[:2000]},
                ],
            })

        initial_state = {
            "messages": [],
            "incident_description": incident_description,
            "tool_call_iterations": 0,
            "rca_findings": "",
            "raw_notes": [],
        }

        rca_findings = ""
        all_messages: list = []

        try:
            run_config = {"recursion_limit": 3000, "metadata": {"trace_id": trace_id}}
            assert self._rca_agent is not None, "RCA agent not built"

            async for event in self._rca_agent.astream(initial_state, config=run_config, stream_mode="updates"):
                # event: {node_name: node_output_dict}
                for node_name, node_output in event.items():
                    if traj_logger:
                        self._log_node_event(traj_logger, node_name, node_output)

                    # Accumulate messages from node outputs
                    if "messages" in node_output:
                        all_messages.extend(node_output["messages"])

                    # Capture rca_findings from compress node
                    if "rca_findings" in node_output:
                        rca_findings = node_output["rca_findings"]

        except Exception as e:
            logger.error(f"Error running RCA agent: {e}")
            partial_summary = dict(self._usage_totals)
            partial_summary["by_node"] = {k: dict(v) for k, v in partial_summary["by_node"].items()}
            if traj_logger:
                traj_logger.log("token_usage_summary", {
                    "trace_id": trace_id, "partial": True, **partial_summary,
                })
                traj_logger.log("error", {"error": str(e), "type": type(e).__name__})
                traj_logger.close()
            if self._on_usage is not None:
                try:
                    self._on_usage({
                        "summary": True, "partial": True,
                        "trace_id": trace_id, **partial_summary,
                    })
                except Exception as cb_err:
                    logger.warning(f"on_usage summary listener failed: {cb_err}")
            self._current_traj_logger = None
            self._current_trace_id = None
            raise

        final_output = self._validate_causal_graph_json(rca_findings)

        # Complete the message list: prepend user message, append compress output
        all_messages = [HumanMessage(content=user_message_text)] + all_messages
        if rca_findings:
            all_messages.append(AIMessage(content=rca_findings))

        # Build trajectory using SDK schema (system prompt set separately)
        agent_name = self.config.agent.name or "RCA-Agent"
        agent_traj = TrajectoryConverter.from_langchain_messages(
            all_messages, agent_name=agent_name, system_prompt=system_prompt_text
        )
        trajectory = Trajectory(agent_trajectories=[agent_traj])

        usage_summary = dict(self._usage_totals)
        usage_summary["by_node"] = {k: dict(v) for k, v in usage_summary["by_node"].items()}
        summary_event = {"trace_id": trace_id, **usage_summary}

        traj_file: str | None = None
        if traj_logger:
            traj_logger.log("token_usage_summary", summary_event)
            traj_logger.log_result(final_output, trace_id)
            traj_file = str(traj_logger.file_path)
            traj_logger.close()

        if self._on_usage is not None:
            try:
                self._on_usage({"summary": True, **summary_event})
            except Exception as e:
                logger.warning(f"on_usage summary listener failed: {e}")

        self._current_traj_logger = None
        self._current_trace_id = None

        return AgentResult(
            task=incident_description,
            trace_id=trace_id,
            final_output=final_output,
            trajectory=trajectory,
            metadata={"trajectory_file": traj_file, "token_usage": usage_summary},
        )

    @staticmethod
    def _empty_totals() -> dict[str, Any]:
        return {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cache_read": 0,
            "cache_creation": 0,
            "reasoning": 0,
            "by_node": {},
        }

    def _handle_usage(self, node: str, attempt: int, usage: dict[str, Any]) -> None:
        """Bridge per-API-call token usage to trajectory + external listener.

        Called from the graph after every successful ``ainvoke``. Captures
        every attempt (including ``compress_rca_findings`` repair turns),
        which would otherwise be invisible. Also accumulates into
        ``self._usage_totals`` for the end-of-run summary.
        """
        itd = usage.get("input_token_details") or {}
        otd = usage.get("output_token_details") or {}
        event = {
            "trace_id": self._current_trace_id,
            "node": node,
            "attempt": attempt,
            "input_tokens": usage.get("input_tokens", 0) or 0,
            "output_tokens": usage.get("output_tokens", 0) or 0,
            "total_tokens": usage.get("total_tokens", 0) or 0,
            "cache_read": itd.get("cache_read", 0) or 0,
            "cache_creation": itd.get("cache_creation", 0) or 0,
            "reasoning": otd.get("reasoning", 0) or 0,
        }

        totals = self._usage_totals
        totals["calls"] += 1
        for k in ("input_tokens", "output_tokens", "total_tokens",
                  "cache_read", "cache_creation", "reasoning"):
            totals[k] += event[k]
        per_node = totals["by_node"].setdefault(
            node,
            {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0,
             "cache_read": 0, "cache_creation": 0, "reasoning": 0},
        )
        per_node["calls"] += 1
        for k in ("input_tokens", "output_tokens", "total_tokens",
                  "cache_read", "cache_creation", "reasoning"):
            per_node[k] += event[k]

        traj = self._current_traj_logger
        if traj is not None:
            try:
                traj.log("token_usage", event)
            except Exception as e:
                logger.warning(f"failed to log token_usage to trajectory: {e}")
        if self._on_usage is not None:
            try:
                self._on_usage(event)
            except Exception as e:
                logger.warning(f"on_usage listener failed: {e}")

    @staticmethod
    def _log_node_event(traj_logger: TrajectoryLogger, node_name: str, node_output: dict) -> None:
        """Log graph node events in the format the dashboard frontend expects.

        Uses ``event_type`` names: ``llm_end``, ``tool_call``, ``tool_result``.
        """
        from langchain_core.messages import ToolMessage

        messages = node_output.get("messages", [])

        if node_name == "llm_call":
            for msg in messages:
                if isinstance(msg, AIMessage):
                    data: dict[str, Any] = {
                        "content": msg.content[:2000] if isinstance(msg.content, str) else "",
                    }
                    if hasattr(msg, "usage_metadata") and msg.usage_metadata:
                        data["usage"] = dict(msg.usage_metadata)
                    if msg.tool_calls:
                        data["tool_calls"] = [
                            {"name": tc.get("name", ""), "args": tc.get("args", {})}
                            for tc in msg.tool_calls
                        ]
                        traj_logger.log("llm_end", data)
                        for tc in msg.tool_calls:
                            traj_logger.log("tool_call", {
                                "tool_name": tc.get("name", ""),
                                "args": tc.get("args", {}),
                                "tool_call_id": tc.get("id", ""),
                            })
                    else:
                        traj_logger.log("llm_end", data)

        elif node_name == "tool_node":
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    traj_logger.log("tool_result", {
                        "tool_name": getattr(msg, "name", ""),
                        "tool_call_id": msg.tool_call_id,
                        "result": str(msg.content)[:3000],
                    })

        elif node_name == "compress_rca_findings":
            findings = node_output.get("rca_findings", "")
            traj_logger.log("llm_end", {
                "content": findings[:5000],
                "node": "compress_rca_findings",
            })

    @staticmethod
    def _validate_causal_graph_json(causal_graph_output: str) -> str:
        """Final post-graph safety pass over the synthesis output.

        The graph's synthesis node already validates and (up to a budget)
        retries with corrective feedback before emitting ``rca_findings``,
        so by this point the envelope is normally already clean. This
        method exists to keep the contract that downstream callers always
        receive a schema-shaped JSON string, even if the synthesis node
        was bypassed (e.g. tests calling ``run`` with a stub graph).
        """
        from .output_validator import validate_rca_output

        outcome = validate_rca_output(causal_graph_output)
        if outcome.retry_warranted:
            logger.warning(f"RCA synthesis output failed validation: {outcome.errors}")
        return json.dumps(outcome.envelope, ensure_ascii=False)
