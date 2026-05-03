"""LanggraphRCAAgent — standalone Root Cause Analysis agent."""

from __future__ import annotations

import json
import uuid
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
        self, *, config: AgentConfig | str | None = None, name: str | None = None, trajectory_dir: str | None = None
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
            langchain_model, model_with_tools, tools=all_tools, prompt_path=prompt_path, retry_config=retry_config
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
            if traj_logger:
                traj_logger.log("error", {"error": str(e), "type": type(e).__name__})
                traj_logger.close()
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

        traj_file: str | None = None
        if traj_logger:
            traj_logger.log_result(final_output, trace_id)
            traj_file = str(traj_logger.file_path)
            traj_logger.close()

        return AgentResult(
            task=incident_description,
            trace_id=trace_id,
            final_output=final_output,
            trajectory=trajectory,
            metadata={"trajectory_file": traj_file},
        )

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
    def _extract_json_from_text(text: str) -> str | None:
        """Extract JSON object from text that may contain markdown."""
        import re

        if not text or not text.strip():
            return None

        text = text.strip()

        # Try markdown code blocks
        for pattern in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                if extracted.startswith("{"):
                    return extracted

        # Find balanced braces
        first_brace = text.find("{")
        if first_brace == -1:
            return None

        depth = 0
        in_string = False
        escape_next = False
        end_pos = -1

        for i in range(first_brace, len(text)):
            char = text[i]
            if escape_next:
                escape_next = False
                continue
            if char == "\\":
                escape_next = True
                continue
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end_pos = i
                    break

        if end_pos != -1:
            return text[first_brace : end_pos + 1]
        return None

    @staticmethod
    def _validate_causal_graph_json(causal_graph_output: str) -> str:
        """Validate the v2 RCA output schema produced by the synthesis node.

        Required shape:
            {"root_causes": [...], "propagation": [...]}
        with each `root_causes[i]` carrying `service`, `fault_kind`, and at
        least one `evidence` item. Failures are returned as a JSON envelope
        carrying `parse_error`/`parse_warning` plus a truncated `raw_output`
        so the downstream eval can record what the model emitted.
        """
        empty_envelope = {"root_causes": [], "propagation": []}
        if not causal_graph_output or not causal_graph_output.strip():
            logger.warning("RCA output is empty")
            return json.dumps({**empty_envelope, "parse_error": "Empty output"}, ensure_ascii=False)

        extracted_json = LanggraphRCAAgent._extract_json_from_text(causal_graph_output)
        if not extracted_json:
            logger.warning("Could not extract JSON from RCA output")
            return json.dumps(
                {
                    **empty_envelope,
                    "parse_error": "Could not find JSON in output",
                    "raw_output": causal_graph_output[:500],
                },
                ensure_ascii=False,
            )

        try:
            parsed = json.loads(extracted_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse RCA JSON: {e}")
            return json.dumps(
                {
                    **empty_envelope,
                    "parse_error": f"JSON decode error: {e}",
                    "raw_output": causal_graph_output[:500],
                },
                ensure_ascii=False,
            )

        if not isinstance(parsed, dict):
            return json.dumps(
                {
                    **empty_envelope,
                    "parse_error": "Output is not a JSON object",
                    "raw_output": causal_graph_output[:500],
                },
                ensure_ascii=False,
            )

        warnings: list[str] = []
        root_causes = parsed.get("root_causes")
        if not isinstance(root_causes, list) or not root_causes:
            warnings.append("root_causes missing or empty")
            root_causes = []

        cleaned_root_causes: list[dict] = []
        for i, rc in enumerate(root_causes):
            if not isinstance(rc, dict):
                warnings.append(f"root_causes[{i}] is not an object; dropping")
                continue
            if not rc.get("service"):
                warnings.append(f"root_causes[{i}] missing `service`; dropping")
                continue
            if not rc.get("fault_kind"):
                warnings.append(f"root_causes[{i}] missing `fault_kind`; dropping")
                continue
            evidence = rc.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                warnings.append(f"root_causes[{i}] has no evidence; dropping")
                continue
            cleaned_root_causes.append(rc)

        propagation = parsed.get("propagation")
        if not isinstance(propagation, list):
            propagation = []

        result: dict = {"root_causes": cleaned_root_causes, "propagation": propagation}
        if warnings:
            result["parse_warning"] = "; ".join(warnings)
            logger.warning(f"RCA v2 schema warnings: {result['parse_warning']}")
        return json.dumps(result, ensure_ascii=False)
