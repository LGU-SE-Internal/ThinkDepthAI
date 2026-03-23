"""LanggraphRCAAgent — standalone Root Cause Analysis agent."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any

from rcabench_platform.v3.sdk.llm_eval.trajectory import Trajectory

from .config import load_agent_config
from .config.schema import AgentConfig
from .converters import TrajectoryConverter
from .graph import build_rca_agent
from .llm import create_langchain_model
from .tools import think_tool
from .tools_lib import AsyncBaseToolkit, toolkit_to_langchain_tools
from .tools_lib.query_parquet_toolkit import QueryParquetFilesToolkit
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

    def __init__(self, *, config: AgentConfig | str | None = None, name: str | None = None):
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

    async def run(self, input: str | list, trace_id: str | None = None) -> AgentResult:
        """Run RCA analysis on an incident.

        Args:
            input: Incident description (string or list)
            trace_id: Optional trace ID for tracking

        Returns:
            AgentResult containing findings and trajectory
        """
        if not self._initialized:
            await self.build()

        if isinstance(input, list):
            incident_description = json.dumps(input, ensure_ascii=False, indent=2)
        else:
            incident_description = input

        trace_id = trace_id or str(uuid.uuid4())
        logger.info(f"> trace_id: {trace_id}")

        initial_state = {
            "messages": [],
            "incident_description": incident_description,
            "tool_call_iterations": 0,
            "rca_findings": "",
            "raw_notes": [],
        }

        try:
            config = {"recursion_limit": 3000, "metadata": {"trace_id": trace_id}}
            assert self._rca_agent is not None, "RCA agent not built"
            result = await self._rca_agent.ainvoke(initial_state, config=config)
        except Exception as e:
            logger.error(f"Error running RCA agent: {e}")
            raise

        rca_findings = result.get("rca_findings", "")
        final_output = self._validate_causal_graph_json(rca_findings)

        # Build trajectory using SDK schema
        messages = result.get("messages", [])
        agent_name = self.config.agent.name or "RCA-Agent"
        agent_traj = TrajectoryConverter.from_langchain_messages(messages, agent_name=agent_name)
        trajectory = Trajectory(agent_trajectories=[agent_traj])

        return AgentResult(
            task=incident_description,
            trace_id=trace_id,
            final_output=final_output,
            trajectory=trajectory,
        )

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
        """Validate and clean CausalGraph JSON output."""
        if not causal_graph_output or not causal_graph_output.strip():
            logger.warning("CausalGraph output is empty")
            return json.dumps(
                {"nodes": [], "edges": [], "root_causes": [], "component_to_service": {}, "parse_error": "Empty output"},
                ensure_ascii=False,
            )

        extracted_json = LanggraphRCAAgent._extract_json_from_text(causal_graph_output)

        if not extracted_json:
            logger.warning("Could not extract JSON from CausalGraph output")
            return json.dumps(
                {
                    "nodes": [], "edges": [], "root_causes": [], "component_to_service": {},
                    "parse_error": "Could not find JSON in output",
                    "raw_output": causal_graph_output[:500],
                },
                ensure_ascii=False,
            )

        try:
            parsed = json.loads(extracted_json)
            if not isinstance(parsed, dict):
                raise ValueError("Output is not a JSON object")

            required_fields = ["nodes", "edges", "root_causes"]
            missing_fields = [f for f in required_fields if f not in parsed]

            if missing_fields:
                logger.warning(f"CausalGraph missing fields: {missing_fields}")
                return json.dumps(
                    {
                        "nodes": parsed.get("nodes", []),
                        "edges": parsed.get("edges", []),
                        "root_causes": parsed.get("root_causes", []),
                        "component_to_service": parsed.get("component_to_service", {}),
                        "parse_warning": f"Missing: {missing_fields}",
                    },
                    ensure_ascii=False,
                )

            return json.dumps(parsed, ensure_ascii=False)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse CausalGraph JSON: {e}")
            return json.dumps(
                {
                    "nodes": [], "edges": [], "root_causes": [], "component_to_service": {},
                    "parse_error": f"JSON decode error: {e}",
                    "raw_output": causal_graph_output[:500],
                },
                ensure_ascii=False,
            )
        except Exception as e:
            logger.error(f"Unexpected error validating CausalGraph: {e}")
            return json.dumps(
                {
                    "nodes": [], "edges": [], "root_causes": [], "component_to_service": {},
                    "validation_error": str(e),
                    "raw_output": causal_graph_output[:500],
                },
                ensure_ascii=False,
            )
