"""ThinkDepthAI agent adapter for the llm_eval SDK.

Registered as an entry-point so the platform discovers it automatically:

    [project.entry-points."llm_eval.agents"]
    thinkdepthai = "thinkdepthai.agents.eval_agent:ThinkDepthAgent"

Usage::

    rca llm-eval run config.yaml -a thinkdepthai \\
        --ak config_path=config/agent/base.yaml \\
        --source-path /mnt/jfs/rcabench_dataset
"""

from __future__ import annotations

from typing import Any

from rcabench_platform.v3.sdk.llm_eval.agents.base_agent import (
    AgentResult,
    BaseAgent,
    RunContext,
)
from rcabench_platform.v3.sdk.llm_eval.trajectory.schema import Trajectory


class ThinkDepthAgent(BaseAgent):
    """Wraps :class:`LanggraphRCAAgent` to satisfy the SDK's BaseAgent contract."""

    def __init__(
        self,
        config_path: str | None = None,
        trajectory_dir: str = "./trajectories",
        **kwargs: Any,
    ) -> None:
        self._config_path = config_path
        self._trajectory_dir = trajectory_dir

    @staticmethod
    def name() -> str:
        return "thinkdepthai"

    def version(self) -> str | None:
        try:
            from importlib.metadata import version

            return version("thinkdepthai")
        except Exception:
            return None

    async def run(
        self,
        incident: str,
        data_dir: str,
        **kwargs: Any,
    ) -> AgentResult:
        from ..agent import LanggraphRCAAgent
        from ..config import load_agent_config

        ctx: RunContext | None = kwargs.get("ctx")

        agent_config = load_agent_config(self._config_path)
        agent = LanggraphRCAAgent(
            config=agent_config,
            trajectory_dir=self._trajectory_dir,
        )

        async with agent:
            result = await agent.run(incident, trace_id=None)

        run_id = result.trace_id

        # Emit events for the framework
        if ctx:
            ctx.emit({"type": "running", "run_id": run_id})
            traj_file = result.metadata.get("trajectory_file")
            if traj_file:
                ctx.emit({"type": "trajectory_update", "path": traj_file})

        return AgentResult(
            response=result.final_output or "",
            trajectory=result.trajectory if isinstance(result.trajectory, Trajectory) else None,
            metadata={
                "run_id": run_id,
                "trajectory_file": result.metadata.get("trajectory_file"),
            },
        )
