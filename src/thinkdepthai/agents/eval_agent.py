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
        exp_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        self._config_path = config_path
        self._trajectory_dir = trajectory_dir
        self._exp_id = exp_id

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
        import os
        import uuid

        from ..agent import LanggraphRCAAgent
        from ..config import load_agent_config

        ctx: RunContext | None = kwargs.get("ctx")

        run_id = str(uuid.uuid4())

        # Emit "running" immediately so the dashboard shows real-time status
        if ctx:
            ctx.emit({"type": "running", "run_id": run_id})

        # Build trajectory dir: {base}/{exp_id}/
        traj_dir = self._trajectory_dir
        if self._exp_id:
            traj_dir = os.path.join(traj_dir, self._exp_id)

        # Derive a human-readable filename from data_dir.
        # data_dir is like ".../sock-shop_case42/converted" → "sock-shop_case42"
        traj_filename = self._case_name_from_data_dir(data_dir)

        agent_config = load_agent_config(self._config_path)
        agent = LanggraphRCAAgent(
            config=agent_config,
            trajectory_dir=traj_dir,
        )

        async with agent:
            result = await agent.run(
                incident,
                data_dir=data_dir,
                trace_id=run_id,
                trajectory_filename=traj_filename,
            )

        # Emit trajectory path so the dashboard can stream events
        traj_file = result.metadata.get("trajectory_file")
        if ctx and traj_file:
            ctx.emit({"type": "trajectory_update", "path": traj_file})

        return AgentResult(
            response=result.final_output or "",
            trajectory=result.trajectory if isinstance(result.trajectory, Trajectory) else None,
            metadata={
                "run_id": run_id,
                "trajectory_file": traj_file,
            },
        )

    @staticmethod
    def _case_name_from_data_dir(data_dir: str) -> str:
        """Extract a meaningful case name from the data directory path.

        Examples:
            ``/mnt/jfs/rcabench_dataset/sock-shop_case42/converted``
              → ``sock-shop_case42``
            ``/data/train-ticket_fault3``
              → ``train-ticket_fault3``
        """
        from pathlib import Path

        parts = Path(data_dir).parts
        # Walk backwards, skip generic segments like "converted"
        for part in reversed(parts):
            if part not in ("converted", "data", ".", "/"):
                return part
        return "unknown"
