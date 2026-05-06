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
        from importlib.metadata import version

        return version("thinkdepthai")

    def model_name(self) -> str | None:
        if self._config_path:
            try:
                from ..config import load_agent_config

                config = load_agent_config(self._config_path)
                return config.model.model_provider.model
            except Exception:
                return None
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

        agent_config = load_agent_config(self._config_path)

        # Build trajectory dir: {base}/{exp_id}/{case_name}.jsonl
        traj_dir = self._trajectory_dir
        if self._exp_id:
            traj_dir = os.path.join(traj_dir, self._exp_id)

        traj_filename = self._case_name_from_data_dir(data_dir)

        # Emit running + trajectory path BEFORE the run starts so the
        # dashboard can stream events in real-time (not just after completion).
        expected_traj_path = os.path.abspath(os.path.join(traj_dir, f"{traj_filename}.jsonl"))
        if ctx:
            ctx.emit({"type": "running", "run_id": run_id})
            ctx.emit({"type": "trajectory_update", "path": expected_traj_path})

        on_usage = None
        if ctx is not None:
            def on_usage(usage_event: dict[str, Any]) -> None:
                ctx.emit({"type": "token_usage", "run_id": run_id, **usage_event})

        agent = LanggraphRCAAgent(
            config=agent_config,
            trajectory_dir=traj_dir,
            on_usage=on_usage,
        )

        async with agent:
            result = await agent.run(
                incident,
                data_dir=data_dir,
                trace_id=run_id,
                trajectory_filename=traj_filename,
            )

        traj_file = result.metadata.get("trajectory_file") or expected_traj_path
        token_usage = result.metadata.get("token_usage")
        return AgentResult(
            response=result.final_output or "",
            trajectory=result.trajectory if isinstance(result.trajectory, Trajectory) else None,
            metadata={
                "run_id": run_id,
                "trajectory_file": traj_file,
                "token_usage": token_usage,
            },
        )

    @staticmethod
    def _case_name_from_data_dir(data_dir: str) -> str:

        from pathlib import Path

        parts = Path(data_dir).parts
        for part in reversed(parts):
            if part not in ("converted", "data", ".", "/"):
                return part
        return "unknown"
