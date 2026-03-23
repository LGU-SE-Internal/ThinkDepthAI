"""Batch evaluation using rcabench-platform v3 SDK."""

from __future__ import annotations

import os

from rcabench_platform.v3.sdk.llm_eval.config import ConfigLoader, EvalConfig
from rcabench_platform.v3.sdk.llm_eval.eval.benchmarks.base_benchmark import BaseBenchmark, RolloutResult
from rcabench_platform.v3.sdk.llm_eval.eval.data import EvaluationSample
from rich.console import Console

from ..agent import LanggraphRCAAgent
from ..config import load_agent_config

console = Console()


async def run_eval(
    eval_config_path: str,
    agent_config_path: str | None = None,
    exp_id_override: str | None = None,
    judge_only: bool = False,
    stat_only: bool = False,
) -> None:
    """Run the full evaluation pipeline: preprocess → rollout → judge → stat."""

    eval_config: EvalConfig = ConfigLoader.load_eval_config(eval_config_path)

    if exp_id_override:
        eval_config.exp_id = exp_id_override

    # Load agent config
    agent_config = load_agent_config(agent_config_path)

    # Auto-populate model_name
    if not getattr(eval_config, "model_name", None):
        eval_config.model_name = agent_config.model.model_provider.model

    if eval_config.db_url:
        os.environ["LLM_EVAL_DB_URL"] = eval_config.db_url

    benchmark = BaseBenchmark(eval_config)

    if stat_only:
        console.print("[bold]Running stat only...[/]")
        await benchmark.stat()
        return

    if judge_only:
        console.print("[bold]Running judge + stat...[/]")
        await benchmark.judge()
        await benchmark.stat()
        return

    console.print(
        f"[bold]Eval:[/] exp_id=[cyan]{eval_config.exp_id}[/]  "
        f"concurrency=[cyan]{eval_config.concurrency}[/]"
    )

    console.print("[bold]Phase 1:[/] preprocess")
    benchmark.preprocess()

    console.print("[bold]Phase 2:[/] rollout")

    async def runner(sample: EvaluationSample) -> RolloutResult:
        sample_id = str(sample.id)
        incident = (sample.augmented_question or sample.raw_question or "").strip()
        meta = sample.meta if isinstance(sample.meta, dict) else {}
        data_dir: str = meta.get("path", "")

        if not data_dir or not incident:
            console.print(f"  [yellow]SKIP[/] sample id={sample_id}")
            return RolloutResult()

        console.print(f"  [blue]START[/] sample id={sample_id} idx={sample.dataset_index}")

        try:
            agent = LanggraphRCAAgent(config=agent_config)
            async with agent:
                result = await agent.run(incident)

            status = "[green]OK[/]" if result.final_output else "[red]EMPTY[/]"
            console.print(f"  {status} sample id={sample_id}")

            return RolloutResult(
                response=result.final_output or "",
                trajectory_json=result.trajectory.to_json(),
                trace_id=result.trace_id,
            )
        except Exception as e:
            console.print(f"  [red]FAIL[/] sample id={sample_id}: {e}")
            return RolloutResult()

    ok_count, fail_count = await benchmark.rollout(runner, max_samples=eval_config.max_samples)
    console.print(f"  [green]{ok_count} ok[/] / [red]{fail_count} failed[/]")

    console.print("[bold]Phase 3:[/] judge")
    await benchmark.judge()

    console.print("[bold]Phase 4:[/] stat")
    await benchmark.stat()
