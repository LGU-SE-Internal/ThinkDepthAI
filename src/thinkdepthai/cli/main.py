"""ThinkDepthAI CLI."""

import typer

app = typer.Typer(name="thinkdepthai", help="ThinkDepthAI RCA Agent CLI")


@app.command()
def run(
    data_dir: str = typer.Argument(..., help="Path to the data directory with parquet files"),
    incident: str = typer.Argument(..., help="Incident description"),
    config: str = typer.Option(None, "--config", "-c", help="Agent config path"),
    model_config: str = typer.Option(None, "--model-config", "-m", help="Model config path"),
):
    """Run a single RCA investigation."""
    import asyncio

    from .run import run_investigation

    asyncio.run(run_investigation(data_dir=data_dir, incident=incident, config_path=config))


@app.command()
def eval(
    eval_config: str = typer.Argument(..., help="Path to eval config YAML"),
    agent_config: str = typer.Option(None, "--agent-config", "-a", help="Agent config path"),
    exp_id: str = typer.Option(None, "--exp-id", help="Override experiment ID"),
    judge_only: bool = typer.Option(False, "--judge-only", help="Only run judge + stat"),
    stat_only: bool = typer.Option(False, "--stat-only", help="Only run stat"),
    max_steps: int = typer.Option(100, "--max-steps", help="Max tool call steps"),
):
    """Run batch evaluation using rcabench-platform SDK."""
    import asyncio

    from .eval import run_eval

    asyncio.run(
        run_eval(
            eval_config_path=eval_config,
            agent_config_path=agent_config,
            exp_id_override=exp_id,
            judge_only=judge_only,
            stat_only=stat_only,
        )
    )


if __name__ == "__main__":
    app()
