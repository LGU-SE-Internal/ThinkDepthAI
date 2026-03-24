"""ThinkDepthAI CLI."""

import typer
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer(name="thinkdepthai", help="ThinkDepthAI RCA Agent CLI")


@app.command()
def run(
    data_dir: str = typer.Argument(..., help="Path to the data directory with parquet files"),
    incident: str = typer.Argument(..., help="Incident description"),
    config: str = typer.Option(None, "--config", "-c", help="Agent config path"),
    trajectory_dir: str = typer.Option(None, "--trajectory-dir", "-t", help="Save JSONL trajectory to this directory"),
):
    """Run a single RCA investigation."""
    import asyncio

    from .run import run_investigation

    asyncio.run(
        run_investigation(data_dir=data_dir, incident=incident, config_path=config, trajectory_dir=trajectory_dir)
    )


@app.command(name="eval")
def eval_cmd(
    eval_config: str = typer.Argument(..., help="Path to eval config YAML"),
    agent_config: str = typer.Option(None, "--agent-config", "-a", help="Agent config path"),
    exp_id: str = typer.Option(None, "--exp-id", help="Override experiment ID"),
    judge_only: bool = typer.Option(False, "--judge-only", help="Only run judge + stat"),
    stat_only: bool = typer.Option(False, "--stat-only", help="Only run stat"),
    trajectory_dir: str = typer.Option(
        "./trajectories", "--trajectory-dir", "-t", help="Directory for JSONL trajectory logs"
    ),
    no_trajectory: bool = typer.Option(False, "--no-trajectory", help="Disable trajectory JSONL logging"),
    source_path: str = typer.Option(None, "--source-path", help="Dataset root path"),
    dashboard: bool = typer.Option(False, "--dashboard", help="Launch real-time eval dashboard"),
    dashboard_port: int = typer.Option(8765, "--dashboard-port", help="Dashboard server port"),
    dashboard_host: str = typer.Option("0.0.0.0", "--dashboard-host", help="Dashboard server host"),
):
    """Run batch evaluation using rcabench-platform SDK."""
    import asyncio

    from .eval import run_eval

    traj = None if no_trajectory else trajectory_dir
    asyncio.run(
        run_eval(
            eval_config_path=eval_config,
            agent_config_path=agent_config,
            exp_id_override=exp_id,
            judge_only=judge_only,
            stat_only=stat_only,
            trajectory_dir=traj,
            source_path=source_path,
            dashboard=dashboard,
            dashboard_port=dashboard_port,
            dashboard_host=dashboard_host,
        )
    )


if __name__ == "__main__":
    app()
