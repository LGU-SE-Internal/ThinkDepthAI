"""Single investigation run."""

from __future__ import annotations

import json

from rich.console import Console

from ..agent import LanggraphRCAAgent
from ..config import load_agent_config

console = Console()


async def run_investigation(
    data_dir: str,
    incident: str,
    config_path: str | None = None,
) -> str:
    """Run a single RCA investigation and print results."""
    config = load_agent_config(config_path)
    console.print(f"[bold]Running RCA investigation[/] with model: {config.model.model_provider.model}")

    agent = LanggraphRCAAgent(config=config)
    async with agent:
        result = await agent.run(incident)

    console.print("\n[bold green]RCA Results:[/]")
    try:
        parsed = json.loads(result.final_output)
        console.print_json(json.dumps(parsed, indent=2))
    except json.JSONDecodeError:
        console.print(result.final_output)

    console.print(f"\n[dim]trace_id: {result.trace_id}[/]")
    return result.final_output
