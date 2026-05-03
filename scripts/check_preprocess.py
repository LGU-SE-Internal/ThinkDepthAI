#!/usr/bin/env python3
"""Run only the preprocess phase of the eval pipeline and report per-case status.

Useful for verifying that a new dataset (or a heterogeneous mix of old/new
schemas like OpenRCA-2.0-Lite v1) parses cleanly through ``RCABenchProcesser``
before paying for any LLM rollout.

Reads the same eval config as ``rca llm-eval run``; selects samples by the
config's tag filter, attempts ``preprocess_one`` on each, and prints a summary.

Usage::

    LLM_EVAL_DB_URL=sqlite:///./eval.db \
        uv run python scripts/check_preprocess.py config/eval/openrca2_lite.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)
except ImportError:
    pass


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("config", type=Path, help="Path to eval YAML config.")
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    if not args.config.exists():
        sys.exit(f"Config not found: {args.config}")

    from rcabench_platform.v3.sdk.llm_eval.config import ConfigLoader
    from rcabench_platform.v3.sdk.llm_eval.eval import BaseBenchmark

    config = ConfigLoader.load_eval_config(path=str(args.config))

    if config.db_url:
        os.environ["LLM_EVAL_DB_URL"] = config.db_url

    benchmark = BaseBenchmark(config)

    samples = benchmark.dataset.get_samples(
        stage="init",
        agent_type=None,
        model_name=None,
        tags=config.data.tags if config.data else None,
        exclude_trajectories=True,
    )
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    print(f"Found {len(samples)} samples to preprocess.")

    ok: list[str] = []
    fail: list[tuple[str, str]] = []
    for s in samples:
        label = f"{s.source} (dataset={s.dataset})"
        try:
            benchmark.preprocess_one(s)
            ok.append(label)
        except Exception as exc:
            fail.append((label, f"{type(exc).__name__}: {exc}"))
            traceback.print_exc()

    print()
    print(f"=== preprocess summary: ok={len(ok)} fail={len(fail)} ===")
    for label, reason in fail:
        print(f"  FAIL  {label}  {reason}")
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main())
