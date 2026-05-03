#!/usr/bin/env python3
"""Seed the llm_eval DB with the openrca2-lite-v3 cases.

The dataset published as `lincyaw/openrca2-lite-v3
<https://huggingface.co/datasets/lincyaw/openrca2-lite-v3>`_ ships as a
single root containing:

  * ``manifest.jsonl`` — one JSON per case (name, system, fault metadata, …).
  * ``cases/<name>/`` — per-case directory with ``injection.json``,
    ``causal_graph.json``, and the telemetry parquets.

Each case is inserted as a ``DatasetSample`` keyed by
``(dataset="RCABench", source=<name>)``. ``meta.source_data_dir`` points at
``<root>/cases/<name>`` so the v2 RCABenchProcesser reads the right files
without any global ``source_path`` override.

Usage::

    # Pull the dataset (~3.8 GB):
    hf download lincyaw/openrca2-lite-v3 --repo-type dataset \\
        --local-dir ./data/openrca2_lite_v3

    # Seed the eval DB:
    LLM_EVAL_DB_URL=sqlite:///./eval.db \\
        python scripts/seed_v3_db.py --root ./data/openrca2_lite_v3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)
except ImportError:
    pass


DEFAULT_ROOT = os.environ.get("OPENRCA2_LITE_V3_ROOT")
DEFAULT_TAG = "openrca2-lite-v3"
DEFAULT_DATASET = "RCABench"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--root",
        default=DEFAULT_ROOT,
        help=(
            "Path to the dataset root (must contain manifest.jsonl + cases/). "
            "Defaults to $OPENRCA2_LITE_V3_ROOT if set."
        ),
    )
    p.add_argument("--db-url", default=None, help="Overrides LLM_EVAL_DB_URL.")
    p.add_argument("--tag", default=DEFAULT_TAG, help=f"Tag for inserted rows (default: {DEFAULT_TAG})")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help=f"Dataset field (default: {DEFAULT_DATASET})")
    p.add_argument("--max-cases", type=int, default=None, help="Insert only the first N cases.")
    p.add_argument("--cases-from", default=None, help="File with one case name per line; only these are seeded.")
    p.add_argument(
        "--merge-tags",
        action="store_true",
        help="If a row exists, only merge --tag into its tags array (don't overwrite).",
    )
    p.add_argument("--dry-run", action="store_true", help="Build rows without writing.")
    return p.parse_args()


def extract_rc_services(injection: dict[str, Any]) -> list[str]:
    """Pull the GT root-cause service set from injection.json (engine_config[*].app)."""
    services: list[str] = []
    seen: set[str] = set()
    for leaf in injection.get("engine_config") or []:
        if isinstance(leaf, dict):
            app = leaf.get("app")
            if app and app not in seen:
                seen.add(app)
                services.append(str(app))
    if services:
        return services
    # Fallback for old-format ground_truth (single dict OR list-of-dicts).
    gt_raw = injection.get("ground_truth") or {}
    gt_list = gt_raw if isinstance(gt_raw, list) else [gt_raw]
    for gt in gt_list:
        if not isinstance(gt, dict):
            continue
        for svc in gt.get("service") or []:
            if svc and svc not in seen:
                seen.add(svc)
                services.append(str(svc))
    return services


def detect_system(injection: dict[str, Any], case_name: str) -> str:
    """Resolve ``ts`` / ``hs`` / ``otel-demo`` from the injection or case name.

    New-format injection.json carries ``system_type`` per engine_config leaf;
    old-format ts cases carry no system identifier and fall back to the
    ``ts<digit>-`` case-name prefix.
    """
    for leaf in injection.get("engine_config") or []:
        if isinstance(leaf, dict) and leaf.get("system_type"):
            return str(leaf["system_type"])
    if injection.get("system_type"):
        return str(injection["system_type"])
    if case_name.startswith("ts") and len(case_name) >= 4 and case_name[2].isdigit():
        return "ts"
    return str(injection.get("category") or "")


def build_row(case_name: str, case_dir: Path, dataset: str, tag: str, idx: int) -> dict[str, Any] | None:
    injection_path = case_dir / "injection.json"
    causal_path = case_dir / "causal_graph.json"
    if not injection_path.exists() or not causal_path.exists():
        print(
            f"  warn: skip {case_name}: missing injection.json or causal_graph.json under {case_dir}",
            file=sys.stderr,
        )
        return None
    injection = json.loads(injection_path.read_text())
    rc_services = extract_rc_services(injection)
    if not rc_services:
        print(f"  warn: skip {case_name}: no GT services in injection.json", file=sys.stderr)
        return None

    fault_type = injection.get("fault_type", "")
    system = detect_system(injection, case_name)

    meta = {
        "source_data_dir": str(case_dir.resolve()),
        "system": system,
        "fault_type": fault_type,
        "rc_services": rc_services,
    }
    return {
        "dataset": dataset,
        "index": idx,
        "source": case_name,
        "source_index": idx,
        "question": "",
        "answer": ",".join(rc_services),
        "topic": None,
        "level": 0,
        "file_name": case_name,
        "meta": meta,
        "tags": [tag],
    }


def discover_cases(root: Path) -> list[tuple[str, Path]]:
    manifest = root / "manifest.jsonl"
    cases_dir = root / "cases"
    if not manifest.is_file():
        sys.exit(f"manifest.jsonl missing under {root}")
    if not cases_dir.is_dir():
        sys.exit(f"cases/ subdir missing under {root}")
    pairs: list[tuple[str, Path]] = []
    with manifest.open() as f:
        for line in f:
            row = json.loads(line)
            name = str(row["name"])
            pairs.append((name, cases_dir / name))
    print(f"Loaded {len(pairs)} cases from {manifest}")
    return pairs


def main() -> int:
    args = parse_args()
    if not args.root:
        sys.exit(
            "Pass --root <dataset_dir> (or set OPENRCA2_LITE_V3_ROOT). "
            "The dir must contain manifest.jsonl + cases/."
        )
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"--root {root} is not a directory")

    case_pairs = discover_cases(root)
    if args.cases_from:
        whitelist = {ln.strip() for ln in Path(args.cases_from).read_text().splitlines() if ln.strip()}
        case_pairs = [pair for pair in case_pairs if pair[0] in whitelist]
        print(f"Filtered to {len(case_pairs)} cases via {args.cases_from}")
    if args.max_cases is not None:
        case_pairs = case_pairs[: args.max_cases]

    rows: list[dict[str, Any]] = []
    for i, (name, case_dir) in enumerate(case_pairs, start=1):
        row = build_row(name, case_dir, args.dataset, args.tag, idx=i)
        if row is not None:
            rows.append(row)
    print(f"Built {len(rows)} valid rows (dataset={args.dataset}, tag={args.tag})")

    if args.dry_run:
        for row in rows[:3]:
            print(json.dumps({**row, "meta": {**row["meta"], "source_data_dir": "..."}}, indent=2))
        return 0

    if args.db_url:
        os.environ["LLM_EVAL_DB_URL"] = args.db_url
    if not os.environ.get("LLM_EVAL_DB_URL") and not os.environ.get("UTU_DB_URL"):
        sys.exit("Either set LLM_EVAL_DB_URL env var or pass --db-url.")

    from sqlmodel import select

    from rcabench_platform.v3.sdk.llm_eval.db import DatasetSample
    from rcabench_platform.v3.sdk.llm_eval.utils import SQLModelUtils

    inserted = 0
    updated = 0
    skipped = 0
    with SQLModelUtils.create_session() as session:
        existing = session.exec(select(DatasetSample).where(DatasetSample.dataset == args.dataset)).all()
        existing_by_key = {(r.dataset, r.source): r for r in existing}
        next_index = (max((r.index or 0) for r in existing), 0)[0] + 1 if existing else 1

        for row in rows:
            key = (row["dataset"], row["source"])
            if key in existing_by_key:
                cur = existing_by_key[key]
                if args.merge_tags:
                    cur_tags = list(cur.tags or [])
                    changed = False
                    for t in row["tags"]:
                        if t not in cur_tags:
                            cur_tags.append(t)
                            changed = True
                    if changed:
                        cur.tags = cur_tags
                        session.add(cur)
                        updated += 1
                    else:
                        skipped += 1
                else:
                    cur.tags = row["tags"]
                    cur.meta = row["meta"]
                    cur.answer = row["answer"]
                    cur.question = row["question"]
                    cur.topic = row["topic"]
                    cur.file_name = row["file_name"]
                    session.add(cur)
                    updated += 1
            else:
                row_for_insert = dict(row)
                row_for_insert["index"] = next_index
                next_index += 1
                session.add(DatasetSample(**row_for_insert))
                inserted += 1

        session.commit()

    print(f"DB seed complete: inserted={inserted}, updated={updated}, skipped={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
