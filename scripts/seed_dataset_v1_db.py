#!/usr/bin/env python3
"""Seed the llm_eval DB with the dataset_v1_500 cases.

Two on-disk layouts are supported. Pick whichever matches what you have:

- **HF snapshot (recommended for fresh installs)**: a single root with
  ``MANIFEST.json`` plus per-case directories that each contain
  ``injection.json``, ``causal_graph.json``, and the telemetry parquets. This
  is the layout published as `lincyaw/openrca2-v1-500
  <https://huggingface.co/datasets/lincyaw/openrca2-v1-500>`_. Pass
  ``--snapshot-root`` (or set ``DATASET_V1_500_ROOT`` in your env / .env).
- **Local pool layout**: cases live at ``<dataset_root>/cases/<name>`` as
  symlinks into ``<pool_root>/<name>/converted/`` (the on-disk format used
  during dataset construction). Pass both ``--dataset-root`` and
  ``--pool-root`` to use this mode.

Each case is inserted as a ``DatasetSample`` keyed by
``(dataset="RCABench", source=<name>)``. ``meta.source_data_dir`` points at
the per-case directory so the v2 RCABenchProcesser reads the right files
without any global ``source_path`` override.

Usage::

    # HF snapshot:
    hf download lincyaw/openrca2-v1-500 --repo-type dataset \\
        --local-dir ./data/openrca2_v1_500
    LLM_EVAL_DB_URL=sqlite:///./eval.db \\
        python scripts/seed_dataset_v1_db.py \\
            --snapshot-root ./data/openrca2_v1_500

    # Local pool layout:
    LLM_EVAL_DB_URL=sqlite:///./eval.db \\
        python scripts/seed_dataset_v1_db.py \\
            --dataset-root /path/to/dataset_v1_500_2026-05-02 \\
            --pool-root /path/to/_pool_v1_2026-05-02
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


DEFAULT_SNAPSHOT_ROOT = os.environ.get("DATASET_V1_500_ROOT")
DEFAULT_TAG = "dataset_v1_500"
DEFAULT_DATASET = "RCABench"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--snapshot-root",
        default=DEFAULT_SNAPSHOT_ROOT,
        help=(
            "Path to the HF snapshot root (must contain MANIFEST.json plus per-case dirs). "
            "Defaults to $DATASET_V1_500_ROOT if set."
        ),
    )
    p.add_argument("--dataset-root", default=None,
                   help="(Local pool layout) Path with cases/ subdir of symlinks.")
    p.add_argument("--pool-root", default=None,
                   help="(Local pool layout) Path containing <case>/converted/causal_graph.json.")
    p.add_argument("--db-url", default=None, help="Overrides LLM_EVAL_DB_URL.")
    p.add_argument("--tag", default=DEFAULT_TAG, help=f"Tag for inserted rows (default: {DEFAULT_TAG})")
    p.add_argument("--dataset", default=DEFAULT_DATASET, help=f"Dataset field (default: {DEFAULT_DATASET})")
    p.add_argument("--max-cases", type=int, default=None, help="Insert only the first N cases.")
    p.add_argument("--cases-from", default=None, help="File with one case name per line; only these are seeded.")
    p.add_argument("--merge-tags", action="store_true",
                   help="If a row exists, only merge --tag into its tags array (don't overwrite).")
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
    # Fallback for old-format ground_truth.
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


def build_row(case_name: str, case_dir: Path, dataset: str, tag: str, idx: int) -> dict[str, Any] | None:
    injection_path = case_dir / "injection.json"
    causal_path = case_dir / "causal_graph.json"
    if not injection_path.exists() or not causal_path.exists():
        print(f"  warn: skip {case_name}: missing injection.json or causal_graph.json under {case_dir}",
              file=sys.stderr)
        return None
    injection = json.loads(injection_path.read_text())
    rc_services = extract_rc_services(injection)
    if not rc_services:
        print(f"  warn: skip {case_name}: no GT services in injection.json", file=sys.stderr)
        return None

    fault_type = injection.get("fault_type", "")
    system = injection.get("system_type") or injection.get("category") or ""

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


def discover_cases(args: argparse.Namespace) -> list[tuple[str, Path]]:
    """Resolve (case_name, case_dir) pairs from whichever layout the user provided.

    Returns a list of (case_name, absolute_case_dir) tuples in deterministic order.
    """
    if args.snapshot_root:
        snap_root = Path(args.snapshot_root).expanduser().resolve()
        if not snap_root.is_dir():
            sys.exit(f"--snapshot-root {snap_root} is not a directory")
        manifest_path = snap_root / "MANIFEST.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            entries = manifest.get("cases") or []
            case_names = [str(e.get("name") or e.get("case")) for e in entries if (e.get("name") or e.get("case"))]
            if not case_names:
                sys.exit(f"MANIFEST.json at {manifest_path} has no cases")
            print(f"Loaded {len(case_names)} cases from MANIFEST.json")
        else:
            # Fallback: list directories that look like cases.
            case_names = sorted(
                p.name for p in snap_root.iterdir()
                if p.is_dir() and (p / "injection.json").is_file()
            )
            print(f"No MANIFEST.json at {manifest_path}; discovered {len(case_names)} case dirs by injection.json")
        return [(name, snap_root / name) for name in case_names]

    if not (args.dataset_root and args.pool_root):
        sys.exit(
            "No source layout given. Pass --snapshot-root <hf_dir> (or set "
            "DATASET_V1_500_ROOT in your env / .env) for the HF snapshot layout, "
            "or --dataset-root + --pool-root for the legacy local pool layout."
        )
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    pool_root = Path(args.pool_root).expanduser().resolve()
    cases_dir = dataset_root / "cases"
    if not cases_dir.is_dir():
        sys.exit(f"cases dir missing: {cases_dir}")
    if not pool_root.is_dir():
        sys.exit(f"pool root missing: {pool_root}")
    case_names = sorted(p.name for p in cases_dir.iterdir())
    print(f"Loaded {len(case_names)} cases from {cases_dir}")
    return [(name, pool_root / name / "converted") for name in case_names]


def main() -> int:
    args = parse_args()

    case_pairs = discover_cases(args)
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
        existing = session.exec(
            select(DatasetSample).where(DatasetSample.dataset == args.dataset)
        ).all()
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
