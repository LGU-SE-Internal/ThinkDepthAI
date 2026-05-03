#!/usr/bin/env python3
"""Seed the llm_eval DB with the dataset_v1_500_2026-05-02 cases.

This curated 500-case set lives at ``<dataset_root>/cases/<name>``, where each
``<name>`` is a symlink to a bare source case dir (parquets + injection.json).
The companion pool at ``<pool_root>/<name>/converted/`` adds the regenerated
``causal_graph.json`` (and re-symlinks the parquets), which is what the v2
RCABenchProcesser needs to compute alarm endpoints + graph metrics.

Each case is inserted as a ``DatasetSample`` keyed by
``(dataset="RCABench", source=<name>)``. ``meta.source_data_dir`` is set to
``<pool_root>/<name>/converted/`` so the processer reads the correct dir.

Usage::

    LLM_EVAL_DB_URL=sqlite:///./eval.db \
        python scripts/seed_dataset_v1_db.py \
            --dataset-root /home/ddq/AoyangSpace/dataset/dataset_v1_500_2026-05-02 \
            --pool-root /home/ddq/AoyangSpace/dataset/_pool_v1_2026-05-02
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


DEFAULT_DATASET_ROOT = "/home/ddq/AoyangSpace/dataset/dataset_v1_500_2026-05-02"
DEFAULT_POOL_ROOT = "/home/ddq/AoyangSpace/dataset/_pool_v1_2026-05-02"
DEFAULT_TAG = "dataset_v1_500"
DEFAULT_DATASET = "RCABench"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT,
                   help=f"Path with cases/ subdir (default: {DEFAULT_DATASET_ROOT})")
    p.add_argument("--pool-root", default=DEFAULT_POOL_ROOT,
                   help=f"Pool root containing <case>/converted/causal_graph.json (default: {DEFAULT_POOL_ROOT})")
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


def build_row(case_name: str, pool_root: Path, dataset: str, tag: str, idx: int) -> dict[str, Any] | None:
    case_dir = pool_root / case_name / "converted"
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


def main() -> int:
    args = parse_args()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    pool_root = Path(args.pool_root).expanduser().resolve()
    cases_dir = dataset_root / "cases"
    if not cases_dir.is_dir():
        sys.exit(f"cases dir missing: {cases_dir}")
    if not pool_root.is_dir():
        sys.exit(f"pool root missing: {pool_root}")

    case_names = sorted(p.name for p in cases_dir.iterdir())
    if args.cases_from:
        whitelist = {ln.strip() for ln in Path(args.cases_from).read_text().splitlines() if ln.strip()}
        case_names = [n for n in case_names if n in whitelist]
    if args.max_cases is not None:
        case_names = case_names[: args.max_cases]
    print(f"Loaded {len(case_names)} cases from {cases_dir}")

    rows: list[dict[str, Any]] = []
    for i, name in enumerate(case_names, start=1):
        row = build_row(name, pool_root, args.dataset, args.tag, idx=i)
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
