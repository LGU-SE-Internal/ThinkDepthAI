#!/usr/bin/env python3
"""Seed the llm_eval DB with OpenRCA-2.0-Lite v1 cases.

The dataset lives on disk as a directory of per-case folders, each containing
``causal_graph.json``, ``injection.json``, and the telemetry parquet files.
A ``MANIFEST.json`` at the root lists all cases with their fault metadata.

This script reads the manifest and inserts one ``DatasetSample`` row per case
into the SQLite DB that ``rcabench-platform[llm-eval]`` reads from. The row is
keyed by ``(dataset="RCABench", source=<case_name>)``; tagging it with
``openrca2-lite-v1`` lets the eval config select this subset.

Usage::

    # Pull the dataset once:
    hf download lincyaw/openrca2-lite-v1 --repo-type dataset \
        --local-dir ./data/openrca2_lite_v1

    # Seed the DB:
    LLM_EVAL_DB_URL=sqlite:///./eval.db \
        python scripts/seed_lite_v1_db.py \
            --lite-root ./data/openrca2_lite_v1

If ``--db-url`` is given, it overrides ``LLM_EVAL_DB_URL``. The DB schema is
auto-created on first use. ``--lite-root`` defaults to ``$LITE_V1_ROOT`` (or
``./data/openrca2_lite_v1`` if unset) so a configured ``.env`` is enough.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Mirror rcabench-platform's CLI: pick up .env (search cwd → parents) so
# LLM_EVAL_DB_URL set there is respected without an explicit --db-url.
try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True), override=False)
except ImportError:
    pass

DEFAULT_LITE_ROOT = os.environ.get("LITE_V1_ROOT", "./data/openrca2_lite_v1")
DEFAULT_TAG = "openrca2-lite-v1"
DEFAULT_DATASET = "RCABench"


def _unprefix(name: str) -> str:
    """Strip the ``system__faulttype__`` prefix from a lite_v1 case name.

    lite_v1 directories use the prefixed form; the canonical RCABench DB rows
    use the bare case ID. Stripping aligns the two naming conventions.
    """
    parts = name.split("__", 2)
    return parts[-1] if len(parts) == 3 else name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--lite-root",
        default=DEFAULT_LITE_ROOT,
        help=f"Path to OpenRCA-2.0-Lite v1 root (must contain MANIFEST.json). Default: {DEFAULT_LITE_ROOT}",
    )
    p.add_argument(
        "--db-url",
        default=None,
        help="SQLAlchemy DB URL. Overrides LLM_EVAL_DB_URL env var if given.",
    )
    p.add_argument(
        "--tag",
        default=DEFAULT_TAG,
        help=f"Tag to attach to inserted rows (selects this subset in eval config). Default: {DEFAULT_TAG}",
    )
    p.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Dataset name field. Default: {DEFAULT_DATASET}",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Read manifest and report what would be inserted without touching the DB.",
    )
    p.add_argument(
        "--dry-run-db",
        action="store_true",
        help=(
            "Connect to the DB, run all merge logic against existing rows, then "
            "rollback instead of committing. Use this to preview prod-DB changes."
        ),
    )
    p.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Only insert the first N cases (useful for smoke tests).",
    )
    p.add_argument(
        "--cases-from",
        default=None,
        help="Path to a file with one case name per line; only those cases are seeded.",
    )
    p.add_argument(
        "--source-style",
        choices=["prefixed", "unprefixed"],
        default="unprefixed",
        help=(
            "How to set the DB ``source`` column: ``unprefixed`` strips the "
            "``system__faulttype__`` prefix and matches the prod RCABench naming "
            "convention; ``prefixed`` keeps the lite_v1 directory name verbatim."
        ),
    )
    p.add_argument(
        "--merge-tags",
        action="store_true",
        help=(
            "For rows that already exist (by dataset+source), merge --tag into the "
            "existing tags array instead of overwriting other fields. Use this when "
            "seeding into a shared/prod DB to avoid clobbering metadata written by "
            "other pipelines."
        ),
    )
    p.add_argument(
        "--refresh-answer",
        action="store_true",
        help=(
            "Even in --merge-tags mode, re-write the ``answer`` column from this "
            "machine's injection.json when the values differ. Use to backfill "
            "rows whose answer was set by an earlier pipeline that under-counted "
            "ground-truth services."
        ),
    )
    return p.parse_args()


def extract_rc_services(injection: dict[str, Any]) -> list[str]:
    """Pull the ground-truth root-cause service list out of injection.json.

    Both schemas seen in lite_v1 store ``ground_truth`` either as a single dict
    or as a list of dicts; the AegisLab batch cases use the list form.
    """
    gt_raw = injection.get("ground_truth") or {}
    gt_list = gt_raw if isinstance(gt_raw, list) else [gt_raw]
    services: list[str] = []
    seen: set[str] = set()
    for gt in gt_list:
        if not isinstance(gt, dict):
            continue
        for svc in gt.get("service") or []:
            if svc and svc not in seen:
                seen.add(svc)
                services.append(svc)
    return services


def load_manifest(lite_root: Path) -> list[dict[str, Any]]:
    manifest_path = lite_root / "MANIFEST.json"
    if not manifest_path.exists():
        sys.exit(f"MANIFEST.json not found at {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    cases = manifest.get("cases") or []
    if not cases:
        sys.exit(f"MANIFEST.json at {manifest_path} has no cases")
    return cases


def case_to_row(
    case: dict[str, Any],
    lite_root: Path,
    dataset: str,
    tag: str,
    source_index: int,
    source_style: str = "unprefixed",
) -> dict[str, Any] | None:
    """Build the kwargs for a DatasetSample row from a manifest entry.

    Returns None if the case dir is missing required files (skip with warning).
    """
    # MANIFEST entries are inconsistent: some use ``name`` (prefixed with
    # ``system__faulttype__``), others use ``case`` (unprefixed). Reconstruct
    # the prefixed disk-dir name so the on-disk lookup always matches.
    raw_name = case.get("name") or case.get("case")
    if not raw_name:
        print(f"  warn: manifest entry missing name/case: {case}", file=sys.stderr)
        return None
    system = case.get("system", "")
    fault_type = case.get("fault_type", "")
    if "__" in raw_name:
        dir_name = raw_name
    elif system and fault_type:
        dir_name = f"{system}__{fault_type}__{raw_name}"
    else:
        dir_name = raw_name
    name = _unprefix(dir_name) if source_style == "unprefixed" else dir_name

    case_dir = lite_root / dir_name
    injection_path = case_dir / "injection.json"
    causal_path = case_dir / "causal_graph.json"
    if not injection_path.exists() or not causal_path.exists():
        print(f"  warn: skip {name}: missing injection.json or causal_graph.json", file=sys.stderr)
        return None

    injection = json.loads(injection_path.read_text())

    # The MANIFEST's ``rc_services`` field is curation metadata (typically the
    # primary stratification service, often a single name in a possibly
    # decorated form like ``hotel-reserv-search``). The authoritative ground
    # truth is in ``injection.json`` — for batch fault injections it lists
    # every targeted service in bare-name form (``search``, ``profile``…).
    # The downstream judge does set-based matching, so under-reporting here
    # is a real precision issue.
    rc_services = extract_rc_services(injection)
    if not rc_services:
        print(f"  warn: skip {name}: no ground-truth services found", file=sys.stderr)
        return None

    answer = ",".join(rc_services)
    system = case.get("system", "")
    fault_type = case.get("fault_type", injection.get("fault_type", ""))

    # ``question`` and ``topic`` are intentionally left empty to match the
    # existing prod RCABench convention: ``RCABenchProcesser`` builds the
    # actual prompt at preprocess time from ``causal_graph.json`` (alarm
    # endpoints) plus the augmentation template, and writes it to
    # ``augmented_question``. The structured info is still in ``meta``.
    question = ""

    meta = {
        "source_data_dir": str(case_dir.resolve()),
        "system": system,
        "fault_type": fault_type,
        "rc_services": rc_services,
        "alarm_services": case.get("alarm_services", []),
        "skeleton": case.get("skeleton", []),
        "n_skeletons": case.get("n_skeletons"),
        "origin": case.get("origin"),
    }

    origin = case.get("origin")
    tags = [tag]
    if origin in ("new", "old"):
        tags.append(f"{tag}-{origin}")

    return {
        "dataset": dataset,
        "index": source_index,
        "source": name,
        "source_index": source_index,
        "question": question,
        "answer": answer,
        "topic": None,
        "level": 0,
        "file_name": dir_name,
        "meta": meta,
        "tags": tags,
    }


def main() -> int:
    args = parse_args()

    lite_root = Path(args.lite_root).expanduser().resolve()
    if not lite_root.is_dir():
        sys.exit(f"--lite-root {lite_root} is not a directory")

    cases = load_manifest(lite_root)
    if args.cases_from:
        whitelist = {
            line.strip() for line in Path(args.cases_from).read_text().splitlines() if line.strip()
        }
        # Match against either `name` or `case` so unprefixed and prefixed
        # whitelist entries both work.
        cases = [
            c for c in cases
            if (c.get("name") in whitelist) or (c.get("case") in whitelist)
        ]
        print(f"Filtered manifest to {len(cases)} cases via {args.cases_from}")
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    print(f"Loaded {len(cases)} cases from {lite_root / 'MANIFEST.json'}")

    rows: list[dict[str, Any]] = []
    for i, case in enumerate(cases, start=1):
        row = case_to_row(
            case, lite_root, args.dataset, args.tag,
            source_index=i, source_style=args.source_style,
        )
        if row is not None:
            rows.append(row)
    print(
        f"Built {len(rows)} valid rows (dataset={args.dataset}, tag={args.tag}, "
        f"source_style={args.source_style}, merge_tags={args.merge_tags})"
    )

    if args.dry_run:
        print("Dry run — first 3 rows:")
        for row in rows[:3]:
            print(json.dumps({**row, "meta": {**row["meta"], "source_data_dir": "..."}}, indent=2))
        return 0

    if args.db_url:
        os.environ["LLM_EVAL_DB_URL"] = args.db_url
    if not os.environ.get("LLM_EVAL_DB_URL") and not os.environ.get("UTU_DB_URL"):
        sys.exit("Either set LLM_EVAL_DB_URL env var or pass --db-url.")

    from sqlalchemy.orm import defer
    from sqlmodel import select

    from rcabench_platform.v3.sdk.llm_eval.db import DatasetSample
    from rcabench_platform.v3.sdk.llm_eval.utils import SQLModelUtils

    inserted = 0
    updated = 0
    skipped = 0
    with SQLModelUtils.create_session() as session:
        # Defer the heavy ``question``/``meta`` columns: in --merge-tags mode we
        # only ever read & write ``tags`` on existing rows. Skipping these
        # cuts the prod-DB round-trip from minutes to seconds.
        stmt = select(DatasetSample).where(DatasetSample.dataset == args.dataset)
        if args.merge_tags:
            stmt = stmt.options(
                defer(DatasetSample.question),  # type: ignore[arg-type]
                defer(DatasetSample.meta),  # type: ignore[arg-type]
            )
        existing_rows = session.exec(stmt).all()
        existing_by_key = {(r.dataset, r.source): r for r in existing_rows}
        next_index = (max((r.index or 0) for r in existing_rows), 0)[0] + 1 if existing_rows else 1

        for row in rows:
            key = (row["dataset"], row["source"])
            if key in existing_by_key:
                # Reuse the already-loaded row — no extra round-trip.
                existing = existing_by_key[key]
                if args.merge_tags:
                    # Conservative path for shared/prod DBs: only touch the
                    # tags array, leave every other field as-written by the
                    # original seeder (meta paths point at that machine, etc.).
                    current_tags = list(existing.tags or [])
                    changed = False
                    for t in row["tags"]:
                        if t not in current_tags:
                            current_tags.append(t)
                            changed = True
                    if changed:
                        existing.tags = current_tags
                    if args.refresh_answer and existing.answer != row["answer"]:
                        existing.answer = row["answer"]
                        changed = True
                    if changed:
                        session.add(existing)
                        updated += 1
                    else:
                        skipped += 1
                else:
                    existing.tags = row["tags"]
                    existing.meta = row["meta"]
                    existing.answer = row["answer"]
                    existing.question = row["question"]
                    existing.topic = row["topic"]
                    existing.file_name = row["file_name"]
                    session.add(existing)
                    updated += 1
            else:
                # Append after existing index range so prod stays tidy.
                row_for_insert = dict(row)
                row_for_insert["index"] = next_index
                next_index += 1
                session.add(DatasetSample(**row_for_insert))
                inserted += 1

        if args.dry_run_db:
            print("--dry-run-db: rolling back; no writes committed")
            session.rollback()
        else:
            session.commit()

    print(f"DB seed complete: inserted={inserted}, updated={updated}, skipped={skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
