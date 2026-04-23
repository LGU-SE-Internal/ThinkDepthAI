import argparse
import json
import os

from datasets import load_dataset
from sqlmodel import create_engine, text


def init_db(db_path: str = "eval.db"):
    """Create SQLite database with the data table schema."""
    engine = create_engine(f"sqlite:///{db_path}")

    with engine.connect() as conn:
        conn.execute(
            text("""
            CREATE TABLE IF NOT EXISTS data (
                id INTEGER PRIMARY KEY,
                dataset VARCHAR NOT NULL,
                "index" INTEGER NOT NULL,
                source VARCHAR NOT NULL,
                source_index INTEGER,
                question VARCHAR,
                answer VARCHAR,
                topic VARCHAR,
                level INTEGER,
                file_name VARCHAR,
                meta JSON,
                tags JSON
            )
        """)
        )
        conn.commit()

    return engine


def populate_from_hf(engine, dataset_name: str = "lincyaw/rca", split: str = "train"):
    """Load dataset from HuggingFace and insert into the database."""
    print(f"Loading dataset {dataset_name} (split={split})...")
    ds = load_dataset(dataset_name, split=split)
    print(f"Loaded {len(ds)} samples")

    with engine.connect() as conn:
        for i, item in enumerate(ds):
            meta = {
                "api_reports": item.get("api_reports", []),
                "ground_truth": item.get("ground_truth", []),
                "datapack_name": item.get("datapack_name", ""),
                "difficulty": {
                    "spl": item.get("difficulty_spl", 0),
                    "n_svc": item.get("difficulty_n_svc", 0),
                    "n_edge": item.get("difficulty_n_edge", 0),
                    "fault_type": item.get("fault_type", ""),
                    "fault_category": item.get("fault_category", ""),
                },
            }

            conn.execute(
                text("""
                INSERT INTO data (id, dataset, "index", source, source_index, question, answer, topic, level, file_name, meta, tags)
                VALUES (:id, :dataset, :index, :source, :source_index, :question, :answer, :topic, :level, :file_name, :meta, :tags)
            """),
                {
                    "id": item["id"],
                    "dataset": "RCABench",
                    "index": item["id"],
                    "source": item["source"],
                    "source_index": None,
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "topic": "",
                    "level": 0,
                    "file_name": "",
                    "meta": json.dumps(meta, ensure_ascii=False),
                    "tags": json.dumps(item.get("tags", []), ensure_ascii=False),
                },
            )

            if (i + 1) % 100 == 0:
                print(f"  Inserted {i + 1}/{len(ds)} samples...")

        conn.commit()

    print(f"Done. Inserted {len(ds)} samples into the database.")


def populate_from_jsonl(engine, jsonl_path: str):
    """Load dataset from a local JSONL file and insert into the database."""
    print(f"Loading dataset from {jsonl_path}...")

    items = []
    with open(jsonl_path) as f:
        for line in f:
            items.append(json.loads(line.strip()))

    print(f"Loaded {len(items)} samples")

    with engine.connect() as conn:
        for i, item in enumerate(items):
            meta = {
                "api_reports": item.get("api_reports", []),
                "ground_truth": item.get("ground_truth", []),
                "datapack_name": item.get("datapack_name", ""),
                "difficulty": {
                    "spl": item.get("difficulty_spl", 0),
                    "n_svc": item.get("difficulty_n_svc", 0),
                    "n_edge": item.get("difficulty_n_edge", 0),
                    "fault_type": item.get("fault_type", ""),
                    "fault_category": item.get("fault_category", ""),
                },
            }

            conn.execute(
                text("""
                INSERT INTO data (id, dataset, "index", source, source_index, question, answer, topic, level, file_name, meta, tags)
                VALUES (:id, :dataset, :index, :source, :source_index, :question, :answer, :topic, :level, :file_name, :meta, :tags)
            """),
                {
                    "id": item["id"],
                    "dataset": "RCABench",
                    "index": item["id"],
                    "source": item["source"],
                    "source_index": None,
                    "question": item.get("question", ""),
                    "answer": item.get("answer", ""),
                    "topic": "",
                    "level": 0,
                    "file_name": "",
                    "meta": json.dumps(meta, ensure_ascii=False),
                    "tags": json.dumps(item.get("tags", []), ensure_ascii=False),
                },
            )

            if (i + 1) % 100 == 0:
                print(f"  Inserted {i + 1}/{len(items)} samples...")

        conn.commit()

    print(f"Done. Inserted {len(items)} samples into the database.")


def main():
    parser = argparse.ArgumentParser(description="Initialize eval.db for RCA evaluation")
    parser.add_argument("--db", default="eval.db", help="Path to SQLite database (default: eval.db)")
    parser.add_argument("--dataset", default="lincyaw/rca", help="HuggingFace dataset name (default: lincyaw/rca)")
    parser.add_argument("--jsonl", help="Path to local JSONL file (alternative to --dataset)")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    args = parser.parse_args()

    if os.path.exists(args.db):
        print(f"Removing existing database: {args.db}")
        os.remove(args.db)

    print(f"Creating database: {args.db}")
    engine = init_db(args.db)

    if args.jsonl:
        populate_from_jsonl(engine, args.jsonl)
    else:
        populate_from_hf(engine, args.dataset, args.split)

    with engine.connect() as conn:
        result = conn.execute(text("SELECT COUNT(*) FROM data"))
        count = result.fetchone()[0]
        print(f"\nDatabase initialized with {count} samples.")
        print(f'Run evaluation with: db_url: "sqlite:///{args.db}"')


if __name__ == "__main__":
    main()
