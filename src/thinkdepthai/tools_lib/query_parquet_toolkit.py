"""Query Parquet Files Toolkit - Query and analyze parquet files using DuckDB."""

import json
import os
import re
from datetime import datetime
from pathlib import Path

import duckdb

from .base import AsyncBaseToolkit, register_tool

TOKEN_LIMIT = 5000

# Only telemetry parquets — log/trace/metric-bearing files — are exposed to
# the agent. Anything else in a case dir (notably conclusion.parquet, which
# encodes the ground-truth root cause) must stay invisible to prevent
# label leakage during evaluation.
_ALLOWED_PARQUET_RE = re.compile(r"(log|trace|metric)", re.IGNORECASE)


def _is_allowed_parquet(path: str | Path) -> bool:
    return _ALLOWED_PARQUET_RE.search(Path(path).stem) is not None


os.environ["TOKENIZERS_PARALLELISM"] = "false"

_use_char_based = None


def _should_use_char_based_estimation() -> bool:
    global _use_char_based
    if _use_char_based is None:
        env_value = os.getenv("UTU_USE_CHAR_BASED_TOKEN_ESTIMATION", "true").lower()
        _use_char_based = env_value in ("true", "1", "yes")
    return _use_char_based


class QueryParquetFilesToolkit(AsyncBaseToolkit):
    def __init__(self, config: dict | None = None):
        super().__init__(config)

    def _serialize_datetime(self, obj):
        import math

        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, float):
            if math.isnan(obj):
                return None
            elif math.isinf(obj):
                return None
            return obj
        elif isinstance(obj, dict):
            return {key: self._serialize_datetime(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_datetime(item) for item in obj]
        return obj

    def _estimate_token_count(self, text: str) -> int:
        average_chars_per_token = 3
        return (len(text) + average_chars_per_token - 1) // average_chars_per_token

    def _enforce_token_limit(self, payload: str, context: str) -> str:
        token_estimate = self._estimate_token_count(payload)
        if token_estimate <= TOKEN_LIMIT:
            return payload

        current_size = len(json.loads(payload)) if payload.startswith("[") else None
        suggested_limit = None
        if current_size:
            ratio = TOKEN_LIMIT / token_estimate
            suggested_limit = max(1, int(current_size * ratio * 0.8))

        warning = {
            "error": "Result exceeds token budget",
            "context": context,
            "estimated_tokens": token_estimate,
            "token_limit": TOKEN_LIMIT,
            "rows_returned": current_size,
            "suggested_limit": suggested_limit,
            "suggestion": "Reduce LIMIT, add WHERE clauses, or use aggregations.",
        }
        return json.dumps(warning, ensure_ascii=False, indent=2)

    @register_tool
    async def query_parquet_files(self, parquet_files: str | list[str], query: str, limit: int = 10) -> str:
        """Query parquet files using SQL syntax for data analysis and exploration.

        Args:
            parquet_files: Path(s) to parquet file(s)
            query: SQL query to execute
            limit: Maximum number of records to return (default: 10)

        Returns:
            JSON string of query results
        """
        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]

        for fp in parquet_files:
            if not _is_allowed_parquet(fp):
                return json.dumps(
                    {
                        "error": (
                            f"Access denied: {Path(fp).name} is not a telemetry parquet. "
                            "Only files whose name contains 'log', 'trace', or 'metric' are queryable."
                        )
                    }
                )
            if not Path(fp).exists():
                return json.dumps({"error": f"Parquet file not found: {fp}"})

        conn = duckdb.connect(":memory:")
        table_names: set[str] = set()

        try:
            for file_path in parquet_files:
                base_name = Path(file_path).stem
                table_name = base_name
                counter = 1
                while table_name in table_names:
                    table_name = f"{base_name}_{counter}"
                    counter += 1
                table_names.add(table_name)
                conn.execute(f"CREATE VIEW {table_name} AS SELECT * FROM read_parquet('{file_path}')")

            result = conn.execute(query).fetchall()
            columns = [desc[0] for desc in conn.description]
            rows = [dict(zip(columns, row, strict=False)) for row in result]
            serialized_rows = self._serialize_datetime(rows)

            if len(serialized_rows) > limit:
                serialized_rows = serialized_rows[:limit]

            result_json = json.dumps(serialized_rows, ensure_ascii=False, indent=2)
            return self._enforce_token_limit(result_json, "query_parquet_files")
        except Exception as e:
            return json.dumps(
                {"error": f"Query execution failed: {e}", "query": query, "available_tables": list(table_names)}
            )
        finally:
            conn.close()

    @register_tool
    async def get_schema(self, parquet_file: str) -> str:
        """Get schema information of a parquet file.

        Args:
            parquet_file: Path to parquet file to inspect

        Returns:
            JSON string containing file metadata, row count, and column information
        """
        if not _is_allowed_parquet(parquet_file):
            return json.dumps(
                {
                    "error": (
                        f"Access denied: {Path(parquet_file).name} is not a telemetry parquet. "
                        "Only files whose name contains 'log', 'trace', or 'metric' are inspectable."
                    )
                }
            )
        if not Path(parquet_file).exists():
            return json.dumps({"error": f"Parquet file not found: {parquet_file}"})

        conn = duckdb.connect(":memory:")
        try:
            result = conn.execute(f"SELECT * FROM read_parquet('{parquet_file}') LIMIT 0")
            schema = [{"name": desc[0], "type": str(desc[1])} for desc in result.description]
            row_count_result = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{parquet_file}')").fetchone()
            row_count = row_count_result[0] if row_count_result else 0

            schema_info = {"file": parquet_file, "row_count": row_count, "columns": schema}
            result_json = json.dumps(schema_info, ensure_ascii=False, indent=2)
            return self._enforce_token_limit(result_json, "get_schema")
        except Exception as e:
            return json.dumps({"error": f"Failed to extract schema: {e}", "file": parquet_file})
        finally:
            conn.close()

    @register_tool
    async def list_tables_in_directory(self, directory: str) -> str:
        """List all parquet files in a directory with metadata.

        Args:
            directory: Directory path to search for parquet files

        Returns:
            JSON string containing list of files with metadata
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            return json.dumps({"error": f"Directory not found: {directory}"})
        if not dir_path.is_dir():
            return json.dumps({"error": f"Path is not a directory: {directory}"})

        files_info = []
        for file_path in dir_path.glob("*.parquet"):
            if not _is_allowed_parquet(file_path):
                continue
            try:
                conn = duckdb.connect(":memory:")
                row_count_result = conn.execute(f"SELECT COUNT(*) FROM read_parquet('{file_path}')").fetchone()
                row_count = row_count_result[0] if row_count_result else 0
                result = conn.execute(f"SELECT * FROM read_parquet('{file_path}') LIMIT 0")
                column_count = len(result.description)
                conn.close()
                files_info.append(
                    {
                        "filename": file_path.name,
                        "path": str(file_path),
                        "row_count": row_count,
                        "column_count": column_count,
                    }
                )
            except Exception as e:
                files_info.append({"filename": file_path.name, "path": str(file_path), "error": str(e)})

        result_json = json.dumps(files_info, ensure_ascii=False, indent=2)
        return self._enforce_token_limit(result_json, "list_tables_in_directory")
