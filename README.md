# ThinkDepthAI

# ThinkDepthAI LangGraph RCA Agent Workflow

## Architecture Diagram

```mermaid
flowchart TD
    subgraph BuildPhase["Build Phase (LanggraphRCAAgent.build)"]
        B1[Load Toolkit<br/>query_parquet_files] --> B2[Create LangChain Model<br/>bind_tools]
        B2 --> B3[Load Prompt Config<br/>rca.yaml]
        B3 --> B4[Compile LangGraph<br/>build_rca_agent]
    end

    B4 --> START

    subgraph RunPhase["Run Phase (agent.run)"]
        START([START]) --> INIT[Initialize State<br/>incident_description + empty messages]
        INIT --> TRAJ[Initialize TrajectoryLogger<br/>JSONL real-time logs]
        TRAJ --> LLM

        subgraph Loop["ReAct Loop (LangGraph)"]
            LLM["llm_call<br/><br/>System: RCA_ANALYSIS_SP<br/>User: RCA_ANALYSIS_UP"]
            LLM --> DECISION{should_continue?<br/>check tool_calls}
            DECISION -- has tool_calls --> TOOL["tool_node<br/><br/>- query_parquet_files<br/>- list_tables_in_directory<br/>- get_schema<br/>- think_tool (mandatory)"]
            TOOL --> LLM
            DECISION -- no tool_calls --> COMPRESS["compress_rca_findings<br/><br/>System: COMPRESS_FINDINGS_SP<br/>User: COMPRESS_FINDINGS_UP<br/>+ all history messages"]
        end

        COMPRESS --> VALIDATE[JSON Extraction & Validation<br/>_validate_causal_graph_json]
        VALIDATE --> RESULT[Construct Trajectory<br/>Return AgentResult]
    end

    RESULT --> END([END])

    style LLM fill:#e1f5fe
    style TOOL fill:#fff3e0
    style COMPRESS fill:#e8f5e9
    style DECISION fill:#fce4ec
    style Loop fill:#f5f5f5,stroke:#9e9e9e,stroke-dasharray: 5 5
```

---

## State Definition (RCAState)

```python
class RCAState(TypedDict):
    messages: Sequence[BaseMessage]           # conversation history
    tool_call_iterations: int                 # tool call count
    incident_description: str                 # incident description
    rca_findings: str                         # final RCA result
    raw_notes: list[str]                      # raw notes
```

---

## Prompt Examples

### 1. RCA_ANALYSIS_SP (System Prompt - Analysis Phase)

> You are a Root Cause Analysis (RCA) expert conducting systematic investigation of system incidents.
>
> **Your goal is to identify:**
> 1. **Root Cause Service**: Which service is the origin of the failure
> 2. **Fault Propagation Path**: How the error propagated through the system as a causal graph
>
> **Available Data Types:**
> - **Logs**: normal_logs.parquet, abnormal_logs.parquet
> - **Traces**: normal_traces.parquet, abnormal_traces.parquet
> - **Metrics**: normal_metrics.parquet, abnormal_metrics.parquet
>
> **Available Tools:**
> 1. `query_parquet_files` - Query parquet files using SQL
> 2. `list_tables_in_directory` - List all parquet files
> 3. `get_schema` - Get schema information
>
> **CRITICAL: Use `think_tool` after each search to reflect on results and plan next steps**
>
> **Tool Call Budget**: 10-15 typical, **stop after 20**
>
> **Output MUST be CausalGraph JSON** with `nodes`, `edges`, `root_causes`, `component_to_service`

---

### 2. RCA_ANALYSIS_UP (User Prompt - Analysis Phase)

> Please conduct a Root Cause Analysis for the following incident:
>
> ## Incident Description
> `{incident_description}`
>
> ## Your Mission
> Identify:
> 1. **Root Cause Service**: The service where the failure originated
> 2. **Fault Propagation Graph**: The complete causal chain from root cause to all affected services
>
> ## Investigation Strategy
> 1. **Discover Available Data** - `list_tables_in_directory`
> 2. **Understand Data Structure** - `get_schema`
> 3. **Identify Anomalies** - Query abnormal vs normal data
> 4. **Trace Service Dependencies** - Use trace data
> 5. **Determine Root Cause** - Find the earliest abnormal service
> 6. **Map Propagation Path** - Build edges A->B
>
> **Output Format (MUST produce CausalGraph JSON):**
> ```json
> {
>   "nodes": [{"component": "service-name", "state": ["HIGH_ERROR_RATE"], "timestamp": 1234567890}],
>   "edges": [{"source": "root-cause-service", "target": "affected-service"}],
>   "root_causes": [{"component": "root-cause-service", "state": ["HIGH_ERROR_RATE"], "timestamp": 1234567890}],
>   "component_to_service": {}
> }
> ```
>
> **Remember**: Use `think_tool` after each query. Stop when you have enough evidence.

---

### 3. COMPRESS_FINDINGS_SP (System Prompt - Synthesis Phase)

> You are an expert Root Cause Analysis synthesizer.
> Your task is to convert investigation findings into structured CausalGraph JSON format.

---

### 4. COMPRESS_FINDINGS_UP (User Prompt - Synthesis Phase)

> You are an RCA expert who has conducted a thorough investigation of a system incident.
> Your job is now to synthesize all findings into a structured CausalGraph JSON format.
>
> **Task:**
> Transform all investigation findings from tool calls into a structured CausalGraph showing:
> 1. **Root Cause**: Which service(s) initiated the failure
> 2. **Propagation Path**: How the fault spread through the system (as a directed graph)
> 3. **All Affected Services**: Complete list of impacted services
>
> **Tool Call Filtering:**
> - **Include**: All query results showing anomalies, errors, failures
> - **Exclude**: `think_tool` calls (internal reasoning)
> - **Focus on**: Concrete evidence of what went wrong and how it propagated
>
> **Output Requirements:**
> You MUST output ONLY a valid JSON object in the CausalGraph format.
>
> **Critical Rules:**
> - Output **ONLY** the JSON object, no markdown, no explanations
> - `root_causes` field is **MANDATORY**
> - The CausalGraph MUST cover the FULL chain: root cause -> intermediate services -> alert endpoint

---

## Output Example (CausalGraph JSON)

```json
{
  "nodes": [
    {"component": "ts-order-service", "state": ["HIGH_ERROR_RATE"], "timestamp": 1744500000000000000},
    {"component": "ts-payment-service", "state": ["TIMEOUT"], "timestamp": 1744500005000000000},
    {"component": "ts-user-service", "state": ["HIGH_LATENCY"], "timestamp": 1744500010000000000}
  ],
  "edges": [
    {"source": "ts-order-service", "target": "ts-payment-service"},
    {"source": "ts-payment-service", "target": "ts-user-service"}
  ],
  "root_causes": [
    {"component": "ts-order-service", "state": ["HIGH_ERROR_RATE"], "timestamp": 1744500000000000000}
  ],
  "component_to_service": {}
}
```

---

## Key Code Paths

| Component | File Path |
|-----------|-----------|
| Agent Class | `src/thinkdepthai/agent.py` |
| Graph Definition | `src/thinkdepthai/graph.py` |
| State Definition | `src/thinkdepthai/state.py` |
| Prompt Template | `src/thinkdepthai/prompts/agents/langgraph/rca.yaml` |

---

## Evaluation

### Prerequisites

1. Install dependencies with all extras:

```bash
uv sync
```

2. Configure environment variables in `.env`:

```bash
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=<your-model>
UTU_LLM_BASE_URL=<your-api-base-url>
UTU_LLM_API_KEY=<your-api-key>
```

3. Prepare the dataset at `/mnt/jfs/rcabench_dataset` (or override with `--source-path`).

4. Prepare the evaluation database. The pipeline requires pre-registered samples in the `data` table. If you have an existing PostgreSQL database with RCABench data, export the `openrca2-lite` subset to a local SQLite file:

```bash
# Example: export from PostgreSQL to SQLite
uv run python3 -c "
import json
from sqlmodel import create_engine, text, SQLModel

pg = create_engine('postgresql://user:pass@host/db')
sqlite = create_engine('sqlite:///eval.db')
SQLModel.metadata.create_all(sqlite)

with pg.connect() as src:
    rows = src.execute(text(\"SELECT * FROM data WHERE tags::jsonb @> '[\\\"openrca2-lite\\\"]'\")).fetchall()

with sqlite.connect() as dst:
    for row in rows:
        dst.execute(text('''
            INSERT INTO data (id, dataset, \"index\", source, source_index, question, answer, topic, level, file_name, meta, tags)
            VALUES (:id, :dataset, :index, :source, :source_index, :question, :answer, :topic, :level, :file_name, :meta, :tags)
        '''), {
            'id': row[0], 'dataset': row[1], 'index': row[2], 'source': row[3],
            'source_index': row[4], 'question': row[5], 'answer': row[6],
            'topic': row[7], 'level': row[8], 'file_name': row[9],
            'meta': json.dumps(row[10]) if row[10] else None,
            'tags': json.dumps(row[11]) if row[11] else None
        })
    dst.commit()
"
```

Then set `db_url` in the config:

```yaml
# config/eval/openrca2_lite.yaml
db_url: "sqlite:///eval.db"
```

### Running Evaluation

Run the full pipeline (preprocess + rollout + judge + stat) with the ThinkDepthAI agent:

```bash
uv run rca llm-eval run config/eval/openrca2_lite.yaml \
  -a thinkdepthai \
  --source-path /mnt/jfs/rcabench_dataset
```

Limit to a single sample for quick testing:

```bash
uv run rca llm-eval run config/eval/openrca2_lite.yaml \
  -a thinkdepthai \
  --source-path /mnt/jfs/rcabench_dataset \
  -l 1 
```

Launch with real-time dashboard:

```bash
uv run rca llm-eval run config/eval/openrca2_lite.yaml \
  -a thinkdepthai \
  --source-path /mnt/jfs/rcabench_dataset \
  --dashboard \
  --dashboard-port 8766
```


## Setup on a Fresh Machine

To run evaluation on a new machine, you need three things: the code, the dataset metadata (in `eval.db`), and the telemetry data.

### Quick Start

```bash
# 1. Clone the repository
git clone <repo-url>
cd ThinkDepthAI

# 2. Run the setup script
./scripts/setup_eval_env.sh
```

This script will:
1. Check that `uv` is installed
2. Install all Python dependencies
3. Download the dataset metadata from HuggingFace (`lincyaw/rca`)
4. Initialize `eval.db` with all 500 samples
5. Verify the database is ready

### Manual Setup

If you prefer manual steps or the script does not work:

```bash
# 1. Install dependencies
uv sync

# 2. Download dataset metadata
uv run python3 -c "
from datasets import load_dataset
import json

ds = load_dataset('lincyaw/rca', split='train')
with open('data/data.jsonl', 'w') as f:
    for item in ds:
        f.write(json.dumps(dict(item), ensure_ascii=False) + '\n')
"

# 3. Initialize eval.db
uv run python3 scripts/init_eval_db.py --jsonl data/data.jsonl --db eval.db

# 4. Download telemetry data (the actual parquet files)
# Option A: From HuggingFace (if network allows)
# Option B: Copy from existing machine: rsync -av /mnt/jfs/rcabench_dataset/ ./data/
# Option C: Use the hf_upload/data/ directory if you have it
```

### After Setup

1. Configure API keys in `.env`:

```bash
UTU_LLM_TYPE=chat.completions
UTU_LLM_MODEL=<your-model>
UTU_LLM_BASE_URL=<your-api-url>
UTU_LLM_API_KEY=<your-api-key>
```

2. Update `config/eval/openrca2_lite.yaml`:

```yaml
db_url: "sqlite:///eval.db"
source_path: "./data"  # or wherever the telemetry data is
```

3. Run evaluation:

```bash
uv run rca llm-eval run config/eval/openrca2_lite.yaml \
  -a thinkdepthai \
  --source-path ./data
```
