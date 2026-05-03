"""Basic tests for ThinkDepthAI agent."""

import json


def test_import_agent():
    """Verify the agent can be imported."""
    from thinkdepthai.agent import AgentResult, LanggraphRCAAgent

    assert LanggraphRCAAgent is not None
    assert AgentResult is not None


def test_import_top_level():
    """Verify the top-level package exports work."""
    from thinkdepthai import LanggraphRCAAgent

    assert LanggraphRCAAgent is not None


def test_config_schema():
    """Verify config schema can be instantiated."""
    from thinkdepthai.config.schema import AgentConfig, ModelConfig, ModelProviderConfig

    mp = ModelProviderConfig(type="chat.completions", model="gpt-4o", api_format="openai")
    mc = ModelConfig(model_provider=mp)
    ac = AgentConfig(model=mc)
    assert ac.type == "langgraph_rca"
    assert ac.model.model_provider.model == "gpt-4o"


def test_prompt_manager():
    """Verify PromptManager can load prompts."""
    from thinkdepthai.prompts import PromptManager

    prompts = PromptManager.get_prompts("agents/langgraph/rca.yaml")
    assert "RCA_ANALYSIS_SP" in prompts
    assert "RCA_ANALYSIS_UP" in prompts
    assert "COMPRESS_FINDINGS_SP" in prompts
    assert "COMPRESS_FINDINGS_UP" in prompts


def test_think_tool():
    """Verify think_tool works."""
    from thinkdepthai.tools import think_tool

    result = think_tool.invoke({"reasoning": "test reasoning"})
    parsed = json.loads(result)
    assert parsed["status"] == "recorded"


def test_state_types():
    """Verify state TypedDicts are importable."""
    from thinkdepthai.state import RCAOutputState, RCAState

    assert RCAState is not None
    assert RCAOutputState is not None


def test_toolkit_registry():
    """Verify toolkit registry has query_parquet_files."""
    from thinkdepthai.agent import TOOLKIT_MAP

    assert "query_parquet_files" in TOOLKIT_MAP


def test_agent_instantiation():
    """Verify agent can be instantiated with explicit config."""
    from thinkdepthai.agent import LanggraphRCAAgent
    from thinkdepthai.config.schema import AgentConfig

    config = AgentConfig()
    agent = LanggraphRCAAgent(config=config, name="test-agent")
    assert agent.config.agent.name == "test-agent"
    assert not agent._initialized


def test_validate_causal_graph_json_v2():
    """v2 envelope: root_causes + propagation, each carrying evidence."""
    from thinkdepthai.agent import LanggraphRCAAgent

    valid = json.dumps({
        "root_causes": [
            {
                "service": "svc-a",
                "fault_kind": "dns",
                "evidence": [
                    {"kind": "log", "sql": "SELECT 1", "claim": "DNS failures observed"}
                ],
            }
        ],
        "propagation": [],
    })
    result = LanggraphRCAAgent._validate_causal_graph_json(valid)
    parsed = json.loads(result)
    assert len(parsed["root_causes"]) == 1
    assert "parse_error" not in parsed

    # Empty input → parse_error envelope, never raises.
    parsed = json.loads(LanggraphRCAAgent._validate_causal_graph_json(""))
    assert "parse_error" in parsed
    assert parsed["root_causes"] == []

    # Markdown-fenced JSON is unwrapped.
    fenced = (
        '```json\n'
        '{"root_causes":[{"service":"x","fault_kind":"dns","evidence":[{"kind":"log","sql":"SELECT 1","claim":"hi"}]}],'
        '"propagation":[]}\n'
        '```'
    )
    parsed = json.loads(LanggraphRCAAgent._validate_causal_graph_json(fenced))
    assert parsed["root_causes"][0]["service"] == "x"


def test_validate_truncated_synthesis_output():
    """Regression: truncated JSON (depth>0 at EOF) should yield a
    specific 'never closed' diagnosis, not a vague 'no JSON' message."""
    from thinkdepthai.output_validator import validate_rca_output

    # Open the outer object + propagation array but never close them.
    truncated = (
        '{"root_causes":[{"service":"a","fault_kind":"dns","evidence":'
        '[{"kind":"log","sql":"SELECT 1","claim":"x"}]}],"propagation":['
    )
    outcome = validate_rca_output(truncated)
    assert outcome.retry_warranted
    assert len(outcome.errors) == 1
    assert "truncated" in outcome.errors[0].lower()
    assert "brace-depth" in outcome.errors[0]


def test_validate_decode_error_actionable():
    """A JSON decode error should surface a single actionable message."""
    from thinkdepthai.output_validator import validate_rca_output

    outcome = validate_rca_output('{"root_causes":[{"service":"x",}]}')  # trailing comma
    assert outcome.retry_warranted
    assert any("decode error" in e.lower() for e in outcome.errors)


def test_validate_dropped_root_causes_warrants_retry():
    """When every root_cause is missing required fields, we retry."""
    from thinkdepthai.output_validator import validate_rca_output

    bad = json.dumps({
        "root_causes": [{"fault_kind": "dns", "evidence": [{"kind": "log", "sql": "SELECT 1", "claim": "hi"}]}],
        "propagation": [],
    })
    outcome = validate_rca_output(bad)
    assert outcome.retry_warranted
    assert any("service" in e for e in outcome.errors)


def test_converter_import():
    """Verify converter can be imported."""
    from thinkdepthai.converters import TrajectoryConverter

    assert hasattr(TrajectoryConverter, "from_langchain_messages")
