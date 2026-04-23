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


def test_validate_causal_graph_json():
    """Test CausalGraph JSON validation."""
    from thinkdepthai.agent import LanggraphRCAAgent

    # Valid JSON
    valid = json.dumps({
        "nodes": [{"component": "svc-a", "state": ["HIGH_ERROR_RATE"]}],
        "edges": [{"source": "svc-a", "target": "svc-b"}],
        "root_causes": [{"component": "svc-a", "state": ["HIGH_ERROR_RATE"]}],
        "component_to_service": {},
    })
    result = LanggraphRCAAgent._validate_causal_graph_json(valid)
    parsed = json.loads(result)
    assert len(parsed["nodes"]) == 1
    assert "parse_error" not in parsed

    # Empty input
    result = LanggraphRCAAgent._validate_causal_graph_json("")
    parsed = json.loads(result)
    assert "parse_error" in parsed

    # JSON in markdown code block
    markdown = '```json\n{"nodes": [], "edges": [], "root_causes": []}\n```'
    result = LanggraphRCAAgent._validate_causal_graph_json(markdown)
    parsed = json.loads(result)
    assert parsed["nodes"] == []
    assert parsed["edges"] == []


def test_extract_json_from_text():
    """Test JSON extraction from text."""
    from thinkdepthai.agent import LanggraphRCAAgent

    # Plain JSON
    assert LanggraphRCAAgent._extract_json_from_text('{"key": "value"}') == '{"key": "value"}'

    # JSON in markdown
    md = 'Some text\n```json\n{"a": 1}\n```\nMore text'
    assert LanggraphRCAAgent._extract_json_from_text(md) == '{"a": 1}'

    # No JSON
    assert LanggraphRCAAgent._extract_json_from_text("no json here") is None

    # Empty
    assert LanggraphRCAAgent._extract_json_from_text("") is None


def test_converter_import():
    """Verify converter can be imported."""
    from thinkdepthai.converters import TrajectoryConverter

    assert hasattr(TrajectoryConverter, "from_langchain_messages")
