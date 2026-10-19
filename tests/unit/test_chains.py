import json
import re
from pathlib import Path

from dht.toolcall import chains
from dht.toolcall.nestful import basic_function_names

CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
RESP = re.compile(r"<tool_response>\s*(\{.*?\})\s*</tool_response>", re.DOTALL)
ROOT = Path(__file__).resolve().parents[2]


def test_tools_are_disjoint_from_nestful():
    nestful = basic_function_names(ROOT / "third_party/nestful/basic_functions.py")
    assert nestful, "vendored NESTful functions not found"
    assert not {t.name for t in chains.TOOLS} & nestful


def test_generation_is_deterministic():
    a = chains.generate(20, 3, "S {tools}")
    b = chains.generate(20, 3, "S {tools}")
    assert a == b
    assert a != chains.generate(20, 4, "S {tools}")


def test_conversations_are_executed_dependent_chains():
    for ex in chains.generate(200, 0, "S {tools}"):
        conv = ex["conversations"]
        assert [m["from"] for m in conv[:2]] == ["system", "human"]
        assert conv[-1]["from"] == "gpt" and "<tool_call>" not in conv[-1]["value"]
        tools = {t["function"]["name"] for t in json.loads(conv[0]["value"][2:])}
        seen_results: list = []
        n_calls = 0
        for i, m in enumerate(conv[2:-1]):
            if m["from"] == "gpt":
                calls = [json.loads(c) for c in CALL.findall(m["value"])]
                assert calls and all(c["name"] in tools for c in calls)
                if i > 0:  # every later turn uses at least one earlier result
                    assert any(v in seen_results for c in calls for v in c["arguments"].values())
                for c in calls:
                    tool = chains.BY_NAME[c["name"]]
                    expected = tool.fn(**c["arguments"])
                    n_calls += 1
                    pending = expected
                assert conv[2 + i + 1]["from"] == "tool"
            else:
                results = [json.loads(r)["content"] for r in RESP.findall(m["value"])]
                seen_results.extend(results)
                assert results[-1] == pending  # responses are the real outputs
        assert n_calls == ex["n_calls"] and 2 <= n_calls <= 6
        assert ex["turns"] >= 2
        assert json.dumps(seen_results[-1]) in conv[-1]["value"]
