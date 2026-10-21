import json
from pathlib import Path

from dht.toolcall.dory_format import openai_tool, parse_calls, split_thinking

SPECS = {
    t["name"]: t
    for t in json.loads(
        (Path(__file__).resolve().parents[2] / "third_party/nestful/tool_specs.json").read_text()
    )
}


def test_openai_tool_schema():
    tool = openai_tool(SPECS["divide"])["function"]
    assert tool["name"] == "divide"
    assert tool["parameters"]["properties"]["arg_0"]["type"] == "number"
    assert tool["parameters"]["required"] == ["arg_0", "arg_1"]
    lst = openai_tool(SPECS["max_number"])["function"]["parameters"]["properties"]["arg_0"]
    assert lst["type"] == "array"


def test_split_and_parse():
    out = (
        "Need 110*5 first.\n</think>\n<tool_call>\n<function=multiply>\n<parameter=arg_0>\n110\n"
        "</parameter>\n<parameter=arg_1>\n5\n</parameter>\n</function>\n</tool_call>\n"
        "<tool_call>\n<function=max_number>\n<parameter=arg_0>\n[1, 2.5]\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    reasoning, answer = split_thinking(out)
    assert reasoning == "Need 110*5 first."
    assert parse_calls(answer) == [
        {"name": "multiply", "arguments": {"arg_0": 110, "arg_1": 5}},
        {"name": "max_number", "arguments": {"arg_0": [1, 2.5]}},
    ]
    assert split_thinking("still thinking")[1] == ""
    assert parse_calls("The final result is 660.") == []
