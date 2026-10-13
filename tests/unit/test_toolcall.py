import json

import pytest

from dht.toolcall.checker import check_example, normalize_string
from dht.toolcall.data import Example, load_bfcl, load_xlam, split_ids
from dht.toolcall.parser import ToolCall, parse_tool_calls
from dht.toolcall.schema import to_openai_tool, xlam_to_bfcl_function

TRIANGLE = {
    "name": "calculate_triangle_area",
    "description": "Area of a triangle.",
    "parameters": {
        "type": "dict",
        "properties": {
            "base": {"type": "integer"},
            "height": {"type": "integer"},
            "unit": {"type": "string"},
            "scale": {"type": "float"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["base", "height"],
    },
}
TRUTH = [
    {
        "calculate_triangle_area": {
            "base": [10],
            "height": [5],
            "unit": ["units", ""],
            "scale": ["", 1.0],
            "tags": ["", ["Big Shape"]],
        }
    }
]


def qwen_call(name, **params):
    body = "".join(
        f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v)}\n</parameter>\n"
        for k, v in params.items()
    )
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def grade(text, functions=(TRIANGLE,), truth=TRUTH):
    return check_example(parse_tool_calls(text, list(functions)), list(functions), truth)


def test_parser_types_values_from_schema():
    text = "thinking aside\n" + qwen_call(
        "calculate_triangle_area", base=10, height="5", unit="units", scale=1, tags=["a b"]
    )
    (call,) = parse_tool_calls(text, [TRIANGLE])
    assert call == ToolCall(
        "calculate_triangle_area",
        {"base": 10, "height": 5, "unit": "units", "scale": 1.0, "tags": ["a b"]},
    )


def test_parser_keeps_unconvertible_text_and_finds_multiple_calls():
    text = qwen_call("calculate_triangle_area", base="ten", height=5) + qwen_call("other", x=1)
    first, second = parse_tool_calls(text, [TRIANGLE])
    assert first.arguments["base"] == "ten"
    assert second.name == "other" and second.arguments == {"x": 1}
    assert parse_tool_calls("no call here", [TRIANGLE]) == []


def test_parser_tolerates_truncated_generation():
    text = "<tool_call>\n<function=calculate_triangle_area>\n<parameter=base>\n10\n"
    (call,) = parse_tool_calls(text, [TRIANGLE])
    assert call.arguments == {"base": 10}


@pytest.mark.parametrize(
    ("params", "reason"),
    [
        ({"base": 10, "height": 5}, "ok"),
        ({"base": 10, "height": 5, "unit": "Units"}, "ok"),
        ({"base": 10, "height": 5, "scale": 1}, "ok"),
        ({"base": 10, "height": 5, "tags": ["big-shape"]}, "ok"),
        ({"base": 10, "height": 6}, "wrong_value"),
        ({"base": 10}, "missing_required"),
        ({"base": 10, "height": 5, "colour": "red"}, "unexpected_param"),
        ({"base": "ten", "height": 5}, "wrong_type"),
        ({"base": True, "height": 5}, "wrong_type"),
    ],
)
def test_single_call_rules(params, reason):
    ok, why = grade(qwen_call("calculate_triangle_area", **params))
    assert why == reason and ok == (reason == "ok")


def test_missing_optional_needs_empty_alternative():
    truth = [{"calculate_triangle_area": {"base": [10], "height": [5], "unit": ["cm"]}}]
    assert grade(qwen_call("calculate_triangle_area", base=10, height=5), truth=truth) == (
        False,
        "missing_optional_value",
    )


def test_call_count_function_name_and_irrelevance():
    assert grade("I can't help") == (False, "no_call")
    assert grade(qwen_call("nope", base=1)) == (False, "unknown_function")
    double = qwen_call("calculate_triangle_area", base=10, height=5) * 2
    assert grade(double) == (False, "wrong_call_count")
    assert grade("No tool applies.", truth=[]) == (True, "ok")
    assert grade(qwen_call("calculate_triangle_area", base=1, height=1), truth=[]) == (
        False,
        "called_irrelevant",
    )


def test_parallel_calls_match_in_any_order():
    truth = [
        {"calculate_triangle_area": {"base": [1], "height": [2]}},
        {"calculate_triangle_area": {"base": [3], "height": [4]}},
    ]
    text = qwen_call("calculate_triangle_area", base=3, height=4) + qwen_call(
        "calculate_triangle_area", base=1, height=2
    )
    assert grade(text, truth=truth) == (True, "ok")
    text = qwen_call("calculate_triangle_area", base=3, height=4) * 2
    assert grade(text, truth=truth) == (False, "no_matching")


def test_nested_dict_values():
    fn = {
        "name": "db",
        "parameters": {
            "type": "dict",
            "properties": {"conditions": {"type": "dict"}},
            "required": ["conditions"],
        },
    }
    truth = [{"db": {"conditions": [{"dept": ["Science"], "school": ["Bluebird HS", ""]}]}}]
    ok_text = qwen_call("db", conditions={"dept": "science"})
    bad_text = qwen_call("db", conditions={"dept": "Science", "extra": 1})
    assert grade(ok_text, functions=[fn], truth=truth) == (True, "ok")
    assert grade(bad_text, functions=[fn], truth=truth) == (False, "wrong_value")


def test_normalize_string_matches_bfcl_rules():
    assert normalize_string("New York, NY") == normalize_string("newyork ny")
    assert normalize_string("it's") == 'it"s'


def test_schema_conversion():
    tool = to_openai_tool(TRIANGLE)["function"]["parameters"]
    assert tool["type"] == "object" and tool["properties"]["scale"]["type"] == "number"
    assert TRIANGLE["parameters"]["type"] == "dict"  # input not mutated
    fn = xlam_to_bfcl_function(
        {
            "name": "f",
            "parameters": {
                "q": {"type": "str", "description": "query"},
                "n": {"type": "int, optional", "default": 5},
                "ids": {"type": "List[int]"},
                "x": {"type": "SomethingElse"},
            },
        }
    )
    props = fn["parameters"]["properties"]
    assert [props[k]["type"] for k in "q n ids x".split()] == ["string", "integer", "array", "any"]
    assert fn["parameters"]["required"] == ["q", "ids", "x"]


def test_load_bfcl_and_xlam(tmp_path):
    question = {
        "id": "simple_0",
        "question": [[{"role": "user", "content": "area?"}]],
        "function": [TRIANGLE],
    }
    (tmp_path / "possible_answer").mkdir()
    (tmp_path / "BFCL_v3_simple.json").write_text(json.dumps(question) + "\n")
    (tmp_path / "possible_answer" / "BFCL_v3_simple.json").write_text(
        json.dumps({"id": "simple_0", "ground_truth": TRUTH}) + "\n"
    )
    (example,) = load_bfcl(tmp_path, "simple")
    assert example == Example(
        "simple_0", [{"role": "user", "content": "area?"}], [TRIANGLE], TRUTH, "simple"
    )

    tools = [
        {"name": "f", "parameters": {"q": {"type": "str"}, "n": {"type": "int", "default": 5}}}
    ]
    record = {
        "id": 7,
        "query": "hi",
        "tools": json.dumps(tools),
        "answers": '[{"name": "f", "arguments": {"q": "a"}}]',
    }
    path = tmp_path / "xlam.json"
    path.write_text(json.dumps([record]))
    (example,) = load_xlam(path)
    assert example.ground_truth == [{"f": {"q": ["a"], "n": ["", 5]}}]
    text = qwen_call("f", q="a", n=5)
    assert check_example(
        parse_tool_calls(text, example.functions), example.functions, example.ground_truth
    )[0]


def test_split_is_deterministic_and_disjoint():
    ids = [f"x{i}" for i in range(100)]
    train, val = split_ids(ids, 10, seed=0)
    assert (train, val) == split_ids(list(reversed(ids)), 10, seed=0)
    assert len(val) == 10 and not set(train) & set(val) and len(train) + len(val) == 100
