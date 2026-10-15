import json

import pytest

from dht.toolcall.nestful import (
    answers_match,
    basic_function_names,
    dependency_depth,
    exact_match,
    execute,
    load_functions,
    load_nestful,
    parse_sequence,
)

FUNCTIONS = """
import math


def add(arg_0, arg_1):
    return arg_0 + arg_1


def multiply(arg_0, arg_1):
    return arg_0 * arg_1


def divide(arg_0, arg_1):
    return arg_0 / arg_1


def power(base, exponent):
    return base ** exponent


def log(x, base=math.e):
    return int(math.log(x, base))


def factorial(n):
    return math.factorial(n)
"""


def _tool(name, params):
    return {
        "name": name,
        "description": name,
        "parameters": {p: {"type": "int or float"} for p in params},
        "output_parameters": {"result": {"type": "int or float"}},
    }


TOOLS = [
    _tool("add", ["arg_0", "arg_1"]),
    _tool("multiply", ["arg_0", "arg_1"]),
    _tool("divide", ["arg_0", "arg_1"]),
    _tool("power", ["arg_0", "arg_1"]),
    _tool("log", ["arg_0", "arg_1"]),
    _tool("factorial", ["arg_0"]),
]
# ((25 / 100) * 80) + 20 = 40.0: a chain of depth 3.
CHAIN = [
    {"name": "divide", "label": "$var_1", "arguments": {"arg_0": 25, "arg_1": 100}},
    {"name": "multiply", "label": "$var_2", "arguments": {"arg_0": "$var_1.result$", "arg_1": 80}},
    {"name": "add", "label": "$var_3", "arguments": {"arg_0": 20, "arg_1": "$var_2.result$"}},
]


@pytest.fixture
def functions(tmp_path):
    path = tmp_path / "basic_functions.py"
    path.write_text(FUNCTIONS)
    return load_functions(path)


def test_dependency_depth_counts_chains_not_calls():
    assert dependency_depth(CHAIN) == 3
    parallel = [
        {"name": "add", "label": "$var_1", "arguments": {"arg_0": 1, "arg_1": 2}},
        {"name": "add", "label": "$var_2", "arguments": {"arg_0": 3, "arg_1": 4}},
        {
            "name": "multiply",
            "label": "$var_3",
            "arguments": {"arg_0": "$var_1.result$", "arg_1": "$var_2.result$"},
        },
    ]
    assert dependency_depth(parallel) == 2
    assert dependency_depth(parallel[:2]) == 1
    # '$var1' labels and unterminated references occur in model output and in a few gold rows.
    odd = [
        {"name": "add", "label": "$var1", "arguments": {"arg_0": 1, "arg_1": 2}},
        {"name": "add", "label": "$var2", "arguments": {"arg_0": "$var1.result", "arg_1": 1}},
    ]
    assert dependency_depth(odd) == 2


def test_execute_resolves_references(functions):
    assert execute(CHAIN, TOOLS, functions) == (True, 40.0)
    swapped = [dict(CHAIN[0]), dict(CHAIN[1]), dict(CHAIN[2])]
    swapped[0] = {**CHAIN[0], "arguments": {"arg_0": 100, "arg_1": 25}}
    ok, value = execute(swapped, TOOLS, functions)
    assert ok and value == 340.0


@pytest.mark.parametrize(
    ("calls", "reason"),
    [
        ([], "no calls"),
        ([{"name": "subtract", "label": "$var_1", "arguments": {}}], "unknown function"),
        (
            [{"name": "add", "label": "$var_1", "arguments": {"arg_0": 1, "x": 2}}],
            "unknown arguments",
        ),
        (
            [
                {
                    "name": "add",
                    "label": "$var_1",
                    "arguments": {"arg_0": "$var_9.result$", "arg_1": 1},
                }
            ],
            "unresolved reference",
        ),
        (
            [{"name": "add", "label": "$var_1", "arguments": {"arg_0": "two", "arg_1": 1}}],
            "non-numeric",
        ),
        (
            [{"name": "divide", "label": "$var_1", "arguments": {"arg_0": 1, "arg_1": 0}}],
            "ZeroDivision",
        ),
        ([{"name": "factorial", "label": "$var_1", "arguments": {"arg_0": 10**9}}], "out of range"),
        (
            [{"name": "power", "label": "$var_1", "arguments": {"arg_0": 10, "arg_1": 10**6}}],
            "out of range",
        ),
        (
            [{"name": "add", "label": "$var_1", "arguments": {"arg_0": True, "arg_1": 1}}],
            "out of range",
        ),
        ([{"name": "log", "label": "$var_1", "arguments": {"arg_1": 10}}], "missing one"),
    ],
)
def test_execute_failures(functions, calls, reason):
    ok, why = execute(calls, TOOLS, functions)
    assert not ok and reason in why


def test_execute_uses_defaults_for_trailing_parameters(functions):
    calls = [{"name": "log", "label": "$var_1", "arguments": {"arg_0": 1000}}]
    assert execute(calls, TOOLS, functions) == (True, 6)  # int(ln 1000)
    calls[0]["arguments"]["arg_1"] = 10
    assert execute(calls, TOOLS, functions)[1] in (2, 3)  # int(log10(1000)) under float error


def test_answers_match_follows_nestful_rounding():
    assert answers_match(40.0001, 40.0)  # rounded to the gold's one decimal
    assert answers_match(1 / 3, 0.333)
    assert not answers_match(0.34, 0.333)
    assert answers_match(40, 40.0)
    assert not answers_match(True, 1.0)
    assert answers_match(1.0000000001e-05, 1e-05)
    assert answers_match((1, 2), [1, 2])


def test_exact_match_renumbers_labels():
    relabeled = [
        {**c, "label": c["label"].replace("var_", "v_")} for c in CHAIN
    ]  # labels differ but references still point at positions 0 and 1
    relabeled[1] = {**relabeled[1], "arguments": {"arg_0": "$v_1.result$", "arg_1": 80}}
    relabeled[2] = {**relabeled[2], "arguments": {"arg_0": 20, "arg_1": "$v_2.result$"}}
    assert not exact_match(relabeled, CHAIN)  # "$v_1" is not a var reference
    renamed = [dict(c) for c in CHAIN]
    renamed[0] = {**CHAIN[0], "label": "$var1"}
    renamed[1] = {**CHAIN[1], "arguments": {"arg_0": "$var1.result", "arg_1": 80}}
    assert exact_match(renamed, CHAIN)
    assert not exact_match(CHAIN[::-1], CHAIN)
    assert not exact_match([{**CHAIN[0], "arguments": {"arg_0": [1]}}], CHAIN[:1])


def test_parse_sequence():
    text = json.dumps(CHAIN) + "\n\nextra"
    assert parse_sequence(text) == CHAIN
    assert parse_sequence('[{"name": "add", "arguments": {"arg_0": 1}}]') == [
        {"name": "add", "label": None, "arguments": {"arg_0": 1}}
    ]
    assert parse_sequence("no list here") == []
    assert parse_sequence("[{broken") == []


def test_load_nestful_and_function_names(tmp_path):
    row = {
        "sample_id": "a",
        "input": "q",
        "output": CHAIN,
        "tools": TOOLS,
        "gold_answer": 40.0,
    }
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(row) + "\n\n")
    (e,) = load_nestful(path)
    assert (e.id, e.query, e.calls, e.answer) == ("a", "q", CHAIN, 40.0)
    fpath = tmp_path / "f.py"
    fpath.write_text(FUNCTIONS)
    assert basic_function_names(fpath) == {"add", "multiply", "divide", "power", "log", "factorial"}
