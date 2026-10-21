import json
import math
import random
from pathlib import Path

import pytest

from dht.toolcall.mathqa import (
    CORE_TOOLS,
    build_task,
    formula_calls,
    leaf_value,
    levels,
    normalize,
    option_value,
    render_dialogue,
    tool_list,
)
from dht.toolcall.nestful import answers_match, execute, load_functions, run_dialogue

VENDORED = Path(__file__).resolve().parents[2] / "third_party" / "nestful"
CATALOG = {t["name"]: t for t in json.loads((VENDORED / "tool_specs.json").read_text())}
FUNCTIONS = load_functions(VENDORED / "basic_functions.py")
SYSTEM = "tools: {tools}"

# NESTful sample "Gary drove from point A to point B ..." and its MathQA formula.
GARY = "multiply(60, divide(multiply(110, 5), subtract(110, 60)))"
GARY_GOLD = [
    {"name": "multiply", "label": "$var_1", "arguments": {"arg_0": 110, "arg_1": 5}},
    {"name": "subtract", "label": "$var_2", "arguments": {"arg_0": 110, "arg_1": 60}},
    {
        "name": "divide",
        "label": "$var_3",
        "arguments": {"arg_0": "$var_1.result$", "arg_1": "$var_2.result$"},
    },
    {"name": "multiply", "label": "$var_4", "arguments": {"arg_0": 60, "arg_1": "$var_3.result$"}},
]


def test_formula_calls_reproduces_nestful_gold():
    assert formula_calls(GARY) == GARY_GOLD


def test_leaves_and_constants():
    assert leaf_value("const_100") == 100 and isinstance(leaf_value("const_100"), int)
    assert leaf_value("const_0_25") == 0.25
    assert leaf_value("const_3_6") == 3.6
    assert leaf_value("const_pi") == math.pi
    assert leaf_value("2/3") == pytest.approx(2 / 3)
    calls = formula_calls("divide(90, subtract(const_1, divide(2, 3)))")
    assert [c["name"] for c in calls] == ["divide", "subtract", "divide"]
    assert calls[1]["arguments"] == {"arg_0": 1, "arg_1": "$var_1.result$"}
    with pytest.raises(ValueError):
        formula_calls("42")


def test_option_value_formats():
    opts = "a ) rs . 400 , b ) rs . 300 , c ) 1 : 20 , d ) - 49 , e ) 2,500"
    assert option_value(opts, "a") == 400.0
    assert option_value(opts, "c") == 0.05
    assert option_value(opts, "d") == -49.0
    assert option_value(opts, "e") == 2500.0
    assert option_value("a ) none of these , b ) 5", "a") is None


def test_tool_list_contains_gold_and_core():
    rng = random.Random(0)
    for _ in range(50):
        tools = tool_list({"power", "sqrt"}, CATALOG, rng)
        names = [t["name"] for t in tools]
        assert {"power", "sqrt", *CORE_TOOLS} <= set(names)
        assert len(names) == len(set(names)) and 7 <= len(names) <= 12


def test_levels_group_independent_calls():
    assert levels(GARY_GOLD) == [1, 1, 2, 3]


@pytest.mark.parametrize("mode", ["literal", "ref"])
def test_rendered_dialogue_replays_to_the_gold_answer(mode):
    tools = tool_list({"multiply", "subtract", "divide"}, CATALOG, random.Random(1))
    conv = render_dialogue("Gary ...", GARY_GOLD, tools, FUNCTIONS, SYSTEM, args=mode)
    assert [t["from"] for t in conv] == ["system", "human", *["gpt", "tool"] * 3, "gpt"]
    assert conv[-1]["value"] == "The final result is 660.0."
    # Replaying the assistant turns through the multi-turn executor gives the same answer.
    turns = iter(t["value"] for t in conv if t["from"] == "gpt")

    def assistant(_feedback):
        text = next(turns)
        return [json.loads(b.split("</tool_call>")[0]) for b in text.split("<tool_call>")[1:]]

    res = run_dialogue(assistant, tools, FUNCTIONS)
    assert res["ok"] and res["errors"] == 0 and res["turns"] == 3
    assert answers_match(res["value"], 660.0)


def test_build_task_filters_and_keeps():
    row = {
        "Problem": "gary drove from point a to point b at 60 km / h ...",
        "annotated_formula": GARY,
        "options": "a ) 600 , b ) 660 , c ) 700 , d ) 720 , e ) 800",
        "correct": "b",
        "category": "physics",
    }
    original = {normalize(row["Problem"]): "Gary drove from point A to point B at 60 Km/h ..."}
    task, why = build_task(row, CATALOG, FUNCTIONS, random.Random(0), original)
    assert why == "ok" and task["answer"] == 660.0
    assert task["query"].startswith("Gary drove") and task["calls"] == GARY_GOLD
    ok, value = execute(task["calls"], task["tools"], FUNCTIONS)
    assert ok and answers_match(value, task["answer"])
    assert build_task({**row, "correct": "a"}, CATALOG, FUNCTIONS, random.Random(0))[1] == (
        "mismatch"
    )
    assert (
        build_task({**row, "annotated_formula": "add(1, 2)"}, CATALOG, FUNCTIONS, random.Random(0))[
            1
        ]
        == "single"
    )
    unknown = {**row, "annotated_formula": "volume_cone(add(1, 2), 3)"}
    assert (
        build_task(
            unknown,
            {k: v for k, v in CATALOG.items() if k != "volume_cone"},
            FUNCTIONS,
            random.Random(0),
        )[1]
        == "op"
    )


def test_near_duplicate_index_ignores_numbers():
    from dht.toolcall.mathqa import NearDuplicateIndex

    idx = NearDuplicateIndex(
        ["A grocer has a sale of Rs. 6435 and Rs. 6927 for 2 months. Find the average sale."]
    )
    rewrite = "A grocer has a sale of Rs. 5000 and Rs. 7000 for 2 months. Find the average sale."
    assert idx.max_similarity(rewrite) == 1.0
    assert (
        idx.max_similarity("A train 120 m long crosses a pole in 6 seconds. Find its speed.") < 0.3
    )


def test_near_duplicate_index_catches_bare_expressions():
    from dht.toolcall.mathqa import NearDuplicateIndex

    idx = NearDuplicateIndex(["63 + 5 * 12 / (180/3) = ?"])
    assert idx.max_similarity("90 + 5 * 12 / (180/3) = ?") == 1.0
    assert idx.max_similarity("|5 - 2| - |6 - 9| = ?") < 0.3


def test_same_computation_index():
    from dht.toolcall.mathqa import SameComputationIndex

    ref = formula_calls("subtract(multiply(divide(750, 18), 39), 750)")
    idx = SameComputationIndex([ref])
    assert idx.matches(formula_calls("subtract(multiply(divide(300, 18), 39), 300)"))
    assert not idx.matches(formula_calls("subtract(multiply(divide(300, 17), 41), 300)"))
    assert not idx.matches(formula_calls("add(multiply(divide(300, 18), 39), 300)"))
