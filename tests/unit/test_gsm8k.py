"""GSM8K → NESTful-format conversion (no downloads: inline examples)."""

from __future__ import annotations

import random
from types import SimpleNamespace

import pytest

from dht.toolcall.gsm8k import (
    build_task,
    connected,
    final_answer,
    question_numbers,
    solution_calls,
)

QUESTION = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in "
    "May. How many clips did Natalia sell altogether in April and May?"
)
SOLUTION = (
    "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\n"
    "Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72"
)


def test_references_and_literals():
    calls = solution_calls(SOLUTION, QUESTION)
    assert [c["name"] for c in calls] == ["divide", "add"]
    assert calls[0]["arguments"] == {"arg_0": 48, "arg_1": 2}
    # 24 is not in the question and is the first annotation's result: a reference
    assert calls[1]["arguments"] == {"arg_0": 48, "arg_1": "$var_1.result$"}
    assert [c["label"] for c in calls] == ["$var_1", "$var_2"]


def test_compound_expression_post_order():
    calls = solution_calls("x <<2*3+4=10>>10\n#### 10", "Numbers 2, 3 and 4.")
    assert [c["name"] for c in calls] == ["multiply", "add"]
    assert calls[1]["arguments"] == {"arg_0": "$var_1.result$", "arg_1": 4}


def test_question_number_stays_literal_even_if_an_earlier_result():
    # 5 is both an earlier result and in the question: literal (the question is the input)
    calls = solution_calls("<<2+3=5>>5 then <<5*10=50>>50\n#### 50", "He has 2, 3 and 5 and 10.")
    assert calls[1]["arguments"]["arg_0"] == 5


def test_bare_number_annotation_adds_no_call_and_aliases():
    sol = "<<6*7=42>>42, so <<42=42>>42, then <<42-2=40>>40\n#### 40"
    calls = solution_calls(sol, "6 and 7")
    assert [c["name"] for c in calls] == ["multiply", "subtract"]
    assert calls[1]["arguments"]["arg_0"] == "$var_1.result$"


def test_numbers_and_answer_parsing():
    assert question_numbers("It costs $1,200 and 3.5 more") == {1200.0, 3.5}
    assert final_answer("...\n#### 1,200") == 1200.0
    with pytest.raises(ValueError):
        final_answer("no answer")
    with pytest.raises(ValueError):
        solution_calls("<<abs(3)=3>>\n#### 3", "")


def test_build_task_executes_to_the_answer():
    spec = {
        n: {
            "name": n,
            "parameters": {"arg_0": {"type": "float"}, "arg_1": {"type": "float"}},
            "output_parameters": {"result": {"type": "float"}},
        }
        for n in ("add", "subtract", "multiply", "divide", "inverse")
    }
    functions = SimpleNamespace(
        add=lambda a, b: a + b,
        subtract=lambda a, b: a - b,
        multiply=lambda a, b: a * b,
        divide=lambda a, b: a / b,
        inverse=lambda a: 1 / a,
    )
    task, why = build_task(
        {"question": QUESTION, "answer": SOLUTION}, spec, functions, random.Random(0)
    )
    assert why == "ok" and task["answer"] == 72.0
    bad, why = build_task(
        {"question": QUESTION, "answer": SOLUTION.replace("#### 72", "#### 70")},
        spec,
        functions,
        random.Random(0),
    )
    assert bad is None and why == "mismatch"


def test_connected_rejects_dead_intermediate_results():
    ok = solution_calls(SOLUTION, QUESTION)
    assert connected(ok)
    # 2*3 = 6 is done in prose, so the next annotation starts from an unannotated 6
    dead = solution_calls("<<200/100=2>>2, so 2*3=6 g, <<6*7=42>>42\n#### 42", "200 100 3 7")
    assert not connected(dead)


def test_gold_answer_is_the_clean_final_answer():
    spec = {
        n: {
            "name": n,
            "parameters": {"arg_0": {"type": "float"}, "arg_1": {"type": "float"}},
            "output_parameters": {"result": {"type": "float"}},
        }
        for n in ("add", "subtract", "multiply", "divide", "inverse")
    }
    functions = SimpleNamespace(
        add=lambda a, b: a + b,
        subtract=lambda a, b: a - b,
        multiply=lambda a, b: a * b,
        divide=lambda a, b: a / b,
        inverse=lambda a: 1 / a,
    )
    row = {"question": "9 of 14 parts of 84", "answer": "<<9/14*84=34>>34\n#### 34"}
    task, why = build_task(row, spec, functions, random.Random(0))
    assert why == "ok" and repr(task["answer"]) == "34.0"
