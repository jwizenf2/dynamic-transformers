"""Synthetic, executed multi-step tool-call conversations (path B training data).

Why: Hermes FC v1 contains no assistant tool call that follows a tool response within one user
turn (docs/research/chain_training_data_2026-10-19.md), so the SFT models never learn to chain.
This module generates conversations that do: a user request whose answer needs 2–6 tool calls,
where later calls take earlier results as arguments, rendered in the Hermes ChatML convention

    system (Hermes function-calling prompt with <tools>) → human request →
    gpt <tool_call>… → tool <tool_response>… → gpt <tool_call>… → … → gpt final answer.

Construction (in the spirit of RandomWorld, arXiv 2506.11045; no code reused):
* A typed library of executable tools in domains disjoint from NESTful's math subset: text,
  integer lists, calendar dates, unit conversion and lookups in fictional tables (invented
  cities, products, employees, so answers must come from tools, not world knowledge). No tool
  shares a name with NESTful's basic functions (checked by a unit test).
* A chain starts from literal values in the request. Each step picks a tool whose first
  parameter takes the previous step's output (so every step depends on the one before) and
  fills other parameters with fresh literals. With probability ``branch_p`` the chain starts
  with two independent calls in one assistant turn that a binary tool then combines.
* Every call is executed; the tool turn carries the real result. The final assistant turn
  states the last result. Unused distractor tools are added to the tool list.

The request text is template-built from per-tool phrases (no LLM paraphrase), which keeps the
data exact and reproducible; its wording is plainer than human requests.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# --- fictional lookup tables ------------------------------------------------------------------

_SYL = ["var", "len", "hol", "mir", "tas", "kor", "bel", "dun", "sel", "ran", "vok", "ith", "pel"]


def _names(n: int, seed: int, parts: int = 2) -> list[str]:
    rng = random.Random(seed)
    out: set[str] = set()
    while len(out) < n:
        out.add("".join(rng.choice(_SYL) for _ in range(parts)).capitalize())
    return sorted(out)


CITIES = _names(40, 1, 3)
COUNTRIES = _names(12, 2)
CURRENCIES = [f"{c[:2].upper()}{chr(65 + i)}" for i, c in enumerate(COUNTRIES)]  # unique codes
PRODUCTS = [
    f"{n} {k}"
    for n, k in zip(_names(30, 3), ["lamp", "kettle", "chair", "radio"] * 8, strict=False)
]
PEOPLE = _names(30, 4)
DEPARTMENTS = ["Billing", "Logistics", "Research", "Support", "Design", "Legal"]

_r = random.Random(5)
CITY_COUNTRY = {c: _r.choice(COUNTRIES) for c in CITIES}
COUNTRY_CAPITAL = {
    c: _r.choice([x for x in CITIES if CITY_COUNTRY[x] == c] or CITIES) for c in COUNTRIES
}
COUNTRY_CURRENCY = dict(zip(COUNTRIES, CURRENCIES, strict=True))
RATE_TO_USD = {c: round(_r.uniform(0.2, 3.0), 3) for c in CURRENCIES}
PRODUCT_PRICE = {p: round(_r.uniform(5, 400), 2) for p in PRODUCTS}
PRODUCT_STOCK = {p: _r.randint(0, 250) for p in PRODUCTS}
PERSON_DEPT = {p: _r.choice(DEPARTMENTS) for p in PEOPLE}
DEPT_MANAGER = {d: _r.choice(PEOPLE) for d in DEPARTMENTS}

# --- tools ------------------------------------------------------------------------------------

JSON_TYPE = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "list": "array",
    "date": "string",
}


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    params: tuple[tuple[str, str], ...]  # (name, type); the first one may take a prior output
    returns: str
    fn: Callable[..., Any]
    phrase: str  # request wording; {p} placeholders per parameter
    noun: str  # how a later step refers to this tool's output

    def spec(self) -> dict:
        props = {}
        for p, t in self.params:
            props[p] = {
                "type": JSON_TYPE[t],
                "description": f"{p} ({'YYYY-MM-DD' if t == 'date' else t})",
            }
            if t == "list":
                props[p]["items"] = {"type": "integer"}
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": [p for p, _ in self.params],
                },
            },
        }


def _date(s: str) -> dt.date:
    return dt.date.fromisoformat(s)


TOOLS: list[Tool] = [
    # text
    Tool(
        "reverse_text",
        "Reverse the characters of a text.",
        (("text", "str"),),
        "str",
        lambda text: text[::-1],
        "reverse {text}",
        "the reversed text",
    ),
    Tool(
        "uppercase_text",
        "Convert a text to upper case.",
        (("text", "str"),),
        "str",
        lambda text: text.upper(),
        "convert {text} to upper case",
        "the upper-case text",
    ),
    Tool(
        "count_vowels",
        "Count the vowels (a, e, i, o, u) in a text.",
        (("text", "str"),),
        "int",
        lambda text: sum(ch in "aeiouAEIOU" for ch in text),
        "count the vowels in {text}",
        "the vowel count",
    ),
    Tool(
        "text_length",
        "Number of characters in a text.",
        (("text", "str"),),
        "int",
        lambda text: len(text),
        "get the length of {text}",
        "the length",
    ),
    Tool(
        "repeat_text",
        "Repeat a text a number of times, separated by '-'.",
        (("text", "str"), ("times", "int")),
        "str",
        lambda text, times: "-".join([text] * times),
        "repeat {text} {times} times",
        "the repeated text",
    ),
    Tool(
        "join_texts",
        "Join two texts with a space.",
        (("first", "str"), ("second", "str")),
        "str",
        lambda first, second: f"{first} {second}",
        "join {first} and {second}",
        "the joined text",
    ),
    # integer lists
    Tool(
        "sort_values",
        "Sort a list of integers in ascending order.",
        (("values", "list"),),
        "list",
        lambda values: sorted(values),
        "sort {values}",
        "the sorted list",
    ),
    Tool(
        "sum_values",
        "Sum of a list of integers.",
        (("values", "list"),),
        "int",
        lambda values: sum(values),
        "add up {values}",
        "the total",
    ),
    Tool(
        "largest_value",
        "Largest integer in a list.",
        (("values", "list"),),
        "int",
        lambda values: max(values),
        "find the largest value in {values}",
        "the largest value",
    ),
    Tool(
        "keep_above",
        "Keep the integers in a list that are greater than a threshold.",
        (("values", "list"), ("threshold", "int")),
        "list",
        lambda values, threshold: [v for v in values if v > threshold],
        "keep the values in {values} that are above {threshold}",
        "the filtered list",
    ),
    Tool(
        "count_items",
        "Number of items in a list.",
        (("values", "list"),),
        "int",
        lambda values: len(values),
        "count the items in {values}",
        "the item count",
    ),
    Tool(
        "append_value",
        "Append an integer to a list.",
        (("values", "list"), ("value", "int")),
        "list",
        lambda values, value: [*values, value],
        "append {value} to {values}",
        "the extended list",
    ),
    # dates
    Tool(
        "add_days",
        "Date that is a number of days after a date.",
        (("date", "date"), ("days", "int")),
        "date",
        lambda date, days: (_date(date) + dt.timedelta(days=days)).isoformat(),
        "find the date {days} days after {date}",
        "that date",
    ),
    Tool(
        "day_of_week",
        "Weekday name of a date.",
        (("date", "date"),),
        "str",
        lambda date: _date(date).strftime("%A"),
        "get the weekday of {date}",
        "the weekday",
    ),
    Tool(
        "days_between",
        "Number of days from one date to another.",
        (("start", "date"), ("end", "date")),
        "int",
        lambda start, end: (_date(end) - _date(start)).days,
        "count the days from {start} to {end}",
        "the number of days",
    ),
    Tool(
        "month_number",
        "Month (1-12) of a date.",
        (("date", "date"),),
        "int",
        lambda date: _date(date).month,
        "get the month number of {date}",
        "the month number",
    ),
    # units
    Tool(
        "celsius_to_fahrenheit",
        "Convert degrees Celsius to Fahrenheit (2 decimals).",
        (("celsius", "float"),),
        "float",
        lambda celsius: round(celsius * 9 / 5 + 32, 2),
        "convert {celsius} degrees Celsius to Fahrenheit",
        "the Fahrenheit value",
    ),
    Tool(
        "km_to_miles",
        "Convert kilometres to miles (2 decimals).",
        (("km", "float"),),
        "float",
        lambda km: round(km * 0.621371, 2),
        "convert {km} km to miles",
        "the distance in miles",
    ),
    Tool(
        "percent_of",
        "A percentage of a value (2 decimals).",
        (("value", "float"), ("percent", "float")),
        "float",
        lambda value, percent: round(value * percent / 100, 2),
        "take {percent} percent of {value}",
        "that amount",
    ),
    # lookups in fictional tables
    Tool(
        "city_country",
        "Country a city is in.",
        (("city", "str"),),
        "str",
        lambda city: CITY_COUNTRY.get(city, "unknown"),
        "find which country {city} is in",
        "that country",
    ),
    Tool(
        "country_capital",
        "Capital city of a country.",
        (("country", "str"),),
        "str",
        lambda country: COUNTRY_CAPITAL.get(country, "unknown"),
        "look up the capital of {country}",
        "the capital",
    ),
    Tool(
        "country_currency",
        "Currency code of a country.",
        (("country", "str"),),
        "str",
        lambda country: COUNTRY_CURRENCY.get(country, "unknown"),
        "look up the currency of {country}",
        "the currency",
    ),
    Tool(
        "rate_to_usd",
        "Exchange rate of a currency to USD.",
        (("currency", "str"),),
        "float",
        lambda currency: RATE_TO_USD.get(currency, 0.0),
        "get the USD rate of {currency}",
        "the exchange rate",
    ),
    Tool(
        "product_price",
        "Price of a product in USD.",
        (("product", "str"),),
        "float",
        lambda product: PRODUCT_PRICE.get(product, 0.0),
        "look up the price of {product}",
        "the price",
    ),
    Tool(
        "product_stock",
        "Units of a product in stock.",
        (("product", "str"),),
        "int",
        lambda product: PRODUCT_STOCK.get(product, 0),
        "check the stock of {product}",
        "the stock count",
    ),
    Tool(
        "employee_department",
        "Department an employee works in.",
        (("employee", "str"),),
        "str",
        lambda employee: PERSON_DEPT.get(employee, "unknown"),
        "find the department of {employee}",
        "that department",
    ),
    Tool(
        "department_manager",
        "Manager of a department.",
        (("department", "str"),),
        "str",
        lambda department: DEPT_MANAGER.get(department, "unknown"),
        "find the manager of {department}",
        "the manager",
    ),
]
BY_NAME = {t.name: t for t in TOOLS}

# Types a tool's first parameter accepts from a previous output (a city name is a str, etc.).
_LOOKUP_FIRST = {
    "city_country": CITIES,
    "product_price": PRODUCTS,
    "product_stock": PRODUCTS,
    "employee_department": PEOPLE,
}


def _literal(t: str, rng: random.Random, name: str = "") -> Any:
    if t == "int":
        return rng.randint(2, 9) if name in ("times", "days") else rng.randint(1, 60)
    if t == "float":
        return round(rng.uniform(1, 250), 1)
    if t == "list":
        return [rng.randint(1, 99) for _ in range(rng.randint(3, 7))]
    if t == "date":
        return (dt.date(2024, 1, 1) + dt.timedelta(days=rng.randint(0, 900))).isoformat()
    return rng.choice(["harbor", "lantern", "meadow", "copper", "violet", "summit", "pebble"])


def _first_literal(tool: Tool, rng: random.Random) -> Any:
    """A valid literal for a tool's first parameter (lookups need an existing table key)."""
    if tool.name in _LOOKUP_FIRST:
        return rng.choice(_LOOKUP_FIRST[tool.name])
    if tool.name in ("country_capital", "country_currency"):
        return rng.choice(COUNTRIES)
    if tool.name == "rate_to_usd":
        return rng.choice(CURRENCIES)
    if tool.name == "department_manager":
        return rng.choice(DEPARTMENTS)
    p, t = tool.params[0]
    return _literal(t, rng, p)


def _literal_args(tool: Tool, rng: random.Random) -> dict:
    args = {tool.params[0][0]: _first_literal(tool, rng)}
    for p, t in tool.params[1:]:
        args[p] = _literal(t, rng, p)
    return args


def _followers(prev: Tool, value: Any) -> list[Tool]:
    """Tools whose first parameter can take ``value`` (the previous output) meaningfully."""
    out = []
    for t in TOOLS:
        first = t.params[0][1]
        if first != prev.returns and not (first == "float" and prev.returns == "int"):
            continue
        if t.name in _LOOKUP_FIRST and value not in _LOOKUP_FIRST[t.name]:
            continue
        if t.name in ("country_capital", "country_currency") and value not in COUNTRIES:
            continue
        if t.name == "rate_to_usd" and value not in CURRENCIES:
            continue
        if t.name == "department_manager" and value not in DEPARTMENTS:
            continue
        if t.returns == "list" and isinstance(value, list) and t.name == "keep_above" and not value:
            continue
        if prev.returns == "list" and isinstance(value, list) and not value:
            continue
        out.append(t)
    return out


def _producers(t: str) -> list[Tool]:
    """Single-parameter tools returning type ``t`` (starts of a branch)."""
    return [x for x in TOOLS if x.returns == t and len(x.params) == 1]


# Binary tools whose two parameters share a type that single-parameter tools produce.
_COMBINERS = [
    t
    for t in TOOLS
    if len(t.params) == 2 and t.params[0][1] == t.params[1][1] and _producers(t.params[0][1])
]


@dataclass
class Step:
    tool: Tool
    args: dict
    result: Any
    turn: int
    deps: dict = field(default_factory=dict)  # param -> index of the step whose output it uses


def make_chain(rng: random.Random, length: int, branch_p: float = 0.2) -> list[Step]:
    """A dependent chain of ``length`` executed calls (2 ≤ length ≤ 6)."""
    steps: list[Step] = []
    turn = 0
    if rng.random() < branch_p and length >= 3:
        # two independent calls in one turn, combined by a binary tool
        comb = rng.choice(_COMBINERS)
        for _ in range(2):
            tool = rng.choice(_producers(comb.params[0][1]))
            args = _literal_args(tool, rng)
            steps.append(Step(tool, args, tool.fn(**args), turn))
        turn += 1
        args = {comb.params[0][0]: steps[0].result, comb.params[1][0]: steps[1].result}
        steps.append(
            Step(comb, args, comb.fn(**args), turn, {comb.params[0][0]: 0, comb.params[1][0]: 1})
        )
        turn += 1
    else:
        tool = rng.choice(TOOLS)
        args = _literal_args(tool, rng)
        steps.append(Step(tool, args, tool.fn(**args), turn))
        turn += 1
    while len(steps) < length:
        prev = steps[-1]
        cands = [t for t in _followers(prev.tool, prev.result) if t.name != prev.tool.name]
        if not cands:
            break
        tool = rng.choice(cands)
        args = {tool.params[0][0]: prev.result}
        for p, t in tool.params[1:]:
            args[p] = _literal(t, rng, p)
        steps.append(Step(tool, args, tool.fn(**args), turn, {tool.params[0][0]: len(steps) - 1}))
        turn += 1
    return steps


def _fmt(v: Any) -> str:
    if isinstance(v, str):
        return f'"{v}"'
    return json.dumps(v)


def request_text(steps: list[Step]) -> str:
    parts = []
    for s in steps:
        filled = {}
        for p, _ in s.tool.params:
            filled[p] = steps[s.deps[p]].tool.noun if p in s.deps else _fmt(s.args[p])
        parts.append(s.tool.phrase.format(**filled))
    if steps[0].turn == steps[1].turn:  # branch: two independent starts
        text = f"First {parts[0]}, and separately {parts[1]}. Then {', then '.join(parts[2:])}."
    else:
        text = f"Please {parts[0]}" + "".join(f", then {p}" for p in parts[1:]) + "."
    return text[0].upper() + text[1:] + " What is the final result?"


def render(steps: list[Step], rng: random.Random, system_template: str, n_distractors=(2, 5)):
    """Hermes-style conversation: list of {"from", "value"} turns."""
    used = {s.tool.name for s in steps}
    extra = rng.sample([t for t in TOOLS if t.name not in used], rng.randint(*n_distractors))
    tools = [s.tool for s in steps if s.tool.name in used]
    tools = list({t.name: t for t in tools + extra}.values())
    rng.shuffle(tools)
    system = system_template.format(tools=json.dumps([t.spec() for t in tools], ensure_ascii=False))
    conv = [{"from": "system", "value": system}, {"from": "human", "value": request_text(steps)}]
    for turn in sorted({s.turn for s in steps}):
        group = [s for s in steps if s.turn == turn]
        calls = "\n".join(
            "<tool_call>\n"
            + json.dumps({"name": s.tool.name, "arguments": s.args}, ensure_ascii=False)
            + "\n</tool_call>"
            for s in group
        )
        conv.append({"from": "gpt", "value": calls})
        responses = "\n".join(
            "<tool_response>\n"
            + json.dumps({"name": s.tool.name, "content": s.result}, ensure_ascii=False)
            + "\n</tool_response>"
            for s in group
        )
        conv.append({"from": "tool", "value": responses})
    conv.append({"from": "gpt", "value": f"The final result is {_fmt(steps[-1].result)}."})
    return conv


def generate(n: int, seed: int, system_template: str, lengths=(2, 6), branch_p: float = 0.2):
    """``n`` conversations ``{"id", "conversations", "n_calls", "turns"}``; deterministic."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        steps = make_chain(rng, rng.randint(*lengths), branch_p)
        if len(steps) < 2:
            continue
        out.append(
            {
                "id": f"chain_{seed}_{len(out)}",
                "conversations": render(steps, rng, system_template),
                "n_calls": len(steps),
                "turns": len({s.turn for s in steps}),
            }
        )
    return out
