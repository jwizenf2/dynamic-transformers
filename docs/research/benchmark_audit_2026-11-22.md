# Benchmark audit (2026-11-22)

Independent audit of the evaluation pipeline (gold chains, grading, limits, parsing, tools,
leakage, GSM8K conversion) on the five main chain v3 seed 0 sets: NESTful [0, 240), [240, 720),
[720, 1200), GSM8K test (800), MathQA clean (292). Scripts (not committed, scratch):
gold replay, re-grading of stored values, dialogue replay from stored texts (94% reproduce the
stored value and call count exactly; the rest differ only because stored texts are cut at 300
characters). Grading code was not changed: `answers_match` is NESTful's own scorer and all
published numbers use it.

## Findings

1. MathQA golds carry float noise (stored executed floats, e.g. `mathqa-validation-567`
   28.000000000000025, -503 10.000000000000002). With NESTful's rule (round the prediction to all
   of the gold's decimals) exact answers are rejected: 4 runs at d2 and 4 at d4 of 292; MathQA
   accuracy at both depths is ~1.4 points low (d2 68 → 72, d4 105 → 109 correct). "4 helps"
   unchanged (46); "4 hurts" 9 → 10.
2. Same effect on long-decimal NESTful golds: 4 runs at d2 and 5 at d4 of 1,200 (< 0.5 point).
   GSM8K is unaffected (golds are the `####` values).
3. The rounding rule also accepts approximate answers (most golds have one decimal, so ±0.05),
   e.g. km/h → m/s via × 0.2778. Passing only by rounding: NESTful 10/1,200 at d2, 16/1,200 at d4
   (NESTful [0, 240): 3 vs 7). Under strict grading "4 helps" is essentially unchanged by slice
   (33 → 33, 84 → 83, 85 → 87); "4 hurts" rises slightly. A disclosure point (NESTful's scorer).
4. Vendored NESTful tools: `sqrt` is `math.isqrt` (truncates; TypeError on floats), `log` returns
   an int. One MathQA gold is wrong because of it (-3468: 16.0 instead of 22.63); 3 dialogues
   looped on sqrt TypeErrors. One MathQA gold is wrong at source (-4061, 9.5 = arithmetic mean
   instead of the harmonic 9.47; d4 credited only by rounding).
5. NESTful duplicates: 2 identical-query pairs within [240, 720), 2 across [240, 720) and
   [720, 1200); 66 of 480 [720, 1200) queries share a near-identical template with [0, 720)
   (NESTful's own rewrites with other numbers).
6. GSM8K conversion can link a constant to an earlier equal-valued result (e.g. `gsm8k-test-956`);
   affects chain-length/dependency labels and plan wording, not answers. A 30-task spot check
   against the source annotations found no wrong value or operation.

## Clean

All 2,292 gold chains execute to their gold; stored grades equal re-grading (0 disagreements);
identical prompts, caches, decoding, stop tokens and limits at every depth; turn-limit hits are
slightly more common at d2 than d4 (e.g. NESTful [0, 240) 15 vs 9; none favour d4); 256-token
caps 0–4 per depth, none in a correct run; malformed tool calls ~0 at d2/d4; echoes/identity last
calls 0–3 per depth; no overlap of any test query with the v3 SFT data (max template similarity
< 0.4 for NESTful and GSM8K).

## Reading

No issue changes the 2-vs-4 comparisons or the "4 helps" sets by more than a few requests; the
MathQA levels are ~1.4 points low at both depths. For the paper: keep NESTful's scorer as primary,
report a sensitivity row (relative tolerance 1e-6 plus strict grading), and state the tool
semantics, the two bad MathQA golds and the NESTful duplicates as limitations.
