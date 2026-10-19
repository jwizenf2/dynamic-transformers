# Multi-step tool-call training data for path B (2026-10-19)

Survey for a training mix that teaches dependent chaining (call → `<tool_response>` → next call
that uses the result) while NESTful stays test-only. Fractions marked "sample" are estimates from
inspected rows (dependency heuristic: a later argument value appears in an earlier tool response
and nowhere in the user/system text). Exact-match contamination check (normalised 80-character
query prefixes, 1,818 NESTful queries): 0 hits in Hermes ×2, ToolACE, NesTools, the first 1,041
BUTTON rows, the TinyAgent test file and Seal-Tools train.

**Root cause confirmed:** Hermes v1 `func-calling.json` (1,893) and
`glaive-function-calling-5k.json` (5,209) have 0 assistant `<tool_call>` turns after a
`<tool_response>` within the same user turn. The SFT data never shows call → result → next call,
which matches the multi-turn NESTful result (98–100% of dialogues stop after one tool response).

| Dataset @ revision | License | Size | Dependent chains | Tool responses | Notes |
|---|---|---|---|---|---|
| ibm-research/nestful @f1ed646 | Apache-2.0 | 1,861 | all | no | The benchmark; HF split is named "train" but paper/repo treat it as the test set; no train split exists. Math part derived from MathQA test. Do not use. |
| Salesforce/APIGen-MT-5k @abc4a51 | CC-BY-NC-4.0, gated | 5,000 | ~94% (sample of 400); 49% have ≥2 back-to-back calls in a user turn | yes, executed (τ-bench) | Easy conversion; non-commercial; contaminates τ-bench. |
| BUTTONInstruct (GitHub PKU-Baichuan-MLSystemLab/BUTTON @47cb720) | CC-BY-4.0 | ~8k | ~59% (first 1,041 rows); mean 2.6 call steps | yes, simulated | Easy conversion. |
| Zichen1024/SAP-9k @0234b31 | Apache-2.0 | 9,644 | built for argument provenance (not sampled) | yes, executed | Built on BFCL v4 multi-turn and τ²-bench backends: contaminates BFCL. |
| squeeze-ai-lab/TinyAgent-dataset @07aae41 | MIT | 40k | 70% use `$k` output references (test file) | partly | Plan-style; macOS tools. |
| Team-ACE/ToolACE @6bda777 | Apache-2.0 | 11,300 | 302 rows with a call after a response | yes, simulated | Small usable subset. |
| Agent-Ark/Toucan-1.5M @0df3cf3 | Apache-2.0 | 1.5M | ~42% (sample of 60) | yes, live MCP | Very large; tools not re-executable offline. |
| casey-martin/Seal-Tools @d0fe224 | MIT | 12,022 | 501 (4.2%) | no | Call lists only. |
| NesTools (GitHub hhan1018/NesTools @fb617f3) | Apache-2.0 | 1,000 | ~830 | no | An eval set; calls only. |
| Nanbeige/ToolMind @8020ed1 | Apache-2.0 | 360k | 13/40 sampled | yes, simulated | Includes reprocessed APIGen-MT and BUTTON. |
| Hermes v1 @dae3e1d / Glaive v2 @e7f4b64 | Apache-2.0 | 7.1k / 113k | 0 in Hermes | yes | Current SFT data. |
| ToolBench @d56fdd8 | Apache-2.0 | ~126k | some, noisy (not measured) | yes (2023 RapidAPI) | APIs drifted. |
| Magnet, ToolFlow | — | — | — | — | No public data. |

Synthesis methods: RandomWorld (arXiv 2506.11045, EMNLP 2025) composes typed executable tools
into call DAGs and reports SoTA NESTful F1 after SFT, growing with data volume; its repo has no
license, so reimplement rather than reuse. APIGen-MT, SAP, ToolMind and Toucan use
executor-grounded multi-turn synthesis.

Prior work note: 2608.18171 fine-tuned Ouro-2.6B on Hermes v1 and reports NESTful win rate
0.190 → 0.371 (their metric), with no nested training data.

## Recommended mix (~5–10k)
1. Core, 3–5k: our own executed chains (RandomWorld-style) rendered natively in Hermes ChatML,
   2–6 dependent calls with real `<tool_response>`s. Not NESTful's 40 tool specs, its executable
   functions or MathQA test. Math chains from MathQA train only with renamed/different tools and
   near-duplicate removal against the 1,415 NESTful math queries (text + formula).
2. ~2.5k APIGen-MT-5k rows with ≥2 back-to-back calls (research use; flags τ-bench overlap).
3. ~2–3k BUTTONInstruct dependent rows, plus the 302 ToolACE chain rows. SAP-9k only if BFCL
   multi-turn is not an evaluation set.
