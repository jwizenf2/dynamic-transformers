# NESTful files (vendored)

Copied unmodified from https://github.com/IBM/NESTFUL at commit
`fc2c4123e73500a56185a5fb354f05d1c8b4890c` (Apache-2.0, see `LICENSE`):

| File | Upstream path | SHA256 |
|---|---|---|
| `basic_functions.py` | `data_v2/executable_functions/basic_functions.py` | `db8048658f7ccd930c691a36dd0831a7d46dbffa2a275932f60af43792128212` |
| `icl_examples.json` | `src/icl_examples.json` | `fa10032a95bb1d1177f25ddbb6bc102140cac0b41e4d19ea78eecc8b249b18e7` |

Vendored because ai-lab jobs can reach Hugging Face but not raw GitHub. The samples themselves
come from the Hugging Face copy (`ibm-research/nestful`, pinned in `scripts/lab/ouro_nestful.py`);
its 1,416 math-subset rows were checked to be identical to the GitHub `nestful_data.jsonl`
(the 108 differing fields are gold-answer stringifications in non-math samples, which are not used).
