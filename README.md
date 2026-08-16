# AI-style to natural prose: seq2seq proof of concept

This experiment tests one narrow hypothesis: can a small pretrained sequence-to-sequence model learn to rewrite formulaic prose more naturally without changing its meaning?

The project does **not** optimize against AI detectors. Its primary gates are human preference and meaning preservation.

## Run in Google Colab

1. Open `notebooks/train_and_evaluate.ipynb` in Colab and select a GPU runtime.
2. Upload a UTF-8 CSV named `pairs.csv`. You can use `data/example_pairs.csv` for a smoke test.
3. Run all cells. The notebook trains `google/flan-t5-small`, evaluates the untouched-input baseline and fine-tuned model, and saves the model plus predictions in `artifacts/`.

The CSV schema is:

| column | required | meaning |
|---|---:|---|
| `source` | yes | Formulaic/AI-style input text |
| `target` | yes | Human-written natural rewrite |
| `group_id` | yes | Shared origin/topic ID used to prevent leakage |

Use at least a few hundred carefully reviewed pairs for the actual experiment. The included examples only verify that the pipeline runs.

## Experimental contract

- Split by `group_id`, never by individual row.
- Keep the test set untouched until model choices are fixed.
- Compare against the identity baseline (returning `source` unchanged).
- Report semantic similarity and preservation of numbers/proper nouns alongside SARI.
- Inspect low-similarity outputs manually; aggregate metrics cannot establish meaning preservation.
- Decide success with a blind human comparison on the held-out test set.

Suggested success gates for the first real dataset:

- Human raters prefer the model output to the source for naturalness on at least 60% of examples.
- At least 95% of outputs preserve the original meaning according to human review.
- No systematic loss or invention of names, numbers, negation, or factual claims.

## Files

- `notebooks/train_and_evaluate.ipynb`: self-contained Colab workflow
- `data/example_pairs.csv`: smoke-test data, not a meaningful training corpus
- `requirements.txt`: pinned dependency ranges
