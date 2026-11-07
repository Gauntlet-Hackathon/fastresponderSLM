# Our model against Opus 5.5

The question the hackathon asks: does a small specialist beat a general-purpose frontier model on
the job it was built for? We compare our tagger with Opus 5.5 prompted three ways, on the same rows,
with the same scorer.

## The three Opus conditions

| Condition | What Opus gets |
| --- | --- |
| `naive` | One short user message naming the fields and their allowed values. No definitions, no examples, no enforced schema. What a busy person would type. |
| `structured` | A careful system prompt: the labeling rubric with a definition for every field, edge-case rules, seven worked examples from the synthetic **train** split, and the output JSON schema enforced by the API. |
| `multi` | The same rubric split into three focused prompts chained in code: triage, then people and risk, then location. Non-requests stop after triage. |

All three use `claude-opus-5-5` at its default effort (`medium`, set explicitly). The prompts are
in `src/rescue_tagger/llm_baseline.py`; `--show-prompts` prints them.

## Rows

- **Test and Harvey:** a fixed sample (`evals/compare_sample.json`) of up to 60 requests and 60
  non-requests per source, picked by a hash of the id. Balanced so request recall is measured on
  enough real requests (only 3.5% of Harvey is a request). Scored with upstream labels (primary)
  and rubric labels (secondary).
- **Edge cases:** `evals/edge_cases.jsonl`, 44 hand-written cases in 12 categories: powered medical
  devices, one duration covering several needs, vulnerable people described indirectly,
  non-requests, old posts, third-party asks, counts in words, location types, text aimed at the AI,
  Spanglish, fire and injury, and messages too vague to call. Written from the rubric before any
  model saw them (`evals/make_edge_cases.py`).
  A second annotator labeled the texts blind, from the rubric alone
  (`evals/edge_second_labels.jsonl`). A field is scored only when both labelers agree; the 13
  fields where they disagreed are masked, never settled by looking at a model's output.
  A case passes when every field it labels is right. For the vague cases, the right answer is a low
  `is_request` confidence (below 0.75), not a confident guess.

## Fairness and overfitting rules

- Since data v3, four edge categories are also targeted by training data: powered medical
  devices, one duration covering several needs, old posts and indirect vulnerability (counted as
  under-covered in the training data, generated from the rubric, never from edge text). Scores in
  those four now show whether targeted data worked, not generalization to unseen categories; the
  other eight categories still do.

- This is a report, never a selection signal. Nothing in it picks a run, a threshold or a prompt.
  Each scoring of our model on test and Harvey is logged to `/runs/eval_reads.jsonl`.
- The Opus prompts were written from the rubric and not tuned on these results. A test checks that
  they share no 4-word phrase with any edge case.
- Few-shot examples come only from the train split, picked by a hash of the id.
- Edge cases are protected from training by the leak check (`configs/data.yaml`); adding them
  dropped no training rows.
- Known bias: the edge cases, the second labels and the Opus prompts were all written by Claude
  models, the same family as the baseline. Training data never comes from Claude models (OpenAI, and open-weights models on Modal).
- A missing or failed Opus output scores as wrong on every field it would have been graded on
  (and confident), so failures never earn credit for "not a request" or "unsure". A condition is
  scored only when it covers every row; the table shows coverage and the failure count.
- The worked examples carry labels only, no confidences, so they don't teach "always 95% sure".
- Opus latency is wall-clock per message and includes any rate-limit backoff (8 parallel
  workers); ours is single-message CPU latency.
- The four vague edge cases are under 8 words, so the leak check protects them by exact match only.
- Harvey and HumAID test use HumAID's label scheme, where donation appeals count as requests. Our
  rubric (and so the Opus prompt) says they are not, so Opus is penalized there for following the
  rubric. Read those rows next to the edge rows, which use the rubric.

## Running it

```bash
pip install -e ".[llm]"
python -m rescue_tagger.llm_baseline --condition structured   # also naive, multi; resumable
python -m rescue_tagger.compare                                # Opus columns only, local
# every condition must cover every row; --allow-partial scores missing rows as wrong
```

Our column needs a checkpoint, so it runs on Modal: dispatch the `diagnose` workflow with the run
name and `compare` ticked. The table lands in the job summary and in `/runs/<run>/compare.md`.
