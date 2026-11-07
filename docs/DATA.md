# Data card

Every row the model trains or is tested on comes from one of three sources, converted to one
record shape (`src/rescue_tagger/records.py`). A row carries only the labels its source can vouch
for. Anything missing is unknown and masked out of the loss, never treated as a negative.

## Sources

| Source | Rows (built) | License | Labeled by | Heads it trains | Spans |
| --- | --- | --- | --- | --- | --- |
| [Munro disaster response messages](https://github.com/rmunro/disaster_response_messages) @ `1657e1c` | ~23.5k | CC BY 4.0 | People (responders and translators, 2010-2012) | is_request, needs, request_kind (partial) | none |
| [HumAID](https://huggingface.co/datasets/QCRI/HumAID-event-type) @ `388eefc` | ~75k (incl. ~9.9k Harvey hold-out) | CC BY-NC-SA 4.0 | Qualified crowd workers, 2 of 3 agreement | is_request (weak, weight 0.5), request_kind=other | none |
| Synthetic (this repo) | growing | ours | Label-first spec, then blind OpenAI verifier | every head | all five span tags |

### Label mappings

**Munro.** `is_request` = `request`. `needs` = `water` / `food` / `medical_products`, only on
requests (an offer of water is not a need). `request_kind` = `other` for non-requests; for requests,
`search_and_rescue` -> rescue, `missing_people` -> welfare_check, any supply column -> supply,
otherwise unknown. Rows with `related = 2` and ~50 upstream duplicate rows are dropped. Messages are
the English translations; the originals (Haitian Creole, Urdu) stay in `meta`.
In train and dev, `news`-genre rows are relabeled not-a-request: a news article is never someone
asking for help, and upstream `request = 1` there mostly means "the article mentions needs"
(about 4% of dev news rows). Test keeps the upstream labels.

**HumAID.** `is_request` = class `requests_or_urgent_needs`. That class also covers donation
appeals, which our rubric does not count as requests, so HumAID rows get loss weight 0.5.
Non-requests get `request_kind = other`. `dont_know_cant_judge` rows are dropped.
In train and dev, `is_request` is left unknown for `sympathy_and_support` and
`rescue_volunteering_or_donation_effort`: those classes hold many real asks ("please send help,
they have no water") labeled as not-a-request. Test and the Harvey hold-out keep upstream labels.
Found from dev error examples (2026-11-07); fixed per category, never per example. Rows left with
no labels at all are dropped. Train configs carry `data_version: 2` so runs on the old labels are
never compared with these.

**Version 3 (2026-11-07).** Adds `b0003-focus`, a synthetic batch for categories the first batches
barely covered, counted from the training data and never from any eval set: powered medical devices
without power (10 of 1,706 rows, none also labeled `medical`, though the rubric counts dependence on
treatment as medical), one duration covering several needs (14 rows), old posts (76) and vulnerable
people described indirectly. 188 rows kept of 261 tried (72%; the remaining 439 seeds were
not tried because the OpenAI credits ran out): 51 device, 82 shared-duration, 30 old-post and 52
indirect rows. (Rejection rows for the 439 untried seeds, all "no credits" API errors, were
removed from `b0003-focus.rejected.jsonl`; it keeps only real verifier and span rejections.)
Two fixes to all batches, by `data/synthetic/fix_v3.py`: every old-post row named "Harvey 2017"
and no other row named Harvey, a shortcut aimed straight at the Harvey hold-out, so those 106 rows
now name another past storm (labels unchanged, spans moved); and the 10 earlier rows with a powered
device but no `medical` label get it. Future old posts are prompted with a non-Harvey storm.
Inference now gives a duration to every need listed with it ("no food or water for 2 days"), not
only the nearest one (`durations.needs_for_span`). Real-text dev is unchanged, so the first v3 run must hold the v2
champion's real-text score (`REAL_DEV_SAME` in `loop.py`) before it becomes the v3 baseline.

**Open-weights generation (2026-11-07, from batch b0004).** Gemma 4 31B (Apache-2.0) writes and
Phi-4 (MIT) labels blind, on Modal (`modal_jobs/synth.py`); batches land in `data/synthetic/staging/`
for review first. When the verifier disagrees on one or two fields, those fields become unknown
(masked from the loss) and the row is kept, with the disagreements in `meta.masked`. It is
rejected if the verifier disagrees on whether it is a request at all, or on three or more fields
(`apply_verdict`, `MAX_MASKED`). With whole-row rejection the 20-spec pilot kept 2 rows: the text
was fine, and most disagreements were the verifier missing a powered device or reading cross
streets as an address.

**Synthetic.** A seeded RNG samples a spec (language, style, request or not, request kind, hazards,
vulnerable people, needs with durations, location type, people count, time reference, contact,
prompt injection). `gpt-5.5` writes a message expressing exactly that spec and quotes the span
substrings. `gpt-5.4-mini` then labels the text blind. Rows where any head disagrees, or a quote is
not found verbatim, are rejected (about 20% in the pilot). Labels come from the spec, not from the
generator's reading of its own text.

## Splits and hold-outs

- Munro and HumAID keep their upstream train / dev / test splits. Synthetic rows are split by a hash
  of their id (85 / 10 / 5), so a row's split never changes.
- **Every `test` row is evaluation-only.** So are **Hurricane Harvey tweets** (Houston, Aug 2017):
  hurricane tweets dated 2017-08-17 to 2017-09-03 plus any tweet mentioning Harvey or Houston. They
  are written to `harvey.jsonl` as real raw text for golden set B and out-of-domain tests.
- **Leak check:** any train or dev row whose text matches a protected text (all test rows, Harvey,
  the contract fixtures, and the product repo's `evals/sets` and `evals/drafts` when `EVAL_REPO` is
  set) exactly or near-verbatim (half its word 8-grams) is dropped. The first build dropped 301 rows.
- The sealed holdout (set E) is never read by anything in this repo.

## Data v4 (Gemma focus batch, product sets C and D)

v4 adds the product team's hand-written sets B, C and D (train and dev rows only; see
`data/product/README.md`) and one Gemma batch. Product dev rows pick checkpoints but are kept out
of the select-half scores that rank runs, so real-text dev is unchanged from v3: v4 runs are judged
against the v3 champion and must hold its real-text score and critical fields.

v4's Gemma batch was written by Gemma 4 31B (open weights, on Modal) and checked blind by Phi-4.
When Phi-4 disagrees on one or two fields, those fields are masked (dropped from the labels, so
they train nothing) rather than the row being thrown out; a disputed `is_request` or three or
more disputed fields rejects the row. Masking can lose labels but never adds a wrong one.

- **b0005-focus-gemma:** the four v3 focus categories. 322 of 600 kept (278 rejected, 75 of them
  cut-off JSON). 274 kept rows have at least one masked field: hazards 101, request_kind 84,
  location_type 80, needs 72, vulnerable 46, people_count 37. Caveat: Phi-4 often misses
  "device without power" (more so in Spanish), so the device hazard is masked in 57 of 93
  device rows. The batch still adds device text; the label is simply not taught on those rows.
  When a masked field has spans (a count's COUNT, a need's NEED_DURATION), the row's spans
  become unknown too, so the token head never learns a disputed span (19 rows).
- **Vague messages:** short messages whose right answer is "unsure" (soft `is_request` 0.5).
  They carry no hard `is_request`, so a yes/no verifier could never reject one; their verifier
  instead answers needs_help / no_help / cannot_tell, and only cannot_tell keeps the row.
  b0006 was generated before that rule, so its texts are re-checked under it and land in a later
  data version (synthetic only, so real-text dev stays the same as v4).

## Data v5 (re-checked vague messages)

- **b0007-vague-gemma-rv:** b0006's 300 Gemma texts re-checked blind by Phi-4 under the
  cannot-tell rule. 146 kept (81 English, 49 Spanish, 16 mixed), 154 rejected because the
  verifier read them as clearly needing help or clearly not. Every kept row has soft
  `is_request` 0.5; 32 have location type none (no usable place), 85 an address or landmark
  (a bare place with no ask), and 29 a location type the verifier disputed (masked).
- Synthetic only, so real-text dev is the same as v4 and laya-v5 must hold v4's real-text score.

## Data v6 (Gemma batch with Vietnamese)

- **b0008-v6-gemma:** 307 of 600 kept (293 rejected by the blind Phi-4 check). Languages: 131
  English, 68 Spanish, 61 Vietnamese (new in v6), 47 mixed. Focus categories: indirect vulnerable
  151, powered medical device 63, old posts 51, shared durations 42. Weights come from training
  counts after v5, never from eval text. 257 rows have one or two fields masked where the verifier
  disagreed.
- Synthetic only, so real-text dev is unchanged.

## Data v7 (possible pranks)

- **b0009-hoax-gemma:** messages that read like a request for help but carry a hoax sign (an
  absurd detail, "lol"/"jk", meme phrasing, a friend tagging someone as a joke, a celebrity or
  fictional character, an over-the-top tone). Soft `is_request` 0.5 (the rubric: pranks and
  unclear posts get low confidence, never a confident yes); stated details are labeled as
  written. 90 of 400 kept: the blind verifier read 245 as real pleas (rejected) and 42 failed to
  parse. 47 English, 19 Spanish, 13 Vietnamese, 11 mixed.

## Data v8 (code-switching, old posts, third-party asks)

- **b0010-mix-gemma** (`--mix`, seeds 350000+): requests built around one of three categories
  Paawan asked for: code-switching (35%), old posts from a named past storm (30%) and asks made
  for someone who is not with the writer (35%); 30% of the other two are code-switched too.
  363 of 600 kept (163 rejected by the blind verifier, 74 parse errors). 169 mixed
  English/Spanish, 105 Vietnamese, 56 English, 33 Spanish; 58 old posts, 132 third-party asks.
- A Vietnamese row that switches into English is labeled `lang: vi` (the language it is mostly
  written in); English/Spanish mixing is `mixed`, as before.
- Third-party asks keep the request kind they were written for (rescue, evacuation, welfare
  check, supply); where the verifier read it differently the field is masked.

## Data v9 (implied hazards, messy typing)

- **b0011-implied-gemma** (`--implied`, seeds 360000+): requests whose P1 hazards are implied by
  the situation and never named (an attic, a stalled car in a flooded underpass, water at the
  door, smoke or sparks, a fall), 75% in messy typing (typos, no punctuation, terse, hedged);
  60% also describe a vulnerable person indirectly ("she's 82", "my son"). Written from the
  rubric's P1 kinds, not from any evaluation row.
- 311 of 600 kept (222 rejected by the blind verifier, 67 parse errors). 154 English, 61
  Spanish, 49 mixed, 47 Vietnamese. The verifier disagreed on the hazards of 197 kept rows, so
  their hazards are masked; 114 rows carry hazard labels (rising water 85, fire 23, on roof 11,
  trapped 11, injury 9).
- Training for v9 also weights a present hazard or vulnerable kind 3x in the loss
  (`positive_weight`).

## Known gaps

- Public data is English only. Spanish and mixed text come from synthetic data alone.
- No public source labels hazards, vulnerable people, location type, people count or spans; those
  heads learn from synthetic data only. Golden set B on real text is the check on that.
- HumAID outnumbers everything else about 2 to 1 and is mostly non-requests. The trainer needs
  per-source sampling weights, not just loss weights.
- HumAID's license is non-commercial share-alike. If the weights must be commercially usable,
  disable it in `configs/data.yaml` and retrain.

## Data v10 (Opus rubric labels for real train tweets)

- **d0001-opus** (`rescue_tagger.data.distill`, source `distill`): 2,500 real tweets from the
  TRAIN split of Munro (1,275) and HumAID (1,225), half labelled request by their source and half
  not, picked by a hash of the id. Opus 5.5 labelled each with the structured rubric prompt from
  `llm_baseline` (training time only; the deployed model stays offline). Each replaces its
  original row in the build, so no text is trained under two labels. Dev, test, Harvey, edge and
  product rows are never sent.
- Labels: 599 requests, 223 soft is_request 0.5 (Opus under 0.6 confident); hazards (injury 22,
  trapped 21, rising water 10, fire 3, on roof 2) and vulnerable people (child 55, medical 22,
  pregnant 9, elderly 9, disabled 3) at confidence 0.5 or more. Spans are left unknown.
- Opus agrees with the source's is_request label on 1,796 of 2,277 rows (79%); the rest are
  mostly donation appeals and news that HumAID/Munro call requests and the rubric does not.
- Real dev keeps the source labels, so a v10 model may score a little lower on real-dev
  is_request even when it follows the rubric better.
- Share of each epoch: synthetic 0.40, Munro 0.27, HumAID 0.18, distill 0.13, product 0.02.

