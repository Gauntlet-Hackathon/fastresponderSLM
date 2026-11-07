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
indirect rows. Real-text dev is unchanged, so the first v3 run must hold the v2
champion's real-text score (`REAL_DEV_SAME` in `loop.py`) before it becomes the v3 baseline.

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

## Known gaps

- Public data is English only. Spanish and mixed text come from synthetic data alone.
- No public source labels hazards, vulnerable people, location type, people count or spans; those
  heads learn from synthetic data only. Golden set B on real text is the check on that.
- HumAID outnumbers everything else about 2 to 1 and is mostly non-requests. The trainer needs
  per-source sampling weights, not just loss weights.
- HumAID's license is non-commercial share-alike. If the weights must be commercially usable,
  disable it in `configs/data.yaml` and retrain.
