# Product eval rows (sets C and D, train and dev only)

`rescue_eval_train_dev.jsonl`: 67 hand-written, hand-reviewed flood-rescue messages from the product
team's eval sets, given to us on 2026-11-07 to train on and pick checkpoints with.

- Set C, hard cases (40): typos, Spanish and Vietnamese, landmark-only locations, reposts, sarcasm,
  several households in one post. Set D, abstain cases (27): no usable location, conflicting
  headcounts, possible pranks, unclear requests or hazards.
- `split: train` (33) is gradient data. `split: dev` (34) only picks checkpoints (half the
  checkpoint score, `checkpoint_product_weight`) and is the head-by-head report
  (`python -m rescue_tagger.product_eval`).
- The product team holds back 102 more rows from these sets as the scored test split. They are not
  here and must never be recreated. The leak check still reads the product's eval files, so any of
  our rows that resembles a held-back row is dropped (C-049), and so is any row describing the same
  household as a held-back row (`gold_duplicate_group`: C-058 dev, C-059 train). A dev row of the
  same household as a train row (C-053) does not pick checkpoints. Net: 31 train, 32 dev rows.
- Product dev rows stay out of the select half of dev that ranks runs (`dev_real`), so real-text
  dev is unchanged from v3.
- No tweets: safe to train on and keep in the repo. How each field becomes a training target is
  in `src/rescue_tagger/data/product.py`.
