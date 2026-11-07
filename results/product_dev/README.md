# Product dev reports (sets B, C and D)

`python -m rescue_tagger.product_eval` on the product team's dev rows.

These are not held-out reads for checkpoints trained from data v4 on: the product dev rows pick
the checkpoint (half the checkpoint score), so a v4+ model's numbers here are selection-biased and
optimistic. s13 and v3 never saw these rows, so their numbers are clean. The product's held-back
test split is the real score.
