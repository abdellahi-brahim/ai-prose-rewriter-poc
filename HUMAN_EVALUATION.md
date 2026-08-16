# Blind evaluation protocol

Use `blind_review.csv` without opening `blind_review_key.csv`. Ideally, each held-out item should be rated independently by at least two people who did not create its target rewrite.

For every row:

1. Read the source, then options A and B.
2. Choose which option sounds more natural: `A`, `B`, or `TIE`.
3. Mark whether each option preserves every substantive part of the source meaning (`Y` or `N`).
4. In `notes`, record omissions, invented claims, changed certainty, changed negation, or altered names/numbers.

Naturalness means the text sounds direct and purposeful in context. Do not reward an option merely for being shorter. Meaning preservation takes priority over style.

After ratings are locked, use `blind_review_key.csv` to decode which option came from the model. Report:

- model win, loss, and tie rates for naturalness;
- model meaning-preservation rate;
- inter-rater agreement when there is more than one rater;
- representative failure categories, especially changed facts or intent.

The PoC supports the hypothesis only if it clears the success gates in the README on a genuinely held-out test set. Automated scores alone are insufficient.
