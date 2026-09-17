# Token Alignment Report

Character spans were aligned with fast-tokenizer offset mappings and overlapping windows.

| Split | Records | Windows | Total spans | Successfully aligned | Boundary-adjusted | Truncated | Failed |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 23558 | 26051 | 32203 | 32197 | 29 | 0 | 6 |
| dev | 2994 | 3324 | 4020 | 4017 | 0 | 0 | 3 |
| test | 2989 | 3313 | 4358 | 4358 | 1 | 0 | 0 |
| **Total** | **29541** | **32688** | **40581** | **40572** | **30** | **0** | **9** |

## Deterministic handling

- Special and padding tokens receive `-100`.
- The first token of a span receives `B-*`; subsequent tokens receive `I-*`.
- Partial entity fragments in an overflow window receive `-100`, never `O`.
- If a character boundary falls inside a WordPiece, adjustments totaling at most three characters are supervised and reported explicitly.
- Larger token-boundary expansions/contractions are treated as explicit tokenizer-unrepresentable failures and receive `-100`, avoiding semantically unsafe supervision.
- Predictions from overlapping windows are merged by averaged logits at identical character offsets.

## Affected labels and diagnostics

- Affected labels: BANK_ACCOUNT_NUMBER=1, EMAIL=2, PERSON_NAME=17, PHONE=1, ROUTING_NUMBER=1, STREET_ADDRESS=4, SWIFT_BIC=13
- Diagnostic reasons: character_boundary_inside_or_outside_token=30, exact_token_boundaries=40542, token_boundary_adjustment_exceeds_three_characters=7, tokenizer_emitted_no_token_for_span=2

Unexplained/truncation alignment blockers: **0**.
Explicitly tokenizer-unrepresentable spans excluded from token supervision: **9**.
Tokenizer-unrepresentable spans are listed in the machine-readable diagnostic file and are not silently treated as `O`.
