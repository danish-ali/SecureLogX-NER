# ML-v1.3 Tokenizer Alignment Comparison

| Model | View | Spans | Aligned | Adjusted | Truncated | Failed | Unexplained |
|---|---|---:|---:|---:|---:|---:|---:|
| bert | challenge_dev | 860 | 840 | 0 | 0 | 20 | 0 |
| bert | probe | 17 | 17 | 0 | 0 | 0 | 0 |
| bert | standard_dev | 4020 | 4017 | 0 | 0 | 3 | 0 |
| bert | train | 42111 | 42033 | 29 | 0 | 78 | 0 |
| deberta | challenge_dev | 860 | 840 | 0 | 0 | 20 | 0 |
| deberta | probe | 17 | 17 | 1 | 0 | 0 | 0 |
| deberta | standard_dev | 4020 | 4017 | 2875 | 0 | 3 | 0 |
| deberta | train | 42111 | 42035 | 25622 | 0 | 76 | 0 |

BIO mapping is complete for both models: 25 entities / 51 labels, `O=0`.
DeBERTa-v3 uses `DebertaV2TokenizerFast` (SentencePiece). Truncation or unknown failed
reasons would have stopped full DeBERTa training. Allowed explained failures are
`tokenizer_emitted_no_token_for_span`, `token_boundary_adjustment_exceeds_three_characters`,
and `multiple_gold_spans_share_a_wordpiece` (duplicate identical gold spans in frozen
v1.3 log4j accountlike records; those tokens are ignored in the loss).
DeBERTa boundary adjustments are higher because SentencePiece tokens often include a
leading space; adjustments stay within the three-character allowance.
The frozen dataset was not modified for either tokenizer.
