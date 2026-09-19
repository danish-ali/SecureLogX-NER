# ML-v1.3 Corrected Development Evaluation

**Development only. No sealed challenge or original-test inference. No retraining.**

The common evaluator now trims tokenizer-added surrounding whitespace from decoded entity spans before entity-level scoring/runtime span output.

## Selected checkpoints after corrected evaluation

| Metric | BERT | DeBERTa |
|---|---:|---:|
| Selected epoch | 3 | 1 |
| Selection score | 0.963232 | 0.937421 |
| Standard micro F1 | 0.911143 | 0.871826 |
| Standard macro F1 | 0.933123 | 0.890406 |
| Challenge micro F1 | 0.979592 | 0.954003 |
| Challenge macro F1 | 0.995349 | 0.989678 |
| BUSINESS_ID standard P/R/F1 | 0.551/0.963/0.701 | 0.539/0.911/0.678 |
| SSN standard F1 | 0.961538 | 0.833333 |
| ITIN standard F1 | 0.989474 | 0.895238 |
| AUTH_TOKEN standard F1 | 1.000000 | 1.000000 |
| API_KEY standard F1 | 0.915888 | 0.912442 |
| Combined security miss+partial | 225 | 380 |
| Account-like O -> BUSINESS_ID (standard) | 94 | 94 |

Corrected decision: **BERT SELECTED BY CORRECTED DEVELOPMENT EVALUATION**

BERT has 155 fewer corrected combined security-critical miss/partial outcomes.

Historical ML-v1.3 reports and the original predeclared selection result are preserved.
The sealed challenge remains blocked until this corrected development result is reviewed.
