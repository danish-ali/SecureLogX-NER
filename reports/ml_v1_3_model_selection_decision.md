# ML-v1.3 Model Selection Decision

**DEBERTA SELECTED FOR FINAL VALIDATION**

DeBERTa has 32 fewer combined security-critical miss/partial outcomes
(BERT **224**, DeBERTa **192**). That exceeds the predeclared material margin of 5.
Challenge high-risk full-mask recall is tied at **1.000000**. The hashed comparison
gate therefore selects DeBERTa. The sealed challenge was not used.

## Persistent BERT problems

| Failure | BERT | DeBERTa | Improved? |
|---|---:|---:|---|
| Account-like gold-O → BUSINESS_ID (standard+challenge) | 94 | 94 | No |
| AUTH_TOKEN vs BUSINESS_ID (challenge) | 0 | 0 | Tie (already clean) |
| SSN → ITIN (standard+challenge) | 0 | 54 | No; DeBERTa worse |
| Challenge BUSINESS_ID recall | 0.962963 | 0.981481 | Small gain |

Both encoders reproduce the same **94** account-like O→BUSINESS_ID overmasks on
standard development. Challenge AUTH_TOKEN vs BUSINESS_ID is clean for both.
That pattern is evidence of **data/task ambiguity and/or a need for deterministic
context-resolver logic**, not a reason to keep switching encoders for the
account-like failure.

## What the DeBERTa security delta is

- Standard-dev SECURITY_CRITICAL_MISS is tied at **108**.
- The combined 32-count reduction is **4** fewer challenge misses, **15** fewer
  challenge partials, and **13** fewer standard partials.
- Standard POLICY_SAFE_WRONG_CLASS rises from **41** (BERT) to **127** (DeBERTa).
  Those spans would still be masked; they are not NER successes.
- Standard exact NER collapses for DeBERTa (micro F1 **0.263** vs BERT **0.911**;
  SSN recall **0.039**; ITIN recall **0.000**; API_KEY recall **0.000**;
  IP_ADDRESS F1 **0.077**). Challenge NER remains high for both, and DeBERTa
  SentencePiece had **0** boundary adjustments on the v1.3 challenge versus
  **2875** on standard-dev. The standard exact-span drop is consistent with
  tokenizer boundary mismatch, not with a better account-like decision rule.
- Standard high-risk full-mask recall is slightly lower for DeBERTa
  (**0.962600** vs **0.968833**). Challenge high-risk full-mask recall is tied
  at 1.0.

## Cost (secondary)

- Train minutes: BERT **62.06**, DeBERTa **202.68**
- Peak GPU GiB: BERT **3.32**, DeBERTa **4.17**
- Model bytes: BERT **431,058,932**, DeBERTa **735,507,468**
- 64-record challenge latency: BERT **0.305 s**, DeBERTa **0.521 s**

The predeclared ordered security rule is not overridden after seeing these
numbers. Computational cost is secondary and does not reverse the decision.

The sealed challenge was not opened. No ONNX export or Java change was performed.
