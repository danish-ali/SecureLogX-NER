# ML-v1.3 Security-Outcome Analysis of ML-v1.2 Development Errors

This report reclassifies frozen ML-v1.2 development prediction errors by
masking/security outcome. Historical NER metrics and gates are unchanged.
The sealed challenge was not opened.

## Headline counts

- Security-critical misses: **123**
- Security-critical partials: **100**
- Policy-safe wrong-class: **143**
- Over-masking: **233**
- Sensitive-span recall: **0.975624**
- Full-mask recall: **0.955807**
- High-risk full-mask recall: **0.987492**

## By development view

| View | Miss | Partial | Policy-safe wrong class | Overmasking | Sensitive-span recall | Full-mask recall | High-risk full-mask recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| standard_dev | 112 | 78 | 105 | 233 | 0.972139 | 0.952736 | 0.983972 |
| challenge_dev | 11 | 22 | 38 | 0 | 0.989279 | 0.967836 | 0.997475 |

## Focused failure modes

| Mode | Count | Security reading |
|---|---:|---|
| Account-like gold-O → BUSINESS_ID | 94 | OVERMASKING; operational noise, not a sensitive leak |
| SSN → ITIN | 3 | POLICY_SAFE_WRONG_CLASS; full span still masked |
| business_id_vs_auth_token errors | 18 | mostly BUSINESS_ID absorbed as AUTH_TOKEN (policy-safe or overmasking) |
| AUTH_TOKEN misses/partials | 3 | SECURITY_CRITICAL if non-zero |
| API_KEY misses/partials | 9 | SECURITY_CRITICAL if non-zero |
| BUSINESS_ID false negatives (miss/partial/wrong-class) | 74 | miss/partial can leak a business identifier; wrong-class is policy-safe if covered |
| REQUEST_ID-tagged failures | 18 | subtype metadata only; outcome follows the BUSINESS_ID span |
| IP vs technical-reference errors | 0 | v1.2 largely repaired morphology-over-context IP overmasking |
| High-risk partial spans | 8 | SECURITY_CRITICAL_PARTIAL |

## Interpretation

The persistent 94 account-like gold-O → BUSINESS_ID events are over-masking,
not security misses. SSN → ITIN remains a genuine NER error that is policy-safe
under default MASK/MASK_LAST4 actions. Challenge AUTH_TOKEN gold spans were
recalled; the remaining AUTH_TOKEN issue is BUSINESS_ID predicted as AUTH_TOKEN
inside `business_id_vs_auth_token`. API_KEY gold spans on the v1.2 challenge
were fully covered. Residual security risk is concentrated in uncovered
BUSINESS_ID / high-risk misses or partials, not in the account-like overmasking bucket.
