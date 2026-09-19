# ML-v1.3 Policy Equivalence Matrix

Default product behavior: every canonical ML-v1 entity is masking-sensitive.
Java policy remains authoritative after ONNX export; this matrix is the
analysis contract used to separate leaks from safe wrong-class/overmasking.

| Gold entity | Default action | Predicted entity | Predicted action | Security outcome if full gold span is covered |
|---|---|---|---|---|
| SSN | MASK_LAST4 | ITIN | MASK | POLICY_SAFE_WRONG_CLASS |
| SSN | MASK_LAST4 | O | NONE | SECURITY_CRITICAL_MISS |
| AUTH_TOKEN | FULL_MASK | O | NONE | SECURITY_CRITICAL_MISS |
| AUTH_TOKEN | FULL_MASK | BUSINESS_ID | MASK | POLICY_SAFE_WRONG_CLASS |
| BUSINESS_ID | MASK | AUTH_TOKEN | FULL_MASK | POLICY_SAFE_WRONG_CLASS |
| BUSINESS_ID | MASK | O | NONE | SECURITY_CRITICAL_MISS |
| API_KEY | FULL_MASK | AUTH_TOKEN | FULL_MASK | POLICY_SAFE_WRONG_CLASS |
| CREDIT_CARD_NUMBER | MASK_LAST4 | BUSINESS_ID | MASK | POLICY_SAFE_WRONG_CLASS |
| IP_ADDRESS | MASK | BUSINESS_ID | MASK | POLICY_SAFE_WRONG_CLASS |
| O | NONE | BUSINESS_ID | MASK | OVERMASKING |
| O | NONE | IP_ADDRESS | MASK | OVERMASKING |
| O | NONE | AUTH_TOKEN | FULL_MASK | OVERMASKING |
| SSN | MASK_LAST4 | SSN (partial span) | MASK_LAST4 | SECURITY_CRITICAL_PARTIAL |

A wrong class remains an NER error. Policy-safe means the full sensitive
characters are still masked under default actions, not that the label is correct.
