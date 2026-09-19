# ML-v1.3 Failure-Mode Coverage

| Failure mode | Train records | Dev records |
|---|---:|---:|
| `accountlike_o_vs_business_id` | 576 | 120 |
| `auth_token_vs_business_id` | 576 | 120 |
| `high_risk_identifier` | 384 | 80 |
| `ip_vs_technical_reference` | 384 | 80 |
| `ssn_vs_itin` | 384 | 80 |

Account-like O examples use service/deploy/environment account names as gold `O`
beside a true BUSINESS_ID. AUTH_TOKEN examples put Bearer tokens in Authorization
context beside opaque business identifiers. SSN and ITIN appear in payroll/tax
contexts rather than by class frequency alone. IP versus technical-reference keeps
dotted-quad versions as `O`.
