# ML-v1.1 BUSINESS_ID Distribution

BUSINESS_ID subtypes are provenance only. The canonical neural ontology still exposes one `BUSINESS_ID` label.

- Frozen ML-v1 train BUSINESS_ID spans: **1,360**
- New training-addition BUSINESS_ID spans: **1100** (80.9% of the parent count)
- Frozen ML-v1 train SSN spans: **843**
- New training-addition SSN spans: **600** (71.2% of the parent count)

## Subtype provenance

| BUSINESS_ID subtype | Train addition | Dev challenge | Sealed challenge | New total |
|---|---:|---:|---:|---:|
| CUSTOMER_ID | 100 | 20 | 20 | 140 |
| ACCOUNT_ID | 100 | 20 | 0 | 120 |
| APPLICATION_ID | 100 | 20 | 20 | 140 |
| TRANSACTION_ID | 100 | 20 | 20 | 140 |
| USER_ID | 100 | 20 | 20 | 140 |
| ORDER_ID | 50 | 0 | 20 | 70 |
| CASE_ID | 100 | 20 | 0 | 120 |
| TICKET_ID | 100 | 20 | 20 | 140 |
| INVOICE_ID | 100 | 20 | 20 | 140 |
| REFERENCE_ID | 50 | 0 | 20 | 70 |
| REQUEST_ID | 100 | 20 | 20 | 140 |
| WORKFLOW_ID | 100 | 20 | 0 | 120 |

The revision adds the previously absent training subtypes ORDER_ID, CASE_ID, TICKET_ID, INVOICE_ID, REFERENCE_ID, REQUEST_ID, and WORKFLOW_ID, while retaining established customer/account/application/transaction/user contexts. No subtype was added to the 25-entity ontology or 51-label BIO map.
