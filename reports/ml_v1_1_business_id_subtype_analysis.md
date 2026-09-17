# ML-v1.1 BUSINESS_ID Subtype Analysis

**Scope: selected-checkpoint development challenge only. Post-gate regression and sealed inference were not run.**

The classifier still emits only `BUSINESS_ID`; subtype is recovered from frozen gold provenance.

| Subtype | Support | Exact TP | FN | Exact-span recall | Partition note |
|---|---:|---:|---:|---:|---|
| CUSTOMER_ID | 20 | 10 | 10 | 50.00% | present in dev challenge |
| ACCOUNT_ID | 20 | 20 | 0 | 100.00% | present in dev challenge |
| APPLICATION_ID | 20 | 20 | 0 | 100.00% | present in dev challenge |
| TRANSACTION_ID | 20 | 20 | 0 | 100.00% | present in dev challenge |
| USER_ID | 20 | 20 | 0 | 100.00% | present in dev challenge |
| ORDER_ID | 0 | 0 | 0 | N/A | absent from dev challenge; not evaluated after gate stop |
| CASE_ID | 20 | 20 | 0 | 100.00% | present in dev challenge |
| TICKET_ID | 20 | 20 | 0 | 100.00% | present in dev challenge |
| INVOICE_ID | 20 | 20 | 0 | 100.00% | present in dev challenge |
| REFERENCE_ID | 0 | 0 | 0 | N/A | absent from dev challenge; not evaluated after gate stop |
| REQUEST_ID | 20 | 17 | 3 | 85.00% | present in dev challenge |
| WORKFLOW_ID | 20 | 20 | 0 | 100.00% | present in dev challenge |

ACCOUNT_ID and WORKFLOW_ID are directly represented here. ORDER_ID and REFERENCE_ID were deliberately reserved for other partitions, so their generalization cannot be claimed from this stopped experiment.
