# ML-v1.1 Challenge Split Summary

The dev challenge and sealed final challenge were independently authored and are family-disjoint from training and from each other. This report intentionally contains no challenge text or synthetic values.

| Challenge | Records | Groups | Families | Entity spans |
|---|---:|---:|---:|---:|
| Dev-only context conflict | 480 | 240 | 12 | 440 |
| Sealed final challenge | 480 | 240 | 12 | 420 |

### Dev-only context conflict families

- `ml_v1_1_dev_challenge_account_depositshape_receipt_v1_1`
- `ml_v1_1_dev_challenge_application_claim_locator_v1_1`
- `ml_v1_1_dev_challenge_case_identity_locator_v1_1`
- `ml_v1_1_dev_challenge_customer_identity_receipt_v1_1`
- `ml_v1_1_dev_challenge_deployment_network_quad_v1_1`
- `ml_v1_1_dev_challenge_invoice_identity_locator_v1_1`
- `ml_v1_1_dev_challenge_protocol_artifact_quad_v1_1`
- `ml_v1_1_dev_challenge_request_identity_locator_v1_1`
- `ml_v1_1_dev_challenge_ticket_cardshape_receipt_v1_1`
- `ml_v1_1_dev_challenge_transaction_cardshape_receipt_v1_1`
- `ml_v1_1_dev_challenge_user_bankshape_receipt_v1_1`
- `ml_v1_1_dev_challenge_workflow_identity_locator_v1_1`

### Sealed final challenge families

- `ml_v1_1_sealed_challenge_customer_deposit_folio_v1_1`
- `ml_v1_1_sealed_challenge_edge_release_quad_v1_1`
- `ml_v1_1_sealed_challenge_incident_identity_folio_v1_1`
- `ml_v1_1_sealed_challenge_invoice_payment_folio_v1_1`
- `ml_v1_1_sealed_challenge_member_identity_folio_v1_1`
- `ml_v1_1_sealed_challenge_proxy_artifact_quad_v1_1`
- `ml_v1_1_sealed_challenge_registry_identity_folio_v1_1`
- `ml_v1_1_sealed_challenge_request_deposit_folio_v1_1`
- `ml_v1_1_sealed_challenge_shipment_identity_folio_v1_1`
- `ml_v1_1_sealed_challenge_socket_schema_quad_v1_1`
- `ml_v1_1_sealed_challenge_submission_identity_folio_v1_1`
- `ml_v1_1_sealed_challenge_transaction_payment_folio_v1_1`

## Gate coverage

- Dev BUSINESS_ID vs SSN families: **6** (minimum 5)
- Dev BUSINESS_ID vs card/account families: **4** (minimum 3)
- Dev network/technical families: **2** (minimum 2)
- Sealed challenge families: **12** (required 8-12)
- Sealed formats: `json`, `key_value`, `text`

## Sealed-test policy

The sealed challenge was generated and hash-frozen before any ML-v1.1 training. Development-time predictions are forbidden. No model was loaded and no predictions were produced in this phase. Only aggregate family/count/hash metadata is reported here.
