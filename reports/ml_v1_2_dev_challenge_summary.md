# ML-v1.2 Dev Challenge Summary

The ML-v1.1 dev challenge has already influenced development decisions, so ML-v1.2 introduces a fresh, independently authored development challenge. This report contains no challenge text or synthetic values.

| Property | Value |
|---|---:|
| Records | 540 |
| Contrast groups | 270 |
| Families | 15 |
| Entity spans | 1026 |
| Multi-entity ratio | 0.900 |

## Families

- `ml_v1_2_dev_challenge_appeal_itin_attestation_v1_2`
- `ml_v1_2_dev_challenge_artifact_version_relay_v1_2`
- `ml_v1_2_dev_challenge_catalog_coordinate_socket_v1_2`
- `ml_v1_2_dev_challenge_citation_tax_registration_v1_2`
- `ml_v1_2_dev_challenge_dispatch_dotted_locator_v1_2`
- `ml_v1_2_dev_challenge_exchange_api_authorization_v1_2`
- `ml_v1_2_dev_challenge_operator_license_enrollment_v1_2`
- `ml_v1_2_dev_challenge_orchestration_aba_clearance_v1_2`
- `ml_v1_2_dev_challenge_permit_passport_inspection_v1_2`
- `ml_v1_2_dev_challenge_petition_identity_clearance_v1_2`
- `ml_v1_2_dev_challenge_portfolio_deposit_destination_v1_2`
- `ml_v1_2_dev_challenge_protocol_revision_endpoint_v1_2`
- `ml_v1_2_dev_challenge_remittance_auth_session_v1_2`
- `ml_v1_2_dev_challenge_shipment_card_adjudication_v1_2`
- `ml_v1_2_dev_challenge_subscriber_benefit_identity_v1_2`

## Category coverage (records/families)

| Category | Dev coverage |
|---|---:|
| `business_id_vs_api_key` | 36/1 |
| `business_id_vs_auth_token` | 36/1 |
| `business_id_vs_bank_account` | 36/1 |
| `business_id_vs_credit_card` | 36/1 |
| `business_id_vs_driver_license` | 36/1 |
| `business_id_vs_ip_address` | 36/1 |
| `business_id_vs_itin` | 36/1 |
| `business_id_vs_passport` | 36/1 |
| `business_id_vs_routing_number` | 36/1 |
| `business_id_vs_ssn` | 72/2 |
| `business_id_vs_tax_id` | 36/1 |
| `ip_address_vs_technical_reference` | 108/3 |

Coverage includes BUSINESS_ID against SSN, credit card, bank account, routing number, ITIN, TAX_ID, passport, driver license, API key, auth token, and IP address; the actual ML-v1.1 worst category (`ip_address_vs_technical_reference`) receives 3 dedicated families; CUSTOMER_ID and REQUEST_ID contexts are present; and 486 records are multi-entity logs pairing the focus value with a genuine companion entity.

## Independence

Validator-enforced: zero family, exact-text, source-ID, focus-value-hash, and normalized-skeleton overlap with the ML-v1 train/dev/test splits, the ML-v1.1 training addition, the ML-v1.1 dev challenge, the sealed final challenge, and the ML-v1.2 training addition (surface styles: `countercheck_audit`, `countercheck_bracketed`, `countercheck_csv`, `countercheck_json`, `countercheck_kv`, `countercheck_nested_json`, `countercheck_query`, `countercheck_syslog`).

## Sealed-test policy

The sealed final challenge was not modified, opened, or evaluated in this phase. Its byte hash was verified exclusively through deterministic in-memory regeneration by the frozen v1.1 generator.
