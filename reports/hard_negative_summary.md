# Hard-Negative Scenario Summary

- Hard-negative template families: **12**
- Hard-negative records: **1128**

| Template family | Format | Assigned split | Records | Expected annotation rationale |
|---|---|---|---:|---|
| `application_id_account_shape_json_v1` | json | train=94 | 94 | The long numeric application ID is BUSINESS_ID, not a bank account. |
| `build_identifier_ip_shape_v1` | text | train=94 | 94 | The dotted value is a build identifier, not a network address. |
| `email_like_service_route_v1` | text | train=94 | 94 | The email-shaped string names an internal route rather than a person's mailbox. |
| `error_trace_adjacent_phone_ip_v1` | text | train=94 | 94 | Annotate the caller phone and client IP; the UUID is a trace identifier. |
| `free_text_version_reference_v1` | text | train=94 | 94 | The dotted token is a software version, not an IP address. |
| `key_value_accountlike_batch_v1` | key_value | dev=94 | 94 | The long number is explicitly a batch identifier, not a financial account. |
| `order_reference_ssn_shape_warn_v1` | text | test=94 | 94 | The SSN-shaped order reference is a business identifier, not an SSN. |
| `structured_json_numeric_noise_v1` | json | train=94 | 94 | HTTP status, retry count, and request UUID are operational fields. |
| `trace_adjacent_auth_token_kv_v1` | key_value | train=94 | 94 | Annotate the authorization token only; the adjacent UUID is operational trace data. |
| `transaction_id_adjacent_card_v1` | text | train=94 | 94 | Annotate the card as CREDIT_CARD_NUMBER and normalize the equally long TRANSACTION_ID to BUSINESS_ID. |
| `transaction_id_card_shape_kv_v1` | key_value | train=94 | 94 | The 16-digit value is a TRANSACTION_ID normalized to BUSINESS_ID, not a payment card. |
| `uuid_adjacent_email_json_v1` | json | train=94 | 94 | Annotate the personal email only; the adjacent UUID is a request ID. |
