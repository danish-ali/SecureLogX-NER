#!/usr/bin/env python3
"""Generate isolated SecureLogX ML-v1.1 context-contrast records.

The generator is intentionally independent from the frozen ML-v1 pipeline.  It
returns canonical records in memory and writes only when :func:`write_jsonl` is
called (or an explicit CLI output path is supplied).  Every contrast group is a
pair: the exact same surface value appears in two different semantic contexts,
with the annotation determined by context rather than morphology alone.

Randomness is derived from ``seed + partition + family + group`` hashes.  As a
result, inserting or reordering catalog entries cannot change an existing
family's records.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import string
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


SOURCE = "securelogx_context_contrast_v1_1"
GENERATOR_VERSION = "securelogx-ml-v1.1-context-contrast-v1"
PARTITIONS: Tuple[str, ...] = (
    "train_addition",
    "dev_challenge",
    "sealed_challenge",
)

BUSINESS_ID_SUBTYPES: Tuple[str, ...] = (
    "CUSTOMER_ID",
    "ACCOUNT_ID",
    "APPLICATION_ID",
    "TRANSACTION_ID",
    "USER_ID",
    "ORDER_ID",
    "CASE_ID",
    "TICKET_ID",
    "INVOICE_ID",
    "REFERENCE_ID",
    "REQUEST_ID",
    "WORKFLOW_ID",
)

CANONICAL_TARGET_LABELS = frozenset(
    {
        "BUSINESS_ID",
        "SSN",
        "CREDIT_CARD_NUMBER",
        "BANK_ACCOUNT_NUMBER",
        "IP_ADDRESS",
        "PERSON_NAME",
        "STREET_ADDRESS",
        "O",
    }
)

SURFACE_STYLE_TO_FORMAT: Mapping[str, str] = {
    "key_equals": "key_value",
    "key_colon": "key_value",
    "json_flat": "json",
    "json_nested": "json",
    "natural_language": "text",
    "warn_message": "text",
    "error_message": "text",
    "structured_audit": "key_value",
    "mixed_punctuation": "text",
    "abbreviated_key": "key_value",
}

PAIR_COUNTS: Mapping[str, int] = {
    "train_addition": 50,
    "dev_challenge": 20,
    "sealed_challenge": 20,
}

# These are historical ML-v1 evidence and must never reappear in the v1.1
# catalog or generated text.  Keeping the guards next to the generator makes
# the independence requirement executable.
FORBIDDEN_PARENT_FAMILY = "order_reference_ssn_shape_warn_v1"
FORBIDDEN_PARENT_TEXT_FRAGMENT = "was replayed after a warehouse timeout"

VALUE_TOKEN = "[[VALUE]]"
COMPANION_TOKEN = "[[COMPANION]]"
SEQUENCE_TOKEN = "[[SEQ]]"
TRACE_TOKEN = "[[TRACE]]"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class RoleSpec:
    """One semantic rendering of a paired surface value."""

    context_role: str
    expected_label: str
    source_label: str
    surface_style: str
    template: str
    rationale: str
    companion_label: Optional[str] = None
    companion_source_label: Optional[str] = None


@dataclass(frozen=True)
class FamilySpec:
    """A leakage-isolated template family containing paired contexts."""

    template_family: str
    intended_split: str
    contrast_category: str
    morphology_class: str
    pair_count: int
    roles: Tuple[RoleSpec, RoleSpec]


def _role(
    context_role: str,
    expected_label: str,
    source_label: str,
    surface_style: str,
    template: str,
    rationale: str,
    companion_label: Optional[str] = None,
    companion_source_label: Optional[str] = None,
) -> RoleSpec:
    return RoleSpec(
        context_role=context_role,
        expected_label=expected_label,
        source_label=source_label,
        surface_style=surface_style,
        template=template,
        rationale=rationale,
        companion_label=companion_label,
        companion_source_label=companion_source_label,
    )


def _family(
    partition: str,
    slug: str,
    category: str,
    morphology: str,
    left: RoleSpec,
    right: RoleSpec,
) -> FamilySpec:
    return FamilySpec(
        template_family=f"ml_v1_1_{partition}_{slug}_v1_1",
        intended_split=partition,
        contrast_category=category,
        morphology_class=morphology,
        pair_count=PAIR_COUNTS[partition],
        roles=(left, right),
    )


def _business_role(
    subtype: str,
    style: str,
    template: str,
    rationale: str,
) -> RoleSpec:
    return _role(
        "business_identifier",
        "BUSINESS_ID",
        subtype,
        style,
        template,
        rationale,
    )


def _ssn_role(style: str, template: str, rationale: str) -> RoleSpec:
    return _role(
        "social_security_number",
        "SSN",
        "SSN",
        style,
        template,
        rationale,
    )


def _ssn_with_person_role(style: str, template: str, rationale: str) -> RoleSpec:
    """Build a true-SSN context with an independently annotated person."""

    return _role(
        "social_security_number",
        "SSN",
        "SSN",
        style,
        template,
        rationale,
        companion_label="PERSON_NAME",
        companion_source_label="PERSON_NAME",
    )


def _card_role(style: str, template: str, rationale: str) -> RoleSpec:
    return _role(
        "payment_card_number",
        "CREDIT_CARD_NUMBER",
        "CREDIT_CARD_NUMBER",
        style,
        template,
        rationale,
    )


def _bank_role(style: str, template: str, rationale: str) -> RoleSpec:
    return _role(
        "bank_account_number",
        "BANK_ACCOUNT_NUMBER",
        "BANK_ACCOUNT_NUMBER",
        style,
        template,
        rationale,
    )


def _ip_role(style: str, template: str, rationale: str) -> RoleSpec:
    return _role(
        "network_address",
        "IP_ADDRESS",
        "IP_ADDRESS",
        style,
        template,
        rationale,
    )


def _technical_role(style: str, template: str, source_label: str) -> RoleSpec:
    return _role(
        "technical_reference",
        "O",
        source_label,
        style,
        template,
        "The dotted value is explicitly a non-network technical reference and remains O.",
    )


def _catalog_entries() -> Tuple[FamilySpec, ...]:
    """Define all families explicitly so their semantic wording is auditable."""

    families: List[FamilySpec] = [
        # Training: twelve BUSINESS_ID/SSN families cover every required
        # BUSINESS_ID subtype once before any subtype is repeated.
        _family(
            "train_addition", "customer_identity_bridge", "business_id_vs_ssn", "ssn_hyphenated",
            _business_role("CUSTOMER_ID", "key_equals", "INFO commerceCustomer=[[VALUE]] lifecycle=renewal seq=[[SEQ]] trace=[[TRACE]]", "commerceCustomer names an internal customer record, so the value is BUSINESS_ID."),
            _ssn_with_person_role("json_nested", '{"identityCheck":{"socialSecurityNumber":"[[VALUE]]","subjectName":"[[COMPANION]]","outcome":"matched"},"sequence":[[SEQ]],"trace":"[[TRACE]]"}', "The nested identity field explicitly identifies a Social Security number beside the annotated subject name."),
        ),
        _family(
            "train_addition", "account_enrollment_folio", "business_id_vs_ssn", "ssn_compact",
            _business_role("ACCOUNT_ID", "key_colon", "NOTICE businessAccount: [[VALUE]] | enrollment=active | seq:[[SEQ]] | trace:[[TRACE]]", "businessAccount is an application account identifier, not banking data or an SSN."),
            _ssn_with_person_role("natural_language", "During benefit enrollment, Social Security number [[VALUE]] was recorded for member [[COMPANION]]; event [[SEQ]] carries trace [[TRACE]].", "Natural language explicitly gives the value the SSN semantic role beside the annotated member name."),
        ),
        _family(
            "train_addition", "application_intake_marker", "business_id_vs_ssn", "ssn_spaced",
            _business_role("APPLICATION_ID", "json_nested", '{"intake":{"applicationMarker":"[[VALUE]]","stage":"review"},"sequence":[[SEQ]],"trace":"[[TRACE]]"}', "applicationMarker identifies a business workflow application."),
            _ssn_role("error_message", "ERROR identity intake exposed the applicant SSN [[VALUE]]; quarantineEvent=[[SEQ]] trace=[[TRACE]]", "The error text explicitly describes an applicant SSN."),
        ),
        _family(
            "train_addition", "transaction_reconciliation_code", "business_id_vs_ssn", "ssn_dotted",
            _business_role("TRANSACTION_ID", "structured_audit", "AUDIT|txnCode=[[VALUE]]|phase=reconcile|sequence=[[SEQ]]|trace=[[TRACE]]", "txnCode is the identifier of a commercial transaction."),
            _ssn_role("json_flat", '{"event":"identity_reconcile","ssn":"[[VALUE]]","sequence":[[SEQ]],"trace":"[[TRACE]]"}', "The ssn field carries a true Social Security number."),
        ),
        _family(
            "train_addition", "user_access_registry", "business_id_vs_ssn", "ssn_masked_last4",
            _business_role("USER_ID", "json_flat", '{"event":"access_registry","userRegistryId":"[[VALUE]]","sequence":[[SEQ]],"trace":"[[TRACE]]"}', "userRegistryId identifies an application user."),
            _ssn_role("warn_message", "WARN the verification screen retained masked SSN [[VALUE]] during step [[SEQ]]; trace [[TRACE]]", "The masked value is explicitly the retained form of a true SSN."),
        ),
        _family(
            "train_addition", "order_fulfillment_token", "business_id_vs_ssn", "ssn_hyphenated",
            _business_role("ORDER_ID", "natural_language", "Fulfillment resumed commerce order token [[VALUE]] after inventory cleared; sequence [[SEQ]], trace [[TRACE]].", "The prose calls the value a commerce order token, which is BUSINESS_ID."),
            _ssn_role("key_equals", "SECURE identity_subject_ssn=[[VALUE]] verification=manual seq=[[SEQ]] trace=[[TRACE]]", "identity_subject_ssn explicitly marks a true SSN."),
        ),
        _family(
            "train_addition", "case_triage_locator", "business_id_vs_ssn", "ssn_compact",
            _business_role("CASE_ID", "warn_message", "WARN support case locator [[VALUE]] moved to specialist queue; sequence [[SEQ]]; trace [[TRACE]]", "The value locates a support case and is a BUSINESS_ID."),
            _ssn_role("natural_language", "The identity analyst read the claimant's nine-digit Social Security number [[VALUE]] for review [[SEQ]] (trace [[TRACE]]).", "The claimant context explicitly identifies an SSN."),
        ),
        _family(
            "train_addition", "ticket_escalation_locator", "business_id_vs_ssn", "ssn_spaced",
            _business_role("TICKET_ID", "mixed_punctuation", "ESCALATE :: ticketLocator[[VALUE]] ;; tier=fraud-review :: seq=[[SEQ]] :: trace=[[TRACE]]", "ticketLocator identifies an operational support ticket."),
            _ssn_role("json_flat", '{"event":"claimant_verification","social_security_no":"[[VALUE]]","sequence":[[SEQ]],"trace":"[[TRACE]]"}', "social_security_no explicitly identifies the value as SSN."),
        ),
        _family(
            "train_addition", "invoice_settlement_locator", "business_id_vs_ssn", "ssn_dotted",
            _business_role("INVOICE_ID", "key_colon", "billing folio: [[VALUE]] | settlement: pending | sequence: [[SEQ]] | trace: [[TRACE]]", "The billing folio is an invoice business identifier."),
            _ssn_role("error_message", "ERROR benefits export included employee Social Security No. [[VALUE]] at sequence [[SEQ]]; trace [[TRACE]]", "The benefits context explicitly identifies an employee SSN."),
        ),
        _family(
            "train_addition", "reference_registry_locator", "business_id_vs_ssn", "ssn_masked_last4",
            _business_role("REFERENCE_ID", "json_nested", '{"registry":{"businessLocator":"[[VALUE]]","domain":"returns"},"sequence":[[SEQ]],"trace":"[[TRACE]]"}', "businessLocator is a returns-domain business reference."),
            _ssn_role("natural_language", "The benefits letter displayed only the masked Social Security number [[VALUE]] for audit [[SEQ]] under trace [[TRACE]].", "The letter explicitly presents a masked SSN."),
        ),
        _family(
            "train_addition", "request_dispatch_locator", "business_id_vs_ssn", "ssn_hyphenated",
            _business_role("REQUEST_ID", "structured_audit", "AUDIT requestLocator=[[VALUE]] action=dispatch sequence=[[SEQ]] trace=[[TRACE]]", "requestLocator identifies a business request."),
            _ssn_role("key_colon", "identity subject SSN: [[VALUE]] | review: restricted | seq: [[SEQ]] | trace: [[TRACE]]", "The key names a true identity-subject SSN."),
        ),
        _family(
            "train_addition", "workflow_checkpoint_locator", "business_id_vs_ssn", "ssn_compact",
            _business_role("WORKFLOW_ID", "error_message", "ERROR workflow checkpoint [[VALUE]] stalled before approval; sequence [[SEQ]]; trace [[TRACE]]", "The value identifies a business workflow checkpoint."),
            _ssn_role("json_nested", '{"protectedIdentity":{"taxpayerSsn":"[[VALUE]]","review":"restricted"},"sequence":[[SEQ]],"trace":"[[TRACE]]"}', "taxpayerSsn explicitly carries an SSN."),
        ),

        # Training: card-shaped business identifiers paired with real cards.
        _family(
            "train_addition", "payment_trace_duality", "business_id_vs_credit_card", "card_16_luhn",
            _business_role("TRANSACTION_ID", "abbreviated_key", "INFO txnRef=[[VALUE]] state=captured seq=[[SEQ]] trace=[[TRACE]]", "txnRef is a transaction business identifier despite its card-like shape."),
            _card_role("natural_language", "The caller dictated payment card number [[VALUE]] for the declined purchase at event [[SEQ]]; trace [[TRACE]].", "The payment context identifies a true card number."),
        ),
        _family(
            "train_addition", "invoice_cardshape_locator", "business_id_vs_credit_card", "card_16_grouped_spaces",
            _business_role("INVOICE_ID", "json_flat", '{"invoiceLocator":"[[VALUE]]","state":"issued","sequence":[[SEQ]],"trace":"[[TRACE]]"}', "invoiceLocator identifies an invoice, not a payment card."),
            _card_role("warn_message", "WARN payment card [[VALUE]] appeared in a retry note; sequence [[SEQ]]; trace [[TRACE]]", "The warning explicitly calls the value a payment card."),
        ),
        _family(
            "train_addition", "workflow_panlike_checkpoint", "business_id_vs_credit_card", "card_16_grouped_hyphens",
            _business_role("WORKFLOW_ID", "structured_audit", "AUDIT|workflowToken=[[VALUE]]|stage=authorize|sequence=[[SEQ]]|trace=[[TRACE]]", "workflowToken identifies a workflow, despite PAN-like grouping."),
            _card_role("json_nested", '{"payment":{"cardNumber":"[[VALUE]]","purpose":"refund"},"sequence":[[SEQ]],"trace":"[[TRACE]]"}', "cardNumber in the payment object is a credit card number."),
        ),
        _family(
            "train_addition", "request_numeric_token", "business_id_vs_credit_card", "card_16_luhn",
            _business_role("REQUEST_ID", "mixed_punctuation", "REQUEST :: token[[VALUE]] ;; queue=merchant-support :: seq=[[SEQ]] :: trace=[[TRACE]]", "The merchant-support request token is BUSINESS_ID."),
            _card_role("key_colon", "payment card: [[VALUE]] | authorization: declined | seq: [[SEQ]] | trace: [[TRACE]]", "The payment-card key supplies unambiguous card semantics."),
        ),
        _family(
            "train_addition", "ticket_paymentlike_locator", "business_id_vs_credit_card", "card_16_grouped_spaces",
            _business_role("TICKET_ID", "natural_language", "Customer care reopened ticket locator [[VALUE]] for incident [[SEQ]] with trace [[TRACE]].", "The value identifies a customer-care ticket."),
            _card_role("error_message", "ERROR card processor echoed primary card number [[VALUE]] during event [[SEQ]]; trace [[TRACE]]", "The processor context explicitly identifies a card number."),
        ),
        _family(
            "train_addition", "case_panlike_locator", "business_id_vs_credit_card", "card_16_luhn",
            _business_role("CASE_ID", "json_nested", '{"caseManagement":{"caseLocator":"[[VALUE]]","priority":"high"},"sequence":[[SEQ]],"trace":"[[TRACE]]"}', "caseLocator belongs to case management and is BUSINESS_ID."),
            _card_role("key_equals", "SECURE cardNumber=[[VALUE]] purpose=chargeback seq=[[SEQ]] trace=[[TRACE]]", "cardNumber explicitly identifies a payment card."),
        ),

        # Training: account-shaped business identifiers paired with bank data.
        _family(
            "train_addition", "customer_depositshape_locator", "business_id_vs_bank_account", "account_12_digits",
            _business_role("CUSTOMER_ID", "key_equals", "INFO customerLedgerKey=[[VALUE]] segment=retail seq=[[SEQ]] trace=[[TRACE]]", "customerLedgerKey identifies the customer record, not a deposit account."),
            _bank_role("natural_language", "The refund destination was bank account [[VALUE]] for event [[SEQ]] under trace [[TRACE]].", "The refund destination explicitly identifies a bank account."),
        ),
        _family(
            "train_addition", "account_registry_numeric", "business_id_vs_bank_account", "account_10_digits",
            _business_role("ACCOUNT_ID", "json_nested", '{"applicationAccount":{"registryId":"[[VALUE]]","tier":"gold"},"sequence":[[SEQ]],"trace":"[[TRACE]]"}', "registryId identifies an application account rather than bank-account data."),
            _bank_role("structured_audit", "AUDIT bankAccount=[[VALUE]] operation=direct-deposit sequence=[[SEQ]] trace=[[TRACE]]", "bankAccount explicitly marks a domestic bank account."),
        ),
        _family(
            "train_addition", "application_ledgerlike_locator", "business_id_vs_bank_account", "account_14_digits",
            _business_role("APPLICATION_ID", "warn_message", "WARN application locator [[VALUE]] awaited underwriting; sequence [[SEQ]]; trace [[TRACE]]", "The value locates an underwriting application."),
            _bank_role("json_flat", '{"event":"payout","destinationAccount":"[[VALUE]]","sequence":[[SEQ]],"trace":"[[TRACE]]"}', "destinationAccount is bank-account data in a payout event."),
        ),
        _family(
            "train_addition", "user_bankshape_locator", "business_id_vs_bank_account", "account_grouped_hyphens",
            _business_role("USER_ID", "key_colon", "platform user locator: [[VALUE]] | state: enabled | seq: [[SEQ]] | trace: [[TRACE]]", "The platform-user locator is BUSINESS_ID."),
            _bank_role("error_message", "ERROR treasury attempted debit from bank account [[VALUE]] at sequence [[SEQ]]; trace [[TRACE]]", "The treasury debit context identifies a bank account."),
        ),

        # Training: valid IPv4 morphology in network versus explicitly
        # non-network technical-reference contexts.
        _family(
            "train_addition", "client_release_quad", "ip_address_vs_technical_reference", "ipv4_dotted_quad",
            _ip_role("key_equals", "INFO clientIp=[[VALUE]] decision=challenge seq=[[SEQ]] trace=[[TRACE]]", "clientIp explicitly records a network address."),
            _technical_role("natural_language", "Release engineering promoted build coordinate [[VALUE]] during rollout [[SEQ]] with trace [[TRACE]].", "BUILD_COORDINATE"),
        ),
        _family(
            "train_addition", "gateway_build_quad", "ip_address_vs_technical_reference", "ipv4_dotted_quad",
            _ip_role("json_nested", '{"network":{"gatewayAddress":"[[VALUE]]","action":"blocked"},"sequence":[[SEQ]],"trace":"[[TRACE]]"}', "gatewayAddress explicitly identifies an IP address."),
            _technical_role("warn_message", "WARN artifact revision [[VALUE]] failed compatibility gate [[SEQ]]; trace [[TRACE]]", "ARTIFACT_REVISION"),
        ),
        _family(
            "train_addition", "origin_version_quad", "ip_address_vs_technical_reference", "ipv4_dotted_quad",
            _ip_role("error_message", "ERROR connection from origin address [[VALUE]] was rejected; sequence [[SEQ]]; trace [[TRACE]]", "origin address has explicit network semantics."),
            _technical_role("json_flat", '{"releaseCoordinate":"[[VALUE]]","channel":"canary","sequence":[[SEQ]],"trace":"[[TRACE]]"}', "RELEASE_COORDINATE"),
        ),
        _family(
            "train_addition", "peer_artifact_quad", "ip_address_vs_technical_reference", "ipv4_dotted_quad",
            _ip_role("structured_audit", "AUDIT peerAddress=[[VALUE]] transport=tls sequence=[[SEQ]] trace=[[TRACE]]", "peerAddress explicitly records a network peer."),
            _technical_role("mixed_punctuation", "PACKAGE :: revision[[VALUE]] ;; lane=staging :: seq=[[SEQ]] :: trace=[[TRACE]]", "PACKAGE_REVISION"),
        ),

        # Training-only contextual breadth for two additional error classes.
        _family(
            "train_addition", "operator_service_name", "person_name_vs_service_name", "two_token_name",
            _role("person_identity", "PERSON_NAME", "PERSON_NAME", "natural_language", "On-call analyst [[VALUE]] approved recovery event [[SEQ]] under trace [[TRACE]].", "The grammar and analyst role identify a real person."),
            _role("service_actor", "O", "SERVICE_ALIAS", "json_flat", '{"serviceAlias":"[[VALUE]]","operation":"cache_refresh","sequence":[[SEQ]],"trace":"[[TRACE]]"}', "The same name-shaped text is explicitly a non-person service alias."),
        ),
        _family(
            "train_addition", "approver_job_alias", "person_name_vs_service_name", "two_token_name",
            _role("person_identity", "PERSON_NAME", "PERSON_NAME", "warn_message", "WARN approver [[VALUE]] requested a second review for event [[SEQ]]; trace [[TRACE]]", "approver refers to a person participating in the review."),
            _role("service_actor", "O", "JOB_ALIAS", "key_equals", "INFO jobAlias=\"[[VALUE]]\" task=nightly_rollup seq=[[SEQ]] trace=[[TRACE]]", "jobAlias explicitly denotes a non-person software job."),
        ),
        _family(
            "train_addition", "delivery_route_descriptor", "street_address_vs_system_location", "street_like_phrase",
            _role("street_location", "STREET_ADDRESS", "STREET_ADDRESS", "natural_language", "Send the replacement package to [[VALUE]] after event [[SEQ]]; trace [[TRACE]].", "Delivery language identifies a physical street address."),
            _role("system_location", "O", "ROUTE_DESCRIPTOR", "structured_audit", "AUDIT routeDescriptor=\"[[VALUE]]\" topology=logical sequence=[[SEQ]] trace=[[TRACE]]", "routeDescriptor explicitly marks a logical system route, not a physical address."),
        ),
        _family(
            "train_addition", "residence_node_alias", "street_address_vs_system_location", "street_like_phrase",
            _role("street_location", "STREET_ADDRESS", "STREET_ADDRESS", "json_nested", '{"delivery":{"streetAddress":"[[VALUE]]","method":"courier"},"sequence":[[SEQ]],"trace":"[[TRACE]]"}', "streetAddress in a delivery object is a physical address."),
            _role("system_location", "O", "NODE_ALIAS", "error_message", "ERROR compute node alias [[VALUE]] missed heartbeat [[SEQ]]; trace [[TRACE]]", "The text explicitly calls the street-shaped phrase a compute-node alias."),
        ),

        # Dev challenge: six independent BUSINESS_ID/SSN families, four
        # financial-morphology families, and two network conflicts.
        _family(
            "dev_challenge", "customer_identity_receipt", "business_id_vs_ssn", "ssn_hyphenated",
            _business_role("CUSTOMER_ID", "json_flat", '{"customerReceipt":"[[VALUE]]","event":"profile_merge","sequence":[[SEQ]],"trace":"[[TRACE]]"}', "customerReceipt identifies a customer business record."),
            _ssn_role("natural_language", "A compliance reviewer confirmed the beneficiary Social Security number [[VALUE]] in event [[SEQ]]; trace [[TRACE]].", "The beneficiary identity context identifies an SSN."),
        ),
        _family(
            "dev_challenge", "application_claim_locator", "business_id_vs_ssn", "ssn_compact",
            _business_role("APPLICATION_ID", "structured_audit", "AUDIT applicationClaim=[[VALUE]] stage=eligibility sequence=[[SEQ]] trace=[[TRACE]]", "applicationClaim identifies a business application."),
            _ssn_role("json_nested", '{"claimant":{"ssnDigits":"[[VALUE]]","verification":"pending"},"sequence":[[SEQ]],"trace":"[[TRACE]]"}', "ssnDigits is explicitly claimant SSN data."),
        ),
        _family(
            "dev_challenge", "case_identity_locator", "business_id_vs_ssn", "ssn_masked_last4",
            _business_role("CASE_ID", "warn_message", "WARN dispute case token [[VALUE]] entered mediation; sequence [[SEQ]]; trace [[TRACE]]", "The dispute case token is BUSINESS_ID."),
            _ssn_role("key_colon", "masked claimant SSN: [[VALUE]] | audit: [[SEQ]] | trace: [[TRACE]]", "The key explicitly identifies a masked SSN."),
        ),
        _family(
            "dev_challenge", "invoice_identity_locator", "business_id_vs_ssn", "ssn_spaced",
            _business_role("INVOICE_ID", "natural_language", "Collections assigned billing document token [[VALUE]] to cycle [[SEQ]] with trace [[TRACE]].", "The billing document token identifies an invoice."),
            _ssn_role("error_message", "ERROR payroll note contained worker SSN [[VALUE]] during event [[SEQ]]; trace [[TRACE]]", "The payroll context gives explicit SSN semantics."),
        ),
        _family(
            "dev_challenge", "request_identity_locator", "business_id_vs_ssn", "ssn_dotted",
            _business_role("REQUEST_ID", "abbreviated_key", "INFO reqToken=[[VALUE]] channel=partner seq=[[SEQ]] trace=[[TRACE]]", "reqToken is an application request identifier."),
            _ssn_role("mixed_punctuation", "IDENTITY :: Social-Security-No [[VALUE]] ;; seq=[[SEQ]] :: trace=[[TRACE]]", "The identity label explicitly identifies an SSN."),
        ),
        _family(
            "dev_challenge", "workflow_identity_locator", "business_id_vs_ssn", "ssn_hyphenated",
            _business_role("WORKFLOW_ID", "json_nested", '{"orchestration":{"runLocator":"[[VALUE]]","state":"paused"},"sequence":[[SEQ]],"trace":"[[TRACE]]"}', "runLocator identifies a business workflow."),
            _ssn_role("warn_message", "WARN applicant Social Security identifier [[VALUE]] entered an unprotected note at event [[SEQ]]; trace [[TRACE]]", "The applicant context explicitly identifies an SSN."),
        ),
        _family(
            "dev_challenge", "transaction_cardshape_receipt", "business_id_vs_credit_card", "card_16_luhn",
            _business_role("TRANSACTION_ID", "key_equals", "INFO commerceTxn=[[VALUE]] settlement=batch seq=[[SEQ]] trace=[[TRACE]]", "commerceTxn is a transaction identifier."),
            _card_role("json_nested", '{"checkout":{"primaryCard":"[[VALUE]]","result":"declined"},"sequence":[[SEQ]],"trace":"[[TRACE]]"}', "primaryCard in checkout is card data."),
        ),
        _family(
            "dev_challenge", "ticket_cardshape_receipt", "business_id_vs_credit_card", "card_16_grouped_spaces",
            _business_role("TICKET_ID", "error_message", "ERROR incident ticket token [[VALUE]] could not be routed; sequence [[SEQ]]; trace [[TRACE]]", "The incident ticket token is BUSINESS_ID."),
            _card_role("natural_language", "The merchant captured payment card [[VALUE]] for retry [[SEQ]] under trace [[TRACE]].", "The merchant payment context identifies a card."),
        ),
        _family(
            "dev_challenge", "account_depositshape_receipt", "business_id_vs_bank_account", "account_12_digits",
            _business_role("ACCOUNT_ID", "json_flat", '{"platformAccountKey":"[[VALUE]]","operation":"upgrade","sequence":[[SEQ]],"trace":"[[TRACE]]"}', "platformAccountKey identifies an application account."),
            _bank_role("structured_audit", "AUDIT depositAccount=[[VALUE]] action=credit sequence=[[SEQ]] trace=[[TRACE]]", "depositAccount explicitly identifies bank-account data."),
        ),
        _family(
            "dev_challenge", "user_bankshape_receipt", "business_id_vs_bank_account", "account_grouped_hyphens",
            _business_role("USER_ID", "key_colon", "member registry key: [[VALUE]] | action: unlock | seq: [[SEQ]] | trace: [[TRACE]]", "The member registry key is a business user identifier."),
            _bank_role("warn_message", "WARN withdrawal used bank account [[VALUE]] during event [[SEQ]]; trace [[TRACE]]", "The withdrawal context identifies a bank account."),
        ),
        _family(
            "dev_challenge", "deployment_network_quad", "ip_address_vs_technical_reference", "ipv4_dotted_quad",
            _ip_role("natural_language", "The security gateway observed client network address [[VALUE]] for event [[SEQ]] under trace [[TRACE]].", "The network-address phrase identifies an IP address."),
            _technical_role("json_flat", '{"deploymentStamp":"[[VALUE]]","ring":"blue","sequence":[[SEQ]],"trace":"[[TRACE]]"}', "DEPLOYMENT_STAMP"),
        ),
        _family(
            "dev_challenge", "protocol_artifact_quad", "ip_address_vs_technical_reference", "ipv4_dotted_quad",
            _ip_role("json_nested", '{"connection":{"remoteIp":"[[VALUE]]","policy":"deny"},"sequence":[[SEQ]],"trace":"[[TRACE]]"}', "remoteIp explicitly identifies an IP address."),
            _technical_role("error_message", "ERROR protocol artifact label [[VALUE]] was unsupported at step [[SEQ]]; trace [[TRACE]]", "PROTOCOL_ARTIFACT"),
        ),

        # Sealed challenge: independent wording and families.  The mix includes
        # free prose and structured key/value and JSON forms.
        _family(
            "sealed_challenge", "shipment_identity_folio", "business_id_vs_ssn", "ssn_hyphenated",
            _business_role("ORDER_ID", "structured_audit", "AUDIT shipmentFolio=[[VALUE]] action=pack sequence=[[SEQ]] trace=[[TRACE]]", "shipmentFolio identifies a commerce order."),
            _ssn_role("natural_language", "The pension specialist verified the retiree's Social Security number [[VALUE]] for event [[SEQ]] under trace [[TRACE]].", "The retiree identity context explicitly identifies an SSN."),
        ),
        _family(
            "sealed_challenge", "registry_identity_folio", "business_id_vs_ssn", "ssn_compact",
            _business_role("REFERENCE_ID", "json_nested", '{"returnsRegistry":{"locator":"[[VALUE]]","state":"accepted"},"sequence":[[SEQ]],"trace":"[[TRACE]]"}', "The returns-registry locator is BUSINESS_ID."),
            _ssn_role("key_colon", "employee social security digits: [[VALUE]] | event: [[SEQ]] | trace: [[TRACE]]", "The employee key identifies an SSN."),
        ),
        _family(
            "sealed_challenge", "incident_identity_folio", "business_id_vs_ssn", "ssn_masked_last4",
            _business_role("TICKET_ID", "warn_message", "WARN incident folio [[VALUE]] crossed the response threshold; sequence [[SEQ]]; trace [[TRACE]]", "incident folio identifies a support ticket."),
            _ssn_role("json_flat", '{"maskedSsn":"[[VALUE]]","purpose":"identity_display","sequence":[[SEQ]],"trace":"[[TRACE]]"}', "maskedSsn explicitly carries a masked SSN."),
        ),
        _family(
            "sealed_challenge", "member_identity_folio", "business_id_vs_ssn", "ssn_spaced",
            _business_role("USER_ID", "mixed_punctuation", "MEMBER :: registry-folio[[VALUE]] ;; status=active :: seq=[[SEQ]] :: trace=[[TRACE]]", "The registry folio identifies an application user."),
            _ssn_role("error_message", "ERROR survivor-benefit form exposed beneficiary SSN [[VALUE]] at event [[SEQ]]; trace [[TRACE]]", "The benefit form context identifies an SSN."),
        ),
        _family(
            "sealed_challenge", "submission_identity_folio", "business_id_vs_ssn", "ssn_dotted",
            _business_role("APPLICATION_ID", "natural_language", "Underwriting resumed submission folio [[VALUE]] in cycle [[SEQ]] with trace [[TRACE]].", "The submission folio is an application identifier."),
            _ssn_role("json_nested", '{"taxIdentity":{"socialSecurityNo":"[[VALUE]]","status":"verified"},"sequence":[[SEQ]],"trace":"[[TRACE]]"}', "socialSecurityNo explicitly identifies an SSN."),
        ),
        _family(
            "sealed_challenge", "invoice_payment_folio", "business_id_vs_credit_card", "card_16_grouped_hyphens",
            _business_role("INVOICE_ID", "key_equals", "INFO receivableFolio=[[VALUE]] state=open seq=[[SEQ]] trace=[[TRACE]]", "receivableFolio identifies an invoice."),
            _card_role("warn_message", "WARN checkout retained card number [[VALUE]] in event [[SEQ]]; trace [[TRACE]]", "The checkout context explicitly identifies card data."),
        ),
        _family(
            "sealed_challenge", "transaction_payment_folio", "business_id_vs_credit_card", "card_16_luhn",
            _business_role("TRANSACTION_ID", "json_flat", '{"commerceEventKey":"[[VALUE]]","state":"settled","sequence":[[SEQ]],"trace":"[[TRACE]]"}', "commerceEventKey identifies a transaction."),
            _card_role("natural_language", "A payment agent entered credit card [[VALUE]] for authorization [[SEQ]] under trace [[TRACE]].", "The payment-agent context identifies a credit card."),
        ),
        _family(
            "sealed_challenge", "customer_deposit_folio", "business_id_vs_bank_account", "account_14_digits",
            _business_role("CUSTOMER_ID", "error_message", "ERROR customer registry folio [[VALUE]] failed synchronization; sequence [[SEQ]]; trace [[TRACE]]", "The registry folio identifies a customer."),
            _bank_role("json_nested", '{"transfer":{"bankAccountNumber":"[[VALUE]]","direction":"outbound"},"sequence":[[SEQ]],"trace":"[[TRACE]]"}', "bankAccountNumber explicitly identifies bank-account data."),
        ),
        _family(
            "sealed_challenge", "request_deposit_folio", "business_id_vs_bank_account", "account_10_digits",
            _business_role("REQUEST_ID", "abbreviated_key", "INFO rqFolio=[[VALUE]] queue=fulfillment seq=[[SEQ]] trace=[[TRACE]]", "rqFolio identifies a business request."),
            _bank_role("natural_language", "Treasury confirmed bank account [[VALUE]] for disbursement [[SEQ]] with trace [[TRACE]].", "The disbursement context identifies a bank account."),
        ),
        _family(
            "sealed_challenge", "edge_release_quad", "ip_address_vs_technical_reference", "ipv4_dotted_quad",
            _ip_role("structured_audit", "AUDIT edgeClientAddress=[[VALUE]] verdict=throttle sequence=[[SEQ]] trace=[[TRACE]]", "edgeClientAddress explicitly records an IP address."),
            _technical_role("natural_language", "The release board approved package coordinate [[VALUE]] for wave [[SEQ]] under trace [[TRACE]].", "PACKAGE_COORDINATE"),
        ),
        _family(
            "sealed_challenge", "socket_schema_quad", "ip_address_vs_technical_reference", "ipv4_dotted_quad",
            _ip_role("error_message", "ERROR socket peer IP [[VALUE]] failed attestation at event [[SEQ]]; trace [[TRACE]]", "socket peer IP explicitly identifies a network address."),
            _technical_role("json_nested", '{"schema":{"revisionTuple":"[[VALUE]]","compatibility":"strict"},"sequence":[[SEQ]],"trace":"[[TRACE]]"}', "SCHEMA_REVISION"),
        ),
        _family(
            "sealed_challenge", "proxy_artifact_quad", "ip_address_vs_technical_reference", "ipv4_dotted_quad",
            _ip_role("json_flat", '{"proxySourceIp":"[[VALUE]]","decision":"allow","sequence":[[SEQ]],"trace":"[[TRACE]]"}', "proxySourceIp explicitly identifies an IP address."),
            _technical_role("mixed_punctuation", "ARTIFACT :: compatibility[[VALUE]] ;; lane=beta :: seq=[[SEQ]] :: trace=[[TRACE]]", "COMPATIBILITY_MARKER"),
        ),
    ]
    return tuple(families)


def _validate_catalog(entries: Sequence[FamilySpec]) -> None:
    expected_family_counts = {
        "train_addition": 30,
        "dev_challenge": 12,
        "sealed_challenge": 12,
    }
    seen = set()
    counts = {partition: 0 for partition in PARTITIONS}
    styles = set()
    subtypes = set()

    for spec in entries:
        if spec.template_family in seen:
            raise AssertionError(f"Duplicate template family: {spec.template_family}")
        seen.add(spec.template_family)
        if spec.intended_split not in PARTITIONS:
            raise AssertionError(f"Unknown partition: {spec.intended_split}")
        counts[spec.intended_split] += 1
        if spec.pair_count != PAIR_COUNTS[spec.intended_split]:
            raise AssertionError(f"Wrong pair count for {spec.template_family}")
        if len(spec.roles) != 2:
            raise AssertionError(f"Contrast family must have two roles: {spec.template_family}")
        if len({role.context_role for role in spec.roles}) != 2:
            raise AssertionError(f"Contrast roles must be distinct: {spec.template_family}")
        if FORBIDDEN_PARENT_FAMILY in spec.template_family:
            raise AssertionError("Historical ML-v1 family name leaked into v1.1")
        for role in spec.roles:
            styles.add(role.surface_style)
            if role.surface_style not in SURFACE_STYLE_TO_FORMAT:
                raise AssertionError(f"Unknown surface style: {role.surface_style}")
            if role.expected_label not in CANONICAL_TARGET_LABELS:
                raise AssertionError(f"Unsupported label: {role.expected_label}")
            if role.template.count(VALUE_TOKEN) != 1:
                raise AssertionError(f"Template must contain one value token: {spec.template_family}")
            if role.companion_label:
                if role.companion_label not in CANONICAL_TARGET_LABELS - {"O"}:
                    raise AssertionError(f"Unsupported companion label: {role.companion_label}")
                if not role.companion_source_label:
                    raise AssertionError(f"Companion source label missing: {spec.template_family}")
                if role.template.count(COMPANION_TOKEN) != 1:
                    raise AssertionError(f"Companion token missing: {spec.template_family}")
                if role.template.index(COMPANION_TOKEN) < role.template.index(VALUE_TOKEN):
                    raise AssertionError(f"Companion must follow paired value: {spec.template_family}")
            elif COMPANION_TOKEN in role.template:
                raise AssertionError(f"Unexpected companion token: {spec.template_family}")
            if SEQUENCE_TOKEN not in role.template or TRACE_TOKEN not in role.template:
                raise AssertionError(f"Template lacks deterministic audit tokens: {spec.template_family}")
            if FORBIDDEN_PARENT_TEXT_FRAGMENT.casefold() in role.template.casefold():
                raise AssertionError("Historical ML-v1 wording leaked into v1.1")
            if role.expected_label == "BUSINESS_ID":
                if role.source_label not in BUSINESS_ID_SUBTYPES:
                    raise AssertionError(f"Missing BUSINESS_ID subtype: {spec.template_family}")
                subtypes.add(role.source_label)

    if counts != expected_family_counts:
        raise AssertionError(f"Unexpected family counts: {counts!r}")
    if subtypes != set(BUSINESS_ID_SUBTYPES):
        raise AssertionError("Catalog does not cover all required BUSINESS_ID subtypes")
    train_styles = {
        role.surface_style
        for spec in entries
        if spec.intended_split == "train_addition"
        for role in spec.roles
    }
    if len(train_styles) < 8:
        raise AssertionError("Training catalog needs at least eight surface styles")

    for partition in ("dev_challenge", "sealed_challenge"):
        partition_specs = [spec for spec in entries if spec.intended_split == partition]
        business_ssn = sum(spec.contrast_category == "business_id_vs_ssn" for spec in partition_specs)
        business_financial = sum(
            spec.contrast_category in {
                "business_id_vs_credit_card",
                "business_id_vs_bank_account",
            }
            for spec in partition_specs
        )
        network = sum(
            spec.contrast_category == "ip_address_vs_technical_reference"
            for spec in partition_specs
        )
        if business_ssn < 5 or business_financial < 3 or network < 2:
            raise AssertionError(f"Challenge coverage is insufficient for {partition}")


def catalog() -> Tuple[FamilySpec, ...]:
    """Return the immutable, leakage-isolated v1.1 family catalog."""

    entries = _catalog_entries()
    _validate_catalog(entries)
    return tuple(sorted(entries, key=lambda spec: spec.template_family))


def _derived_rng(seed: int, *parts: object) -> random.Random:
    material = "\x1f".join((GENERATOR_VERSION, str(seed), *(str(part) for part in parts)))
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _digits(rng: random.Random, count: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(count))


def _valid_ssn_parts(rng: random.Random) -> Tuple[str, str, str]:
    area = rng.randint(1, 899)
    while area == 666:
        area = rng.randint(1, 899)
    return f"{area:03d}", f"{rng.randint(1, 99):02d}", f"{rng.randint(1, 9999):04d}"


def _luhn_check_digit(prefix: str) -> str:
    for candidate in string.digits:
        number = prefix + candidate
        total = 0
        parity = len(number) % 2
        for index, char in enumerate(number):
            digit = int(char)
            if index % 2 == parity:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        if total % 10 == 0:
            return candidate
    raise AssertionError("Unable to construct Luhn-valid value")


def _value_for_morphology(morphology: str, rng: random.Random) -> str:
    if morphology.startswith("ssn_"):
        area, group, serial = _valid_ssn_parts(rng)
        if morphology == "ssn_hyphenated":
            return f"{area}-{group}-{serial}"
        if morphology == "ssn_compact":
            return area + group + serial
        if morphology == "ssn_spaced":
            return f"{area} {group} {serial}"
        if morphology == "ssn_dotted":
            return f"{area}.{group}.{serial}"
        if morphology == "ssn_masked_last4":
            return f"***-**-{serial}"
    if morphology.startswith("card_16_"):
        prefix = rng.choice(("4", "51", "52", "53", "54", "55"))
        first_fifteen = prefix + _digits(rng, 15 - len(prefix))
        digits = first_fifteen + _luhn_check_digit(first_fifteen)
        if morphology == "card_16_luhn":
            return digits
        if morphology == "card_16_grouped_spaces":
            return " ".join(digits[index:index + 4] for index in range(0, 16, 4))
        if morphology == "card_16_grouped_hyphens":
            return "-".join(digits[index:index + 4] for index in range(0, 16, 4))
    if morphology == "account_10_digits":
        return _digits(rng, 10)
    if morphology == "account_12_digits":
        return _digits(rng, 12)
    if morphology == "account_14_digits":
        return _digits(rng, 14)
    if morphology == "account_grouped_hyphens":
        digits = _digits(rng, 12)
        return f"{digits[:4]}-{digits[4:8]}-{digits[8:]}"
    if morphology == "ipv4_dotted_quad":
        # Valid IPv4 values are required in both network and non-network roles;
        # semantic context, not parser validity, determines the annotation.
        return ".".join(
            str(part)
            for part in (
                rng.randint(11, 223),
                rng.randint(0, 255),
                rng.randint(0, 255),
                rng.randint(1, 254),
            )
        )
    if morphology == "two_token_name":
        first = rng.choice(("Avery", "Camila", "Darius", "Fatima", "Jordan", "Mateo", "Nora", "Priya"))
        last = rng.choice(("Bennett", "Haddad", "Ibrahim", "Nguyen", "Okafor", "Patel", "Rivera", "Sato"))
        return f"{first} {last}"
    if morphology == "street_like_phrase":
        number = rng.randint(10, 9999)
        street = rng.choice(("Cedar Avenue", "Harbor Drive", "Juniper Lane", "Market Street", "Willow Road"))
        return f"{number} {street}"
    raise ValueError(f"Unsupported morphology class: {morphology}")


def _slug(value: str) -> str:
    return _SLUG_RE.sub("-", value.casefold()).strip("-")


def _render_record(
    spec: FamilySpec,
    role: RoleSpec,
    value: str,
    group_index: int,
    seed: int,
) -> Dict[str, object]:
    role_rng = _derived_rng(
        seed,
        spec.intended_split,
        spec.template_family,
        group_index,
        role.context_role,
    )
    companion_value: Optional[str] = None
    if role.companion_label == "PERSON_NAME":
        first = role_rng.choice(
            ("Avery", "Camila", "Darius", "Fatima", "Jordan", "Mateo", "Nora", "Priya")
        )
        last = role_rng.choice(
            ("Bennett", "Haddad", "Ibrahim", "Nguyen", "Okafor", "Patel", "Rivera", "Sato")
        )
        companion_value = f"{first} {last}"
    elif role.companion_label is not None:
        raise ValueError(f"Unsupported companion label: {role.companion_label}")
    # Keep this token numeric so templates marked ``format=json`` remain valid
    # JSON after substitution; uniqueness also comes from the family-scoped
    # trace and source_record_id.
    sequence = str(group_index)
    trace = hashlib.sha256(
        f"{GENERATOR_VERSION}|{seed}|{spec.template_family}|{group_index}|{role.context_role}|{role_rng.getrandbits(64)}".encode("utf-8")
    ).hexdigest()[:20]

    template = role.template.replace(SEQUENCE_TOKEN, sequence).replace(TRACE_TOKEN, trace)
    start = template.index(VALUE_TOKEN)
    text = template.replace(VALUE_TOKEN, value)
    end = start + len(value)
    companion_span: Optional[Tuple[int, int]] = None
    if companion_value is not None:
        companion_start = text.index(COMPANION_TOKEN)
        text = text.replace(COMPANION_TOKEN, companion_value)
        companion_span = (companion_start, companion_start + len(companion_value))
    value_hash = hashlib.sha256(value.encode("utf-8")).hexdigest()

    entities: List[List[object]] = []
    provenance: List[Dict[str, object]] = []
    if role.expected_label != "O":
        entities.append([start, end, role.expected_label])
        provenance_entry: Dict[str, object] = {
            "start": start,
            "end": end,
            "label": role.expected_label,
            "source_label": role.source_label,
            "action": "normalized" if role.expected_label == "BUSINESS_ID" else "kept",
            "training_exclusion": False,
        }
        if role.expected_label == "BUSINESS_ID":
            provenance_entry["business_id_subtype"] = role.source_label
        provenance.append(provenance_entry)
    if companion_span is not None:
        companion_start, companion_end = companion_span
        entities.append([companion_start, companion_end, role.companion_label])
        provenance.append(
            {
                "start": companion_start,
                "end": companion_end,
                "label": role.companion_label,
                "source_label": role.companion_source_label,
                "action": "kept",
                "training_exclusion": False,
            }
        )
    entities.sort(key=lambda entity: (int(entity[0]), int(entity[1]), str(entity[2])))
    provenance.sort(
        key=lambda entry: (int(entry["start"]), int(entry["end"]), str(entry["label"]))
    )

    target: Dict[str, object] = {
        "start": start,
        "end": end,
        "expected_label": role.expected_label,
        "value_sha256": value_hash,
        "source_label": role.source_label,
    }
    if role.expected_label == "BUSINESS_ID":
        target["business_id_subtype"] = role.source_label

    family_slug = _slug(spec.template_family)
    role_slug = _slug(role.context_role)
    contrast_group_id = (
        f"ml-v1.1-{spec.intended_split}-{family_slug}-g{group_index:04d}"
    )
    source_record_id = f"{contrast_group_id}-{role_slug}"

    return {
        "text": text,
        "entities": entities,
        "meta": {
            "source": SOURCE,
            "source_record_id": source_record_id,
            "format": SURFACE_STYLE_TO_FORMAT[role.surface_style],
            "surface_style": role.surface_style,
            "review_status": "synthetic_generated",
            "license_reviewed": True,
            "template_family": spec.template_family,
            "scenario_kind": "context_contrast",
            "annotation_rationale": role.rationale,
            "primary_entity": role.expected_label,
            "contrast_group_id": contrast_group_id,
            "context_role": role.context_role,
            "contrast_category": spec.contrast_category,
            "morphology_class": spec.morphology_class,
            "intended_split": spec.intended_split,
            "generator_version": GENERATOR_VERSION,
            "entity_provenance": provenance,
            "contrast_targets": [target],
        },
    }


def generate_partition(partition: str, seed: int = 42) -> List[Dict[str, object]]:
    """Generate one isolated v1.1 partition entirely in memory."""

    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {', '.join(PARTITIONS)}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    specs = sorted(
        (spec for spec in catalog() if spec.intended_split == partition),
        key=lambda spec: spec.template_family,
    )
    records: List[Dict[str, object]] = []
    for spec in specs:
        for group_index in range(spec.pair_count):
            value_rng = _derived_rng(
                seed,
                spec.intended_split,
                spec.template_family,
                group_index,
                "shared-value",
            )
            value = _value_for_morphology(spec.morphology_class, value_rng)
            for role in sorted(spec.roles, key=lambda item: item.context_role):
                records.append(_render_record(spec, role, value, group_index, seed))

    expected_records = len(specs) * PAIR_COUNTS[partition] * 2
    if len(records) != expected_records:
        raise AssertionError(
            f"Generated {len(records)} records for {partition}; expected {expected_records}"
        )
    return records


def generate_all(seed: int = 42) -> Dict[str, List[Dict[str, object]]]:
    """Generate all three isolated partitions without writing files."""

    return {
        partition: generate_partition(partition, seed=seed)
        for partition in PARTITIONS
    }


def write_jsonl(path: str | Path, records: Sequence[Mapping[str, object]]) -> None:
    """Write explicitly supplied records as canonical UTF-8 JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate isolated SecureLogX ML-v1.1 context-contrast partitions"
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument(
        "--out-train-addition", "--train-out", dest="train_out",
        help="Optional JSONL path for the training addition",
    )
    parser.add_argument(
        "--out-dev-challenge", "--dev-out", dest="dev_out",
        help="Optional JSONL path for the dev challenge",
    )
    parser.add_argument(
        "--out-sealed-challenge", "--sealed-out", dest="sealed_out",
        help="Optional JSONL path for the sealed challenge",
    )
    args = parser.parse_args(argv)

    try:
        generated = generate_all(seed=args.seed)
        requested_paths = {
            "train_addition": args.train_out,
            "dev_challenge": args.dev_out,
            "sealed_challenge": args.sealed_out,
        }
        for partition, path in requested_paths.items():
            if path:
                write_jsonl(path, generated[partition])
    except (AssertionError, OSError, TypeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    families = catalog()
    for partition in PARTITIONS:
        family_count = sum(spec.intended_split == partition for spec in families)
        suffix = f" -> {requested_paths[partition]}" if requested_paths[partition] else " (memory only)"
        print(
            f"{partition}: {len(generated[partition])} records across "
            f"{family_count} families{suffix}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
