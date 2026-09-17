#!/usr/bin/env python3
"""Generate isolated SecureLogX ML-v1.2 counterbalance records.

The generator is deliberately independent from every frozen ML-v1 and ML-v1.1
artifact.  It never opens a parent dataset or the sealed challenge.  Public
generation functions return records in memory; JSONL is written only when a
caller explicitly supplies an output path.

Each family is bidirectional: the same deterministic surface value is used in
two different contexts.  Most families contrast a genuine BUSINESS_ID with a
canonical sensitive entity.  Dedicated IP families contrast a valid IPv4
address with a non-network technical coordinate because that was ML-v1.1's
worst development category.  Companion entities make the records genuinely
multi-entity instead of allowing one-label-per-record shortcuts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import string
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple


SOURCE = "securelogx_counterbalance_v1_2"
GENERATOR_VERSION = "securelogx-ml-v1.2-counterbalance-v1"
PARTITIONS: Tuple[str, ...] = ("train_addition", "dev_challenge")
PAIR_COUNTS: Mapping[str, int] = {
    "train_addition": 20,
    "dev_challenge": 18,
}
EXPECTED_FAMILY_COUNTS: Mapping[str, int] = {
    "train_addition": 40,
    "dev_challenge": 15,
}
EXPECTED_RECORD_COUNTS: Mapping[str, int] = {
    partition: EXPECTED_FAMILY_COUNTS[partition] * PAIR_COUNTS[partition] * 2
    for partition in PARTITIONS
}

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
        "PERSON_NAME",
        "SSN",
        "ITIN",
        "TAX_ID",
        "CREDIT_CARD_NUMBER",
        "BANK_ACCOUNT_NUMBER",
        "ROUTING_NUMBER",
        "BUSINESS_ID",
        "PASSPORT_NUMBER",
        "DRIVER_LICENSE",
        "IP_ADDRESS",
        "AUTH_TOKEN",
        "API_KEY",
        "O",
    }
)

SURFACE_STYLE_TO_FORMAT: Mapping[str, str] = {
    "countercheck_kv": "key_value",
    "countercheck_json": "json",
    "countercheck_nested_json": "json",
    "countercheck_audit": "text",
    "countercheck_syslog": "text",
    "countercheck_bracketed": "text",
    "countercheck_query": "key_value",
    "countercheck_csv": "csv",
}

VALUE_TOKEN = "[[VALUE]]"
COMPANION_TOKEN = "[[COMPANION]]"
SEQUENCE_TOKEN = "[[SEQ]]"
TRACE_TOKEN = "[[TRACE]]"
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class RoleSpec:
    """One semantic interpretation of a family's shared focus value."""

    context_role: str
    expected_label: str
    source_label: str
    surface_style: str
    template: str
    rationale: str
    companion_label: str
    companion_source_label: str
    companion_business_id_subtype: Optional[str] = None


@dataclass(frozen=True)
class FamilySpec:
    """A leakage-isolated pair of counterbalanced semantic contexts."""

    template_family: str
    intended_split: str
    contrast_category: str
    morphology_class: str
    pair_count: int
    semantic_signature: str
    business_id_subtype: str
    roles: Tuple[RoleSpec, RoleSpec]


def _template(
    style: str,
    signature: str,
    focus_field: str,
    companion_field: str,
) -> str:
    """Create a v1.2-only skeleton with the focus entity before its companion."""

    if style == "countercheck_kv":
        return (
            f"COUNTERCHECK scenario={signature} {focus_field}={VALUE_TOKEN} "
            f"{companion_field}={COMPANION_TOKEN} decision=contextual "
            f"sequence={SEQUENCE_TOKEN} correlation={TRACE_TOKEN}"
        )
    if style == "countercheck_json":
        return (
            '{"counterbalanceEvent":"' + signature + '","' + focus_field + '":"'
            + VALUE_TOKEN + '","' + companion_field + '":"' + COMPANION_TOKEN
            + '","decision":"contextual","sequence":' + SEQUENCE_TOKEN
            + ',"correlation":"' + TRACE_TOKEN + '"}'
        )
    if style == "countercheck_nested_json":
        return (
            '{"countercheck":{"scenario":"' + signature + '","evidence":{"'
            + focus_field + '":"' + VALUE_TOKEN + '","' + companion_field
            + '":"' + COMPANION_TOKEN + '"}},"sequence":' + SEQUENCE_TOKEN
            + ',"correlation":"' + TRACE_TOKEN + '"}'
        )
    if style == "countercheck_audit":
        return (
            f"Context audit {signature} recorded {focus_field} {VALUE_TOKEN} before "
            f"linked {companion_field} {COMPANION_TOKEN}; sequence {SEQUENCE_TOKEN}; "
            f"correlation {TRACE_TOKEN}."
        )
    if style == "countercheck_syslog":
        return (
            f"NOTICE counterbalance[{signature}] {focus_field}<{VALUE_TOKEN}> "
            f"{companion_field}<{COMPANION_TOKEN}> semantic=context sequence={SEQUENCE_TOKEN} "
            f"correlation={TRACE_TOKEN}"
        )
    if style == "countercheck_bracketed":
        return (
            f"CONTEXT[{signature}] FOCUS[{focus_field}:{VALUE_TOKEN}] "
            f"RELATED[{companion_field}:{COMPANION_TOKEN}] "
            f"SEQ[{SEQUENCE_TOKEN}] CORR[{TRACE_TOKEN}]"
        )
    if style == "countercheck_query":
        return (
            f"countercheck/{signature}?{focus_field}={VALUE_TOKEN}&"
            f"{companion_field}={COMPANION_TOKEN}&sequence={SEQUENCE_TOKEN}&"
            f"correlation={TRACE_TOKEN}"
        )
    if style == "countercheck_csv":
        return (
            f"COUNTERBALANCE,{signature},{focus_field},{VALUE_TOKEN},"
            f"{companion_field},{COMPANION_TOKEN},{SEQUENCE_TOKEN},{TRACE_TOKEN}"
        )
    raise ValueError(f"Unknown surface style: {style}")


def _role(
    *,
    context_role: str,
    expected_label: str,
    source_label: str,
    surface_style: str,
    semantic_signature: str,
    focus_field: str,
    companion_field: str,
    rationale: str,
    companion_label: str,
    companion_source_label: str,
    companion_business_id_subtype: Optional[str] = None,
) -> RoleSpec:
    return RoleSpec(
        context_role=context_role,
        expected_label=expected_label,
        source_label=source_label,
        surface_style=surface_style,
        template=_template(
            surface_style,
            semantic_signature,
            focus_field,
            companion_field,
        ),
        rationale=rationale,
        companion_label=companion_label,
        companion_source_label=companion_source_label,
        companion_business_id_subtype=companion_business_id_subtype,
    )


def _contrast_family(
    partition: str,
    slug: str,
    category: str,
    morphology: str,
    business_subtype: str,
    business_field: str,
    sensitive_label: str,
    sensitive_field: str,
    business_style: str,
    sensitive_style: str,
    *,
    sensitive_companion_person: bool = False,
) -> FamilySpec:
    """Build a BUSINESS_ID/sensitive-entity bidirectional contrast."""

    signature = f"cb12-{partition.replace('_', '-')}-{slug}"
    business_role = _role(
        context_role="business_identifier",
        expected_label="BUSINESS_ID",
        source_label=business_subtype,
        surface_style=business_style,
        semantic_signature=f"{signature}-business",
        focus_field=business_field,
        companion_field=f"verified{sensitive_label.title().replace('_', '')}",
        rationale=(
            f"{business_field} explicitly names a {business_subtype} business key; "
            f"its {sensitive_label}-like surface does not change that semantic role."
        ),
        companion_label=sensitive_label,
        companion_source_label=sensitive_label,
    )
    if sensitive_companion_person:
        companion_label = "PERSON_NAME"
        companion_source_label = "PERSON_NAME"
        companion_subtype = None
        companion_field = "verifiedSubjectName"
    else:
        companion_label = "BUSINESS_ID"
        companion_source_label = business_subtype
        companion_subtype = business_subtype
        companion_field = f"linked{business_subtype.title().replace('_', '')}"
    sensitive_role = _role(
        context_role=f"canonical_{sensitive_label.casefold()}",
        expected_label=sensitive_label,
        source_label=sensitive_label,
        surface_style=sensitive_style,
        semantic_signature=f"{signature}-sensitive",
        focus_field=sensitive_field,
        companion_field=companion_field,
        rationale=(
            f"{sensitive_field} explicitly names canonical {sensitive_label}; the value "
            "must not be promoted to BUSINESS_ID because of its identifier-like shape."
        ),
        companion_label=companion_label,
        companion_source_label=companion_source_label,
        companion_business_id_subtype=companion_subtype,
    )
    return FamilySpec(
        template_family=f"ml_v1_2_{partition}_{slug}_v1_2",
        intended_split=partition,
        contrast_category=category,
        morphology_class=morphology,
        pair_count=PAIR_COUNTS[partition],
        semantic_signature=signature,
        business_id_subtype=business_subtype,
        roles=(business_role, sensitive_role),
    )


def _ip_technical_family(
    partition: str,
    slug: str,
    business_subtype: str,
    ip_field: str,
    technical_field: str,
    technical_source_label: str,
    ip_style: str,
    technical_style: str,
) -> FamilySpec:
    """Build the dedicated valid-IP versus technical-coordinate counterbalance."""

    signature = f"cb12-{partition.replace('_', '-')}-{slug}"
    companion_field = f"linked{business_subtype.title().replace('_', '')}"
    ip_role = _role(
        context_role="network_address",
        expected_label="IP_ADDRESS",
        source_label="IP_ADDRESS",
        surface_style=ip_style,
        semantic_signature=f"{signature}-network",
        focus_field=ip_field,
        companion_field=companion_field,
        rationale=f"{ip_field} explicitly identifies a network endpoint and is IP_ADDRESS.",
        companion_label="BUSINESS_ID",
        companion_source_label=business_subtype,
        companion_business_id_subtype=business_subtype,
    )
    technical_role = _role(
        context_role="technical_reference",
        expected_label="O",
        source_label=technical_source_label,
        surface_style=technical_style,
        semantic_signature=f"{signature}-technical",
        focus_field=technical_field,
        companion_field=companion_field,
        rationale=(
            f"{technical_field} names a non-network {technical_source_label}; the valid "
            "dotted-quad surface remains O while the adjacent business key is annotated."
        ),
        companion_label="BUSINESS_ID",
        companion_source_label=business_subtype,
        companion_business_id_subtype=business_subtype,
    )
    return FamilySpec(
        template_family=f"ml_v1_2_{partition}_{slug}_v1_2",
        intended_split=partition,
        contrast_category="ip_address_vs_technical_reference",
        morphology_class="ipv4_dotted_quad",
        pair_count=PAIR_COUNTS[partition],
        semantic_signature=signature,
        business_id_subtype=business_subtype,
        roles=(ip_role, technical_role),
    )


def _catalog_entries() -> Tuple[FamilySpec, ...]:
    """Define the 40 train and 15 fresh-dev families explicitly."""

    c = _contrast_family
    i = _ip_technical_family
    train: List[FamilySpec] = [
        # True SSN protection: explicit, abbreviated, snake/camel, natural, JSON,
        # and nested contexts.  Every record is multi-entity; two SSN families
        # place PERSON_NAME beside SSN, while account/application families place
        # their corresponding BUSINESS_ID beside the true SSN.
        c("train_addition", "customer_social_identity_ledger", "business_id_vs_ssn", "ssn_hyphenated", "CUSTOMER_ID", "commerceCustomerKey", "SSN", "socialSecurityNumber", "countercheck_kv", "countercheck_nested_json"),
        c("train_addition", "request_benefit_intake_stamp", "business_id_vs_ssn", "ssn_compact", "REQUEST_ID", "benefitRequestStamp", "SSN", "claimant_ssn", "countercheck_json", "countercheck_audit"),
        c("train_addition", "account_claimant_registry", "business_id_vs_ssn", "ssn_spaced", "ACCOUNT_ID", "serviceAccountLocator", "SSN", "claimantSocialNo", "countercheck_syslog", "countercheck_json"),
        c("train_addition", "application_taxpayer_enrollment", "business_id_vs_ssn", "ssn_dotted", "APPLICATION_ID", "enrollmentApplicationKey", "SSN", "taxpayer_ssn", "countercheck_bracketed", "countercheck_nested_json"),
        c("train_addition", "user_employee_verification", "business_id_vs_ssn", "ssn_hyphenated", "USER_ID", "workforceUserRef", "SSN", "employeeSsn", "countercheck_query", "countercheck_kv", sensitive_companion_person=True),
        c("train_addition", "transaction_member_audit", "business_id_vs_ssn", "ssn_compact", "TRANSACTION_ID", "membershipTransactionKey", "SSN", "memberSocialSecurityNo", "countercheck_csv", "countercheck_audit"),
        c("train_addition", "case_dependent_review", "business_id_vs_ssn", "ssn_spaced", "CASE_ID", "reviewCaseFolio", "SSN", "dependent_ssn", "countercheck_nested_json", "countercheck_syslog"),
        c("train_addition", "workflow_identity_quarantine", "business_id_vs_ssn", "ssn_dotted", "WORKFLOW_ID", "identityWorkflowMarker", "SSN", "subjectSocSec", "countercheck_audit", "countercheck_query"),
        c("train_addition", "customer_subject_identity", "business_id_vs_ssn", "ssn_hyphenated", "CUSTOMER_ID", "subscriberIdentityKey", "SSN", "protectedSsn", "countercheck_json", "countercheck_bracketed", sensitive_companion_person=True),
        c("train_addition", "request_payroll_trace", "business_id_vs_ssn", "ssn_compact", "REQUEST_ID", "payrollRequestTrace", "SSN", "worker_ssn", "countercheck_kv", "countercheck_csv"),

        c("train_addition", "order_pan_settlement", "business_id_vs_credit_card", "card_16_luhn", "ORDER_ID", "settlementOrderToken", "CREDIT_CARD_NUMBER", "paymentCardNumber", "countercheck_json", "countercheck_audit"),
        c("train_addition", "transaction_card_authorization", "business_id_vs_credit_card", "card_16_grouped_spaces", "TRANSACTION_ID", "authorizationTransactionRef", "CREDIT_CARD_NUMBER", "cardholderPan", "countercheck_nested_json", "countercheck_kv"),
        c("train_addition", "invoice_card_refund", "business_id_vs_credit_card", "card_16_grouped_hyphens", "INVOICE_ID", "refundInvoiceLocator", "CREDIT_CARD_NUMBER", "refundCardNo", "countercheck_syslog", "countercheck_json"),
        c("train_addition", "customer_card_vault", "business_id_vs_credit_card", "card_16_luhn", "CUSTOMER_ID", "vaultCustomerIndex", "CREDIT_CARD_NUMBER", "storedCardNumber", "countercheck_bracketed", "countercheck_query"),

        c("train_addition", "account_deposit_registry", "business_id_vs_bank_account", "bank_account_12_digits", "ACCOUNT_ID", "depositRegistryAccountKey", "BANK_ACCOUNT_NUMBER", "bankAccountNumber", "countercheck_query", "countercheck_nested_json"),
        c("train_addition", "request_disbursement_account", "business_id_vs_bank_account", "bank_account_10_digits", "REQUEST_ID", "disbursementRequestCode", "BANK_ACCOUNT_NUMBER", "destinationAccountNo", "countercheck_audit", "countercheck_json"),
        c("train_addition", "workflow_ach_account", "business_id_vs_bank_account", "bank_account_grouped", "WORKFLOW_ID", "achWorkflowFolio", "BANK_ACCOUNT_NUMBER", "beneficiaryAcct", "countercheck_csv", "countercheck_kv"),

        c("train_addition", "transaction_routing_dispatch", "business_id_vs_routing_number", "routing_aba", "TRANSACTION_ID", "dispatchTransactionRouteKey", "ROUTING_NUMBER", "abaRoutingNumber", "countercheck_kv", "countercheck_nested_json"),
        c("train_addition", "request_aba_verification", "business_id_vs_routing_number", "routing_aba", "REQUEST_ID", "verificationRequestRoute", "ROUTING_NUMBER", "bankRoutingNo", "countercheck_json", "countercheck_audit"),
        c("train_addition", "account_routing_migration", "business_id_vs_routing_number", "routing_aba", "ACCOUNT_ID", "migrationAccountRoute", "ROUTING_NUMBER", "routing_number", "countercheck_syslog", "countercheck_query"),

        c("train_addition", "application_itin_review", "business_id_vs_itin", "itin_hyphenated", "APPLICATION_ID", "taxReviewApplicationKey", "ITIN", "individualTaxpayerId", "countercheck_nested_json", "countercheck_kv"),
        c("train_addition", "customer_itin_enrollment", "business_id_vs_itin", "itin_compact", "CUSTOMER_ID", "enrollmentCustomerFolio", "ITIN", "applicant_itin", "countercheck_bracketed", "countercheck_json"),
        c("train_addition", "invoice_taxpayer_filing", "business_id_vs_tax_id", "tax_ein_hyphenated", "INVOICE_ID", "filingInvoiceMarker", "TAX_ID", "employerTaxId", "countercheck_audit", "countercheck_nested_json"),
        c("train_addition", "case_employer_tax", "business_id_vs_tax_id", "tax_ein_compact", "CASE_ID", "employerCaseReference", "TAX_ID", "federal_tax_id", "countercheck_query", "countercheck_syslog"),

        c("train_addition", "ticket_passport_screening", "business_id_vs_passport", "passport_us", "TICKET_ID", "screeningTicketKey", "PASSPORT_NUMBER", "travelerPassportNo", "countercheck_json", "countercheck_audit"),
        c("train_addition", "application_travel_document", "business_id_vs_passport", "passport_us", "APPLICATION_ID", "travelApplicationLocator", "PASSPORT_NUMBER", "passport_number", "countercheck_csv", "countercheck_nested_json"),
        c("train_addition", "user_license_verification", "business_id_vs_driver_license", "driver_license_alphanumeric", "USER_ID", "licensedUserRegistryKey", "DRIVER_LICENSE", "driverLicenseNumber", "countercheck_kv", "countercheck_json"),
        c("train_addition", "customer_license_check", "business_id_vs_driver_license", "driver_license_alphanumeric", "CUSTOMER_ID", "mobilityCustomerMarker", "DRIVER_LICENSE", "operator_license", "countercheck_bracketed", "countercheck_audit"),

        c("train_addition", "workflow_api_credential", "business_id_vs_api_key", "api_key_prefixed", "WORKFLOW_ID", "integrationWorkflowKey", "API_KEY", "serviceApiKey", "countercheck_nested_json", "countercheck_query"),
        c("train_addition", "request_integration_key", "business_id_vs_api_key", "api_key_prefixed", "REQUEST_ID", "connectorRequestKey", "API_KEY", "api_key", "countercheck_syslog", "countercheck_json"),
        c("train_addition", "reference_bearer_session", "business_id_vs_auth_token", "auth_token_prefixed", "REFERENCE_ID", "sessionReferenceToken", "AUTH_TOKEN", "authorizationToken", "countercheck_audit", "countercheck_nested_json"),
        c("train_addition", "application_access_token", "business_id_vs_auth_token", "auth_token_prefixed", "APPLICATION_ID", "accessApplicationToken", "AUTH_TOKEN", "bearer_token", "countercheck_csv", "countercheck_kv"),

        c("train_addition", "case_dotted_network_locator", "business_id_vs_ip_address", "ipv4_dotted_quad", "CASE_ID", "networkCaseLocator", "IP_ADDRESS", "remoteClientIp", "countercheck_query", "countercheck_json"),
        c("train_addition", "customer_dotted_identity", "business_id_vs_ip_address", "ipv4_dotted_quad", "CUSTOMER_ID", "dottedCustomerIdentity", "IP_ADDRESS", "source_ip_address", "countercheck_bracketed", "countercheck_syslog"),

        i("train_addition", "release_version_peer", "REFERENCE_ID", "peerNetworkAddress", "releaseVersionTuple", "RELEASE_VERSION", "countercheck_json", "countercheck_kv"),
        i("train_addition", "package_coordinate_gateway", "REQUEST_ID", "gatewaySourceIp", "packageCoordinate", "PACKAGE_COORDINATE", "countercheck_nested_json", "countercheck_audit"),
        i("train_addition", "schema_revision_proxy", "APPLICATION_ID", "proxyClientAddress", "schemaRevisionQuad", "SCHEMA_REVISION", "countercheck_syslog", "countercheck_json"),
        i("train_addition", "build_tuple_origin", "WORKFLOW_ID", "originIpAddress", "buildVersionTuple", "BUILD_VERSION", "countercheck_query", "countercheck_bracketed"),
        i("train_addition", "firmware_version_client", "TICKET_ID", "clientSocketIp", "firmwareRevision", "FIRMWARE_REVISION", "countercheck_audit", "countercheck_csv"),
        i("train_addition", "dependency_coordinate_edge", "ORDER_ID", "edgeNodeAddress", "dependencyCoordinate", "DEPENDENCY_COORDINATE", "countercheck_kv", "countercheck_nested_json"),
    ]

    # Fresh dev families deliberately use new field vocabulary, semantic
    # signatures, and skeleton/style pairings.  They do not reuse train or
    # ML-v1.1 template-family names.
    dev: List[FamilySpec] = [
        c("dev_challenge", "subscriber_benefit_identity", "business_id_vs_ssn", "ssn_spaced", "CUSTOMER_ID", "subscriberBenefitIndex", "SSN", "beneficiarySocialSecurity", "countercheck_csv", "countercheck_nested_json"),
        c("dev_challenge", "petition_identity_clearance", "business_id_vs_ssn", "ssn_hyphenated", "REQUEST_ID", "clearancePetitionCode", "SSN", "subject_social_no", "countercheck_syslog", "countercheck_bracketed", sensitive_companion_person=True),
        c("dev_challenge", "shipment_card_adjudication", "business_id_vs_credit_card", "card_16_grouped_spaces", "ORDER_ID", "shipmentAdjudicationKey", "CREDIT_CARD_NUMBER", "payerCardNumber", "countercheck_query", "countercheck_audit"),
        c("dev_challenge", "portfolio_deposit_destination", "business_id_vs_bank_account", "bank_account_14_digits", "ACCOUNT_ID", "portfolioServiceKey", "BANK_ACCOUNT_NUMBER", "deposit_account_number", "countercheck_bracketed", "countercheck_json"),
        c("dev_challenge", "orchestration_aba_clearance", "business_id_vs_routing_number", "routing_aba", "WORKFLOW_ID", "orchestrationClearanceId", "ROUTING_NUMBER", "receivingBankRoute", "countercheck_audit", "countercheck_nested_json"),
        c("dev_challenge", "appeal_itin_attestation", "business_id_vs_itin", "itin_hyphenated", "CASE_ID", "appealAttestationCase", "ITIN", "taxpayer_itin", "countercheck_json", "countercheck_syslog"),
        c("dev_challenge", "citation_tax_registration", "business_id_vs_tax_id", "tax_ein_hyphenated", "REFERENCE_ID", "citationRegistrationRef", "TAX_ID", "registeredEmployerTaxNo", "countercheck_nested_json", "countercheck_query"),
        c("dev_challenge", "permit_passport_inspection", "business_id_vs_passport", "passport_us", "APPLICATION_ID", "permitApplicationRef", "PASSPORT_NUMBER", "inspectionPassport", "countercheck_kv", "countercheck_csv"),
        c("dev_challenge", "operator_license_enrollment", "business_id_vs_driver_license", "driver_license_alphanumeric", "USER_ID", "operatorEnrollmentKey", "DRIVER_LICENSE", "motoristLicenseNo", "countercheck_syslog", "countercheck_json"),
        c("dev_challenge", "exchange_api_authorization", "business_id_vs_api_key", "api_key_prefixed", "TRANSACTION_ID", "exchangeTransactionKey", "API_KEY", "partnerApiCredential", "countercheck_audit", "countercheck_bracketed"),
        c("dev_challenge", "remittance_auth_session", "business_id_vs_auth_token", "auth_token_prefixed", "INVOICE_ID", "remittanceInvoiceToken", "AUTH_TOKEN", "sessionAuthorizationToken", "countercheck_query", "countercheck_nested_json"),
        c("dev_challenge", "dispatch_dotted_locator", "business_id_vs_ip_address", "ipv4_dotted_quad", "REQUEST_ID", "dispatchDottedRequest", "IP_ADDRESS", "ingressPeerAddress", "countercheck_csv", "countercheck_kv"),
        i("dev_challenge", "protocol_revision_endpoint", "CUSTOMER_ID", "endpointClientIp", "protocolRevisionTuple", "PROTOCOL_REVISION", "countercheck_bracketed", "countercheck_audit"),
        i("dev_challenge", "artifact_version_relay", "REQUEST_ID", "relaySourceAddress", "artifactVersionCoordinate", "ARTIFACT_VERSION", "countercheck_nested_json", "countercheck_syslog"),
        i("dev_challenge", "catalog_coordinate_socket", "REFERENCE_ID", "socketRemoteIp", "catalogCoordinate", "CATALOG_COORDINATE", "countercheck_json", "countercheck_query"),
    ]
    return tuple(train + dev)


EXPECTED_CATEGORY_COUNTS: Mapping[str, Mapping[str, int]] = {
    "train_addition": {
        "business_id_vs_ssn": 10,
        "business_id_vs_credit_card": 4,
        "business_id_vs_bank_account": 3,
        "business_id_vs_routing_number": 3,
        "business_id_vs_itin": 2,
        "business_id_vs_tax_id": 2,
        "business_id_vs_passport": 2,
        "business_id_vs_driver_license": 2,
        "business_id_vs_api_key": 2,
        "business_id_vs_auth_token": 2,
        "business_id_vs_ip_address": 2,
        "ip_address_vs_technical_reference": 6,
    },
    "dev_challenge": {
        "business_id_vs_ssn": 2,
        "business_id_vs_credit_card": 1,
        "business_id_vs_bank_account": 1,
        "business_id_vs_routing_number": 1,
        "business_id_vs_itin": 1,
        "business_id_vs_tax_id": 1,
        "business_id_vs_passport": 1,
        "business_id_vs_driver_license": 1,
        "business_id_vs_api_key": 1,
        "business_id_vs_auth_token": 1,
        "business_id_vs_ip_address": 1,
        "ip_address_vs_technical_reference": 3,
    },
}


def _validate_catalog(entries: Sequence[FamilySpec]) -> None:
    family_names = set()
    signatures = set()
    templates = set()
    family_counts: Counter[str] = Counter()
    category_counts: Dict[str, Counter[str]] = {
        partition: Counter() for partition in PARTITIONS
    }
    train_subtypes: Counter[str] = Counter()

    for spec in entries:
        if spec.intended_split not in PARTITIONS:
            raise AssertionError(f"Unknown partition: {spec.intended_split}")
        if spec.template_family in family_names:
            raise AssertionError(f"Duplicate template family: {spec.template_family}")
        if spec.semantic_signature in signatures:
            raise AssertionError(f"Duplicate semantic signature: {spec.semantic_signature}")
        if not spec.template_family.startswith(f"ml_v1_2_{spec.intended_split}_"):
            raise AssertionError(f"Non-v1.2 family name: {spec.template_family}")
        if "ml_v1_1" in spec.template_family or "ml-v1.1" in spec.semantic_signature:
            raise AssertionError("ML-v1.1 family marker leaked into the v1.2 catalog")
        if spec.pair_count != PAIR_COUNTS[spec.intended_split]:
            raise AssertionError(f"Wrong pair count: {spec.template_family}")
        if len(spec.roles) != 2 or len({role.context_role for role in spec.roles}) != 2:
            raise AssertionError(f"Family does not contain two distinct roles: {spec.template_family}")
        if spec.business_id_subtype not in BUSINESS_ID_SUBTYPES:
            raise AssertionError(f"Unknown BUSINESS_ID subtype: {spec.business_id_subtype}")

        family_names.add(spec.template_family)
        signatures.add(spec.semantic_signature)
        family_counts[spec.intended_split] += 1
        category_counts[spec.intended_split][spec.contrast_category] += 1
        if spec.intended_split == "train_addition":
            train_subtypes[spec.business_id_subtype] += 1

        role_labels = {role.expected_label for role in spec.roles}
        if spec.contrast_category == "ip_address_vs_technical_reference":
            if role_labels != {"IP_ADDRESS", "O"}:
                raise AssertionError(f"Invalid IP/technical roles: {spec.template_family}")
        elif "BUSINESS_ID" not in role_labels:
            raise AssertionError(f"Missing BUSINESS_ID role: {spec.template_family}")

        for role in spec.roles:
            if role.expected_label not in CANONICAL_TARGET_LABELS:
                raise AssertionError(f"Unsupported focus label: {role.expected_label}")
            if role.companion_label not in CANONICAL_TARGET_LABELS - {"O"}:
                raise AssertionError(f"Unsupported companion label: {role.companion_label}")
            if role.surface_style not in SURFACE_STYLE_TO_FORMAT:
                raise AssertionError(f"Unknown surface style: {role.surface_style}")
            for token in (VALUE_TOKEN, COMPANION_TOKEN, SEQUENCE_TOKEN, TRACE_TOKEN):
                if role.template.count(token) != 1:
                    raise AssertionError(
                        f"Template must contain exactly one {token}: {spec.template_family}"
                    )
            if role.template.index(VALUE_TOKEN) > role.template.index(COMPANION_TOKEN):
                raise AssertionError(f"Focus must precede companion: {spec.template_family}")
            if role.template in templates:
                raise AssertionError(f"Duplicate template text: {spec.template_family}")
            if "ml_v1_1" in role.template or "ml-v1.1" in role.template:
                raise AssertionError("ML-v1.1 wording leaked into a v1.2 template")
            templates.add(role.template)

    if dict(family_counts) != dict(EXPECTED_FAMILY_COUNTS):
        raise AssertionError(f"Unexpected family counts: {dict(family_counts)!r}")
    for partition in PARTITIONS:
        if dict(category_counts[partition]) != dict(EXPECTED_CATEGORY_COUNTS[partition]):
            raise AssertionError(
                f"Unexpected category counts for {partition}: "
                f"{dict(category_counts[partition])!r}"
            )
    if set(train_subtypes) != set(BUSINESS_ID_SUBTYPES):
        raise AssertionError("Training catalog does not retain every BUSINESS_ID subtype")
    if train_subtypes["CUSTOMER_ID"] < 4 or train_subtypes["REQUEST_ID"] < 4:
        raise AssertionError("Weak CUSTOMER_ID and REQUEST_ID contexts are underrepresented")


def catalog() -> Tuple[FamilySpec, ...]:
    """Return the immutable, leakage-isolated v1.2 family catalog."""

    entries = _catalog_entries()
    _validate_catalog(entries)
    return tuple(sorted(entries, key=lambda spec: spec.template_family))


def _derived_rng(seed: int, *parts: object) -> random.Random:
    material = "\x1f".join((GENERATOR_VERSION, str(seed), *(str(part) for part in parts)))
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return random.Random(int.from_bytes(digest, "big"))


def _digits(rng: random.Random, count: int) -> str:
    return "".join(rng.choice(string.digits) for _ in range(count))


def _alnum(rng: random.Random, count: int) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(rng.choice(alphabet) for _ in range(count))


def _valid_ssn(rng: random.Random, morphology: str = "ssn_hyphenated") -> str:
    area = rng.randint(1, 899)
    while area == 666:
        area = rng.randint(1, 899)
    group = rng.randint(1, 99)
    serial = rng.randint(1, 9999)
    parts = f"{area:03d}", f"{group:02d}", f"{serial:04d}"
    if morphology == "ssn_hyphenated":
        return "-".join(parts)
    if morphology == "ssn_compact":
        return "".join(parts)
    if morphology == "ssn_spaced":
        return " ".join(parts)
    if morphology == "ssn_dotted":
        return ".".join(parts)
    raise ValueError(f"Unsupported SSN morphology: {morphology}")


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
    raise AssertionError("Unable to construct a Luhn-valid number")


def _valid_card(rng: random.Random, morphology: str = "card_16_luhn") -> str:
    prefix = rng.choice(("4", "51", "52", "53", "54", "55"))
    first_fifteen = prefix + _digits(rng, 15 - len(prefix))
    value = first_fifteen + _luhn_check_digit(first_fifteen)
    if morphology == "card_16_luhn":
        return value
    if morphology == "card_16_grouped_spaces":
        return " ".join(value[index:index + 4] for index in range(0, 16, 4))
    if morphology == "card_16_grouped_hyphens":
        return "-".join(value[index:index + 4] for index in range(0, 16, 4))
    raise ValueError(f"Unsupported card morphology: {morphology}")


def _valid_routing_number(rng: random.Random) -> str:
    first_eight = [rng.randint(0, 9) for _ in range(8)]
    weighted = (
        3 * (first_eight[0] + first_eight[3] + first_eight[6])
        + 7 * (first_eight[1] + first_eight[4] + first_eight[7])
        + first_eight[2]
        + first_eight[5]
    )
    check = (-weighted) % 10
    return "".join(str(part) for part in (*first_eight, check))


def _valid_ipv4(rng: random.Random) -> str:
    first = rng.choice((11, 23, 37, 45, 64, 68, 73, 81, 91, 104, 128, 142, 151, 172, 181, 198, 203, 211, 217))
    return ".".join(
        str(part)
        for part in (first, rng.randint(0, 255), rng.randint(0, 255), rng.randint(1, 254))
    )


def _value_for_morphology(morphology: str, rng: random.Random) -> str:
    if morphology.startswith("ssn_"):
        return _valid_ssn(rng, morphology)
    if morphology.startswith("card_16_"):
        return _valid_card(rng, morphology)
    if morphology.startswith("bank_account_"):
        if morphology == "bank_account_10_digits":
            return _digits(rng, 10)
        if morphology == "bank_account_12_digits":
            return _digits(rng, 12)
        if morphology == "bank_account_14_digits":
            return _digits(rng, 14)
        if morphology == "bank_account_grouped":
            digits = _digits(rng, 12)
            return f"{digits[:4]}-{digits[4:8]}-{digits[8:]}"
    if morphology == "routing_aba":
        return _valid_routing_number(rng)
    if morphology.startswith("itin_"):
        group = rng.randint(70, 88)
        value = f"9{rng.randint(0, 99):02d}{group:02d}{rng.randint(1, 9999):04d}"
        return f"{value[:3]}-{value[3:5]}-{value[5:]}" if morphology == "itin_hyphenated" else value
    if morphology.startswith("tax_ein_"):
        prefix = rng.choice((10, 12, 20, 27, 30, 35, 45, 47, 81, 82, 83, 84, 85, 86, 87, 88))
        value = f"{prefix:02d}{rng.randint(1, 9_999_999):07d}"
        return f"{value[:2]}-{value[2:]}" if morphology == "tax_ein_hyphenated" else value
    if morphology == "passport_us":
        return rng.choice(string.ascii_uppercase) + _digits(rng, 8)
    if morphology == "driver_license_alphanumeric":
        return rng.choice(("D", "F", "L", "M", "R", "S")) + _digits(rng, 7)
    if morphology == "api_key_prefixed":
        return "ak_live_" + _alnum(rng, 24)
    if morphology == "auth_token_prefixed":
        return "tok_" + _alnum(rng, 32)
    if morphology == "ipv4_dotted_quad":
        return _valid_ipv4(rng)
    raise ValueError(f"Unsupported morphology class: {morphology}")


_DEFAULT_LABEL_MORPHOLOGY: Mapping[str, str] = {
    "SSN": "ssn_hyphenated",
    "CREDIT_CARD_NUMBER": "card_16_luhn",
    "BANK_ACCOUNT_NUMBER": "bank_account_12_digits",
    "ROUTING_NUMBER": "routing_aba",
    "ITIN": "itin_hyphenated",
    "TAX_ID": "tax_ein_hyphenated",
    "PASSPORT_NUMBER": "passport_us",
    "DRIVER_LICENSE": "driver_license_alphanumeric",
    "API_KEY": "api_key_prefixed",
    "AUTH_TOKEN": "auth_token_prefixed",
    "IP_ADDRESS": "ipv4_dotted_quad",
}

_BUSINESS_PREFIX: Mapping[str, str] = {
    "CUSTOMER_ID": "CUS",
    "ACCOUNT_ID": "ACT",
    "APPLICATION_ID": "APP",
    "TRANSACTION_ID": "TXN",
    "USER_ID": "USR",
    "ORDER_ID": "ORD",
    "CASE_ID": "CAS",
    "TICKET_ID": "TKT",
    "INVOICE_ID": "INV",
    "REFERENCE_ID": "REF",
    "REQUEST_ID": "REQ",
    "WORKFLOW_ID": "WFL",
}


def _companion_value(role: RoleSpec, rng: random.Random) -> str:
    if role.companion_label == "BUSINESS_ID":
        subtype = role.companion_business_id_subtype
        if subtype not in _BUSINESS_PREFIX:
            raise ValueError(f"Unsupported companion BUSINESS_ID subtype: {subtype}")
        return f"{_BUSINESS_PREFIX[subtype]}-{_alnum(rng, 5).upper()}-{rng.randint(100, 999)}"
    if role.companion_label == "PERSON_NAME":
        first = rng.choice(("Amara", "Diego", "Elena", "Hassan", "Keiko", "Lucia", "Malik", "Sonia"))
        last = rng.choice(("Barreto", "Chen", "Dubois", "Farouk", "Kovacs", "Mensah", "Novak", "Yamada"))
        return f"{first} {last}"
    morphology = _DEFAULT_LABEL_MORPHOLOGY.get(role.companion_label)
    if morphology is None:
        raise ValueError(f"Unsupported companion label: {role.companion_label}")
    return _value_for_morphology(morphology, rng)


def _slug(value: str) -> str:
    return _SLUG_RE.sub("-", value.casefold()).strip("-")


def _entity_provenance(
    start: int,
    end: int,
    label: str,
    source_label: str,
    *,
    business_subtype: Optional[str] = None,
) -> Dict[str, object]:
    entry: Dict[str, object] = {
        "start": start,
        "end": end,
        "label": label,
        "source_label": source_label,
        "action": "normalized" if label == "BUSINESS_ID" else "kept",
        "training_exclusion": False,
    }
    if label == "BUSINESS_ID":
        entry["business_id_subtype"] = business_subtype or source_label
    return entry


def _contrast_target(
    start: int,
    end: int,
    expected_label: str,
    source_label: str,
    value: str,
    target_role: str,
    *,
    business_subtype: Optional[str] = None,
) -> Dict[str, object]:
    target: Dict[str, object] = {
        "start": start,
        "end": end,
        "expected_label": expected_label,
        "source_label": source_label,
        "target_role": target_role,
        "value_sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
    }
    if expected_label == "BUSINESS_ID":
        target["business_id_subtype"] = business_subtype or source_label
    return target


def _render_record(
    spec: FamilySpec,
    role: RoleSpec,
    focus_value: str,
    group_index: int,
    seed: int,
) -> Dict[str, object]:
    role_rng = _derived_rng(
        seed,
        spec.intended_split,
        spec.template_family,
        group_index,
        role.context_role,
        "companion",
    )
    companion_value = _companion_value(role, role_rng)
    trace = hashlib.sha256(
        (
            f"{GENERATOR_VERSION}|{seed}|{spec.template_family}|{group_index}|"
            f"{role.context_role}|{role_rng.getrandbits(64)}"
        ).encode("utf-8")
    ).hexdigest()[:20]

    rendered = role.template.replace(SEQUENCE_TOKEN, str(group_index)).replace(TRACE_TOKEN, trace)
    focus_start = rendered.index(VALUE_TOKEN)
    rendered = rendered.replace(VALUE_TOKEN, focus_value)
    focus_end = focus_start + len(focus_value)
    companion_start = rendered.index(COMPANION_TOKEN)
    text = rendered.replace(COMPANION_TOKEN, companion_value)
    companion_end = companion_start + len(companion_value)

    entities: List[List[object]] = []
    provenance: List[Dict[str, object]] = []
    if role.expected_label != "O":
        entities.append([focus_start, focus_end, role.expected_label])
        provenance.append(
            _entity_provenance(
                focus_start,
                focus_end,
                role.expected_label,
                role.source_label,
                business_subtype=(spec.business_id_subtype if role.expected_label == "BUSINESS_ID" else None),
            )
        )
    entities.append([companion_start, companion_end, role.companion_label])
    provenance.append(
        _entity_provenance(
            companion_start,
            companion_end,
            role.companion_label,
            role.companion_source_label,
            business_subtype=role.companion_business_id_subtype,
        )
    )
    entities.sort(key=lambda entity: (int(entity[0]), int(entity[1]), str(entity[2])))
    provenance.sort(key=lambda entry: (int(entry["start"]), int(entry["end"]), str(entry["label"])))

    targets = [
        _contrast_target(
            focus_start,
            focus_end,
            role.expected_label,
            role.source_label,
            focus_value,
            "focus",
            business_subtype=(spec.business_id_subtype if role.expected_label == "BUSINESS_ID" else None),
        ),
        _contrast_target(
            companion_start,
            companion_end,
            role.companion_label,
            role.companion_source_label,
            companion_value,
            "companion",
            business_subtype=role.companion_business_id_subtype,
        ),
    ]

    family_slug = _slug(spec.template_family)
    role_slug = _slug(role.context_role)
    contrast_group_id = f"ml-v1.2-{spec.intended_split}-{family_slug}-g{group_index:04d}"
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
            "semantic_signature": spec.semantic_signature,
            "scenario_kind": "counterbalance_context_contrast",
            "annotation_rationale": role.rationale,
            "primary_entity": role.expected_label,
            "contrast_group_id": contrast_group_id,
            "context_role": role.context_role,
            "contrast_category": spec.contrast_category,
            "morphology_class": spec.morphology_class,
            "intended_split": spec.intended_split,
            "generator_version": GENERATOR_VERSION,
            "counterbalance_objective": "identifier_shape_does_not_override_semantic_context",
            "dominant_confusion_direction": (
                "technical_reference_to_business_id"
                if spec.contrast_category == "ip_address_vs_technical_reference"
                else f"{role.expected_label.casefold()}_vs_business_id"
            ),
            "actual_worst_category_target": spec.contrast_category == "ip_address_vs_technical_reference",
            "multi_entity": len(entities) > 1,
            "independence_scope": "new_ml_v1_2_family_and_value_namespace",
            "entity_provenance": provenance,
            "contrast_targets": targets,
        },
    }


def generate_partition(partition: str, seed: int = 42) -> List[Dict[str, object]]:
    """Generate one isolated v1.2 partition entirely in memory."""

    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {', '.join(PARTITIONS)}")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    specs = [spec for spec in catalog() if spec.intended_split == partition]
    records: List[Dict[str, object]] = []
    for spec in specs:
        for group_index in range(spec.pair_count):
            value_rng = _derived_rng(
                seed,
                spec.intended_split,
                spec.template_family,
                group_index,
                "shared-focus-value",
            )
            focus_value = _value_for_morphology(spec.morphology_class, value_rng)
            for role in sorted(spec.roles, key=lambda item: item.context_role):
                records.append(_render_record(spec, role, focus_value, group_index, seed))

    if len(records) != EXPECTED_RECORD_COUNTS[partition]:
        raise AssertionError(
            f"Generated {len(records)} records for {partition}; "
            f"expected {EXPECTED_RECORD_COUNTS[partition]}"
        )
    if len({record["meta"]["source_record_id"] for record in records}) != len(records):
        raise AssertionError(f"Duplicate source_record_id in {partition}")
    return records


def generate_all(seed: int = 42) -> Dict[str, List[Dict[str, object]]]:
    """Generate train additions and fresh dev without reading any parent data."""

    return {partition: generate_partition(partition, seed=seed) for partition in PARTITIONS}


def write_jsonl(path: str | Path, records: Sequence[Mapping[str, object]]) -> None:
    """Write explicitly supplied records as canonical UTF-8 JSONL."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate isolated SecureLogX ML-v1.2 counterbalance partitions"
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic seed")
    parser.add_argument(
        "--out-train-addition",
        "--train-out",
        dest="train_out",
        help="Optional JSONL path for the 1,600-record training addition",
    )
    parser.add_argument(
        "--out-dev-challenge",
        "--dev-out",
        dest="dev_out",
        help="Optional JSONL path for the 540-record fresh development challenge",
    )
    args = parser.parse_args(argv)

    try:
        generated = generate_all(seed=args.seed)
        requested_paths = {
            "train_addition": args.train_out,
            "dev_challenge": args.dev_out,
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
