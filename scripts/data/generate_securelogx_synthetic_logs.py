#!/usr/bin/env python3
"""Generate deterministic, template-attributable SecureLogX ML-v1 logs.

The labeled output contains ordinary positives, context-challenging positives,
ordinary negatives, and explicit hard negatives. Every record has a stable
``meta.template_family`` so related templates can be split as one group.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import random
import re
import string
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


ML_V1_ENTITIES: Tuple[str, ...] = (
    "PERSON_NAME", "DOB", "AGE", "SSN", "ITIN", "TAX_ID", "EMAIL",
    "PHONE", "STREET_ADDRESS", "CITY", "STATE_PROVINCE", "POSTAL_CODE",
    "COUNTRY", "CREDIT_CARD_NUMBER", "BANK_ACCOUNT_NUMBER",
    "ROUTING_NUMBER", "IBAN", "SWIFT_BIC", "BUSINESS_ID",
    "PASSPORT_NUMBER", "DRIVER_LICENSE", "IP_ADDRESS", "DEVICE_ID",
    "AUTH_TOKEN", "API_KEY",
)

BUSINESS_ID_SUBTYPES: Tuple[str, ...] = (
    "CUSTOMER_ID", "ACCOUNT_ID", "LOAN_NUMBER", "APPLICATION_ID",
    "TRANSACTION_ID", "USER_ID",
)

BASE_TIME = datetime(2026, 1, 1, tzinfo=timezone.utc)
PLACEHOLDER_RE = re.compile(r"<<([A-Z_]+)>>")


@dataclass(frozen=True)
class TemplateSpec:
    family: str
    text_format: str
    scenario_kind: str
    rationale: str
    template: str
    target_entity: Optional[str] = None
    secondary_entity: Optional[str] = None
    noise_kind: Optional[str] = None
    noise_entity: Optional[str] = None
    noise_source_label: Optional[str] = None


# Each free-text phrase identifies the semantic role without relying on a
# machine-friendly key immediately beside the value.
ENTITY_PROFILES: Dict[str, Tuple[str, str, str, str]] = {
    "PERSON_NAME": ("person_name", "person identified in the event", "The caller introduced herself as <<TARGET>> before describing the failed transfer.", "Please ask <<TARGET>> to return the signed disclosure before close of business."),
    "DOB": ("date_of_birth", "customer birth date", "The identity check matched when the customer answered <<TARGET>>.", "Eligibility was recalculated from the birth date <<TARGET>> supplied during enrollment."),
    "AGE": ("age", "person age", "The applicant confirmed that she is <<TARGET>> years old during the recorded call.", "Coverage changes when the dependent turns <<TARGET>> next month."),
    "SSN": ("ssn", "Social Security number", "The identity response ended with <<TARGET>> and was immediately quarantined.", "A clerk pasted <<TARGET>> into the identity case narrative instead of the protected form."),
    "ITIN": ("itin", "individual taxpayer identification number", "The nonresident filer supplied <<TARGET>> while completing tax onboarding.", "Withholding review linked the applicant to <<TARGET>>."),
    "TAX_ID": ("tax_id", "business tax identifier", "The vendor's federal filing was located under <<TARGET>>.", "Remittance setup used <<TARGET>> to identify the taxable organization."),
    "EMAIL": ("email", "personal email address", "Send the revised disclosure directly to <<TARGET>> after approval.", "The customer asked us to continue the conversation at <<TARGET>>."),
    "PHONE": ("phone", "personal telephone number", "The customer can be reached after noon at <<TARGET>>.", "A callback was requested from <<TARGET>> once the dispute is assigned."),
    "STREET_ADDRESS": ("street_address", "street address", "The replacement card should be delivered to <<TARGET>>.", "The caller moved last week and now receives statements at <<TARGET>>."),
    "CITY": ("city", "city in the customer address", "The customer reported moving to <<TARGET>> during the support call.", "Delivery was redirected to the branch serving <<TARGET>>."),
    "STATE_PROVINCE": ("state_province", "state or province", "The applicant currently resides in <<TARGET>> according to the signed form.", "Tax withholding changed after the customer relocated to <<TARGET>>."),
    "POSTAL_CODE": ("postal_code", "postal code", "The final delivery zone entered by the caller was <<TARGET>>.", "Statements for the household are routed through <<TARGET>>."),
    "COUNTRY": ("country", "country of residence", "The customer confirmed permanent residence in <<TARGET>>.", "Cross-border review was triggered after the applicant selected <<TARGET>>."),
    "CREDIT_CARD_NUMBER": ("credit_card_number", "payment card number", "The payment message contained <<TARGET>> before redaction could run.", "A customer read <<TARGET>> aloud while disputing the declined purchase."),
    "BANK_ACCOUNT_NUMBER": ("bank_account_number", "domestic bank account number", "The refund was mistakenly directed to <<TARGET>>.", "The caller confirmed that deposits should arrive in <<TARGET>>."),
    "ROUTING_NUMBER": ("routing_number", "bank routing number", "The wire was rejected after using <<TARGET>> for the receiving bank.", "Direct deposit moved through <<TARGET>> according to the enrollment call."),
    "IBAN": ("iban", "international bank account number", "The international refund should settle through <<TARGET>>.", "The beneficiary confirmed <<TARGET>> during the recorded transfer review."),
    "SWIFT_BIC": ("swift_bic", "SWIFT BIC", "The beneficiary bank was identified as <<TARGET>> during wire review.", "Treasury routed the international message through <<TARGET>>."),
    "BUSINESS_ID": ("business_id", "application or business identifier", "Support resumed the customer's case under <<TARGET>>.", "Operations traced the failed workflow back to <<TARGET>>."),
    "PASSPORT_NUMBER": ("passport_number", "passport number", "The traveler presented <<TARGET>> during remote identity review.", "Border documentation associated the applicant with <<TARGET>>."),
    "DRIVER_LICENSE": ("driver_license", "driver license number", "The motorist provided <<TARGET>> while completing identity proofing.", "The uploaded license was indexed under <<TARGET>>."),
    "IP_ADDRESS": ("ip_address", "client IP address", "The suspicious session originated from <<TARGET>> shortly before lockout.", "Network telemetry placed the customer connection at <<TARGET>>."),
    "DEVICE_ID": ("device_id", "customer device identifier", "The enrolled handset reported itself as <<TARGET>> during recovery.", "Fraud review tied the mobile session to <<TARGET>>."),
    "AUTH_TOKEN": ("auth_token", "authorization token", "The diagnostic message accidentally included <<TARGET>> while retrying the request.", "A downstream proxy forwarded <<TARGET>> in the failure narrative."),
    "API_KEY": ("api_key", "API key", "The integration retried with <<TARGET>> before secret scrubbing executed.", "A connector exposed <<TARGET>> in its diagnostic response."),
}


class SecureLogXDataGenerator:
    """Seeded generator covering all canonical labels and confusion cases."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.first_names = ("Avery", "Jordan", "Morgan", "Priya", "Mateo", "Nora", "Elias", "Camila", "Zoe", "Darius", "Mei", "Amara", "Luca", "Fatima")
        self.last_names = ("Bennett", "Patel", "Nguyen", "Rivera", "Okafor", "Kowalski", "Haddad", "Sato", "Mensah", "Dubois", "Ibrahim", "Silva")
        self.domains = ("example.net", "mail.test", "customer.example", "sample.org")
        self.streets = ("Market Street", "Cedar Avenue", "Willow Road", "Harbor Drive", "Juniper Lane", "Magnolia Boulevard", "Lakeview Terrace")
        self.cities = ("Charlotte", "Portland", "Madison", "Sacramento", "Toronto", "Vancouver", "Dublin", "Manchester", "Berlin", "Lisbon")
        self.states = ("North Carolina", "California", "New York", "Texas", "Ontario", "British Columbia", "Leinster", "Bavaria")
        self.countries = ("United States", "Canada", "Ireland", "Germany", "Portugal", "Australia", "New Zealand", "Japan")

    def _letters(self, length: int) -> str:
        return "".join(self.rng.choice(string.ascii_uppercase) for _ in range(length))

    def _digits(self, length: int) -> str:
        return "".join(self.rng.choice(string.digits) for _ in range(length))

    def _timestamp(self, record_index: int) -> str:
        return (BASE_TIME + timedelta(seconds=record_index * 37)).isoformat().replace("+00:00", "Z")

    def _request_id(self) -> str:
        return str(uuid.UUID(int=self.rng.getrandbits(128), version=4))

    @staticmethod
    def _luhn_valid(number: str) -> bool:
        total = 0
        parity = len(number) % 2
        for index, char in enumerate(number):
            digit = int(char)
            if index % 2 == parity:
                digit *= 2
                if digit > 9:
                    digit -= 9
            total += digit
        return total % 10 == 0

    def _credit_card(self) -> str:
        first_fifteen = self.rng.choice(("4", "5")) + self._digits(14)
        for check_digit in string.digits:
            candidate = first_fifteen + check_digit
            if self._luhn_valid(candidate):
                return candidate
        raise AssertionError("Unable to construct Luhn-valid card")

    def _routing_number(self) -> str:
        digits = [self.rng.randint(0, 9) for _ in range(8)]
        checksum = 3 * (digits[0] + digits[3] + digits[6])
        checksum += 7 * (digits[1] + digits[4] + digits[7])
        checksum += digits[2] + digits[5]
        digits.append((-checksum) % 10)
        return "".join(str(digit) for digit in digits)

    @staticmethod
    def _iban_check_digits(country: str, bban: str) -> str:
        rearranged = bban + country + "00"
        numeric = "".join(str(ord(char) - 55) if char.isalpha() else char for char in rearranged)
        return f"{98 - (int(numeric) % 97):02d}"

    def _iban(self) -> str:
        country = self.rng.choice(("GB", "DE"))
        bban = self._letters(4) + self._digits(14) if country == "GB" else self._digits(18)
        return country + self._iban_check_digits(country, bban) + bban

    def _value_for(self, label: str, record_index: int) -> Tuple[str, str]:
        if label == "PERSON_NAME":
            return f"{self.rng.choice(self.first_names)} {self.rng.choice(self.last_names)}", label
        if label == "DOB":
            date = datetime(1940, 1, 1) + timedelta(days=self.rng.randint(0, 24000))
            return date.strftime(self.rng.choice(("%Y-%m-%d", "%m/%d/%Y", "%d %b %Y"))), label
        if label == "AGE":
            return str(self.rng.randint(18, 94)), label
        if label == "SSN":
            return f"{self.rng.randint(101,899):03d}-{self.rng.randint(1,99):02d}-{self.rng.randint(1,9999):04d}", label
        if label == "ITIN":
            return f"9{self.rng.randint(0,99):02d}-{self.rng.randint(70,88):02d}-{self.rng.randint(1,9999):04d}", label
        if label == "TAX_ID":
            return f"{self.rng.randint(10,98):02d}-{self.rng.randint(1,9_999_999):07d}", label
        if label == "EMAIL":
            local = f"{self.rng.choice(self.first_names).lower()}.{self.rng.choice(self.last_names).lower()}+{record_index}"
            return f"{local}@{self.rng.choice(self.domains)}", label
        if label == "PHONE":
            area, exchange, line = self.rng.randint(201, 989), self.rng.randint(201, 989), self.rng.randint(0, 9999)
            return self.rng.choice((f"+1-{area}-{exchange}-{line:04d}", f"({area}) {exchange}-{line:04d}", f"{area}.{exchange}.{line:04d}")), label
        if label == "STREET_ADDRESS":
            unit = self.rng.choice(("", f", Apt {self.rng.randint(1,999)}", f", Suite {self.rng.randint(100,999)}"))
            return f"{self.rng.randint(10,9999)} {self.rng.choice(self.streets)}{unit}", label
        if label == "CITY":
            return self.rng.choice(self.cities), label
        if label == "STATE_PROVINCE":
            return self.rng.choice(self.states), label
        if label == "POSTAL_CODE":
            us_code = f"{self.rng.randint(10000,99999)}"
            ca_code = f"{self._letters(1)}{self.rng.randint(0,9)}{self._letters(1)} {self.rng.randint(0,9)}{self._letters(1)}{self.rng.randint(0,9)}"
            return self.rng.choice((us_code, ca_code)), label
        if label == "COUNTRY":
            return self.rng.choice(self.countries), label
        if label == "CREDIT_CARD_NUMBER":
            return self._credit_card(), label
        if label == "BANK_ACCOUNT_NUMBER":
            return self._digits(self.rng.randint(8, 14)), label
        if label == "ROUTING_NUMBER":
            return self._routing_number(), label
        if label == "IBAN":
            return self._iban(), label
        if label == "SWIFT_BIC":
            value = self._letters(4) + self.rng.choice(("US", "CA", "GB", "DE")) + self._letters(2)
            return value + self.rng.choice(("", self._letters(3))), label
        if label == "BUSINESS_ID":
            subtype = self.rng.choice(BUSINESS_ID_SUBTYPES)
            prefix = {"CUSTOMER_ID": "CUST", "ACCOUNT_ID": "ACCT", "LOAN_NUMBER": "LOAN", "APPLICATION_ID": "APP", "TRANSACTION_ID": "TXN", "USER_ID": "USR"}[subtype]
            return f"{prefix}-{self._letters(2)}-{self._digits(8)}", subtype
        if label == "PASSPORT_NUMBER":
            return self._letters(2) + self._digits(7), label
        if label == "DRIVER_LICENSE":
            return self.rng.choice(("NC", "CA", "TX", "NY")) + self._letters(1) + self._digits(8), label
        if label == "IP_ADDRESS":
            if self.rng.random() < 0.8:
                return str(ipaddress.IPv4Address(self.rng.randint(0x01000000, 0xDFFFFFFE))), label
            return str(ipaddress.IPv6Address(self.rng.getrandbits(128))), label
        if label == "DEVICE_ID":
            value = self.rng.choice((f"device-{self._letters(4)}-{self._digits(10)}", str(uuid.UUID(int=self.rng.getrandbits(128), version=4))))
            return value, label
        if label == "AUTH_TOKEN":
            alphabet = string.ascii_letters + string.digits + "-_"
            return "tok_" + "".join(self.rng.choice(alphabet) for _ in range(48)), label
        if label == "API_KEY":
            alphabet = string.ascii_letters + string.digits
            return "sk_test_" + "".join(self.rng.choice(alphabet) for _ in range(40)), label
        raise ValueError(f"Unsupported ML-v1 label: {label}")

    def _noise_value(self, kind: str) -> str:
        if kind == "cardlike_transaction":
            return self._digits(16)
        if kind == "ssnlike_reference":
            return f"{self.rng.randint(100,999)}-{self.rng.randint(10,99)}-{self.rng.randint(1000,9999)}"
        if kind == "ipv4like_build":
            return f"{self.rng.randint(1,9)}.{self.rng.randint(0,30)}.{self.rng.randint(0,99)}.{self.rng.randint(256,999)}"
        if kind == "accountlike_numeric_id":
            return self._digits(self.rng.randint(10, 14))
        if kind == "email_like_route":
            return f"orders@cluster-{self.rng.randint(1,99)}.internal"
        if kind == "uuid":
            return self._request_id()
        if kind == "http_status":
            return str(self.rng.choice((400, 401, 403, 404, 409, 422, 500, 502, 503)))
        if kind == "version":
            return f"v{self.rng.randint(1,20)}.{self.rng.randint(0,50)}.{self.rng.randint(0,999)}"
        raise ValueError(f"Unknown hard-negative noise kind: {kind}")

    def _positive_catalog(self) -> List[TemplateSpec]:
        catalog: List[TemplateSpec] = []
        for label in ML_V1_ENTITIES:
            key, description, free_a, free_b = ENTITY_PROFILES[label]
            secondary = "PHONE" if label == "EMAIL" else "EMAIL"
            definitions = (
                ("json", "json", "positive", f'{{"timestamp":"[[TIMESTAMP]]","event":"secure_audit","{key}":"<<TARGET>>","request_id":"[[REQUEST_ID]]"}}'),
                ("key_value", "key_value", "positive", f'INFO ts=[[TIMESTAMP]] event=secure_audit {key}="<<TARGET>>" request_id=[[REQUEST_ID]]'),
                ("xml", "xml", "positive", f'<secure_event><timestamp>[[TIMESTAMP]]</timestamp><{key}><<TARGET>></{key}><request_id>[[REQUEST_ID]]</request_id></secure_event>'),
                ("secure", "text", "positive", f'SECURE audit captured the {description}: <<TARGET>>; correlation=[[REQUEST_ID]]'),
                ("info", "text", "positive", f'INFO Privacy workflow recorded the {description} as <<TARGET>> for event [[EVENT_ID]].'),
                ("warn", "text", "context_challenging", f'WARN During manual review, an operator repeated <<TARGET>> while discussing the {description}; trace [[REQUEST_ID]].'),
                ("error", "text", "context_challenging", f'ERROR Validation rejected <<TARGET>> while checking the {description}; correlation [[REQUEST_ID]].'),
                ("free_a", "text", "context_challenging", free_a + " Case [[EVENT_ID]]."),
                ("free_b", "text", "context_challenging", free_b + " Trace [[REQUEST_ID]]."),
                ("multi", "text", "context_challenging", free_a + " Follow-up destination <<SECONDARY>> was verified under [[EVENT_ID]]."),
            )
            for suffix, text_format, scenario_kind, template in definitions:
                catalog.append(
                    TemplateSpec(
                        family=f"{label.lower()}_{suffix}_v1",
                        text_format=text_format,
                        scenario_kind=scenario_kind,
                        rationale=f"Annotate {label} because the context identifies the {description}; operational identifiers remain unannotated.",
                        template=template,
                        target_entity=label,
                        secondary_entity=secondary if suffix == "multi" else None,
                    )
                )
        return catalog

    @staticmethod
    def _hard_negative_catalog() -> List[TemplateSpec]:
        return [
            TemplateSpec("transaction_id_card_shape_kv_v1", "key_value", "hard_negative", "The 16-digit value is a TRANSACTION_ID normalized to BUSINESS_ID, not a payment card.", "INFO transaction_id=<<NOISE>> state=settled request_id=[[REQUEST_ID]]", noise_kind="cardlike_transaction", noise_entity="BUSINESS_ID", noise_source_label="TRANSACTION_ID"),
            TemplateSpec("order_reference_ssn_shape_warn_v1", "text", "hard_negative", "The SSN-shaped order reference is a business identifier, not an SSN.", "WARN Order reference <<NOISE>> was replayed after a warehouse timeout.", noise_kind="ssnlike_reference", noise_entity="BUSINESS_ID", noise_source_label="ORDER_ID"),
            TemplateSpec("build_identifier_ip_shape_v1", "text", "hard_negative", "The dotted value is a build identifier, not a network address.", "Deployment selected build <<NOISE>> for canary validation.", noise_kind="ipv4like_build"),
            TemplateSpec("application_id_account_shape_json_v1", "json", "hard_negative", "The long numeric application ID is BUSINESS_ID, not a bank account.", '{"event":"application_retry","application_id":"<<NOISE>>","status":"queued","request_id":"[[REQUEST_ID]]"}', noise_kind="accountlike_numeric_id", noise_entity="BUSINESS_ID", noise_source_label="APPLICATION_ID"),
            TemplateSpec("email_like_service_route_v1", "text", "hard_negative", "The email-shaped string names an internal route rather than a person's mailbox.", "Messages were published to internal route <<NOISE>> during failover.", noise_kind="email_like_route"),
            TemplateSpec("uuid_adjacent_email_json_v1", "json", "hard_negative", "Annotate the personal email only; the adjacent UUID is a request ID.", '{"request_id":"<<NOISE>>","contact":"<<TARGET>>","result":"failed"}', target_entity="EMAIL", noise_kind="uuid"),
            TemplateSpec("trace_adjacent_auth_token_kv_v1", "key_value", "hard_negative", "Annotate the authorization token only; the adjacent UUID is operational trace data.", "ERROR trace_id=<<NOISE>> credential='<<TARGET>>' upstream=denied", target_entity="AUTH_TOKEN", noise_kind="uuid"),
            TemplateSpec("transaction_id_adjacent_card_v1", "text", "hard_negative", "Annotate the card as CREDIT_CARD_NUMBER and normalize the equally long TRANSACTION_ID to BUSINESS_ID.", "Payment attempt <<NOISE>> used card <<TARGET>> before issuer decline.", target_entity="CREDIT_CARD_NUMBER", noise_kind="cardlike_transaction", noise_entity="BUSINESS_ID", noise_source_label="TRANSACTION_ID"),
            TemplateSpec("structured_json_numeric_noise_v1", "json", "hard_negative", "HTTP status, retry count, and request UUID are operational fields.", '{"status":<<NOISE>>,"retry_count":3,"request_id":"[[REQUEST_ID]]","duration_ms":418}', noise_kind="http_status"),
            TemplateSpec("key_value_accountlike_batch_v1", "key_value", "hard_negative", "The long number is explicitly a batch identifier, not a financial account.", "WARN batch_id=<<NOISE>> item_count=512 shard=7 state=retry", noise_kind="accountlike_numeric_id"),
            TemplateSpec("free_text_version_reference_v1", "text", "hard_negative", "The dotted token is a software version, not an IP address.", "The rollback restored release <<NOISE>> after the health check failed.", noise_kind="ipv4like_build"),
            TemplateSpec("error_trace_adjacent_phone_ip_v1", "text", "hard_negative", "Annotate the caller phone and client IP; the UUID is a trace identifier.", "ERROR Trace <<NOISE>> recorded callback <<TARGET>> from <<SECONDARY>> during recovery.", target_entity="PHONE", secondary_entity="IP_ADDRESS", noise_kind="uuid"),
        ]

    @staticmethod
    def _ordinary_negative_catalog() -> List[TemplateSpec]:
        return [
            TemplateSpec("health_metrics_negative_v1", "key_value", "negative", "Service health metrics contain no ML-v1 entity.", "INFO http_status=<<NOISE>> memory_pct=62 queue_depth=14", noise_kind="http_status"),
            TemplateSpec("cache_rotation_negative_v1", "text", "negative", "Cache version and item counts are ordinary operational values.", "Cache namespace <<NOISE>> rotated after 240 items expired.", noise_kind="version"),
            TemplateSpec("deployment_status_negative_v1", "json", "negative", "Deployment status and request ID are operational data.", '{"event":"deployment","version":"<<NOISE>>","status":"healthy","request_id":"[[REQUEST_ID]]"}', noise_kind="version"),
            TemplateSpec("queue_status_negative_v1", "text", "negative", "Queue depth, partition, and latency contain no ML-v1 entity.", "Queue partition 12 contains 408 messages with p99 latency 73ms."),
        ]

    def template_catalog(self) -> List[TemplateSpec]:
        return self._positive_catalog() + self._hard_negative_catalog() + self._ordinary_negative_catalog()

    def _render(self, spec: TemplateSpec, record_index: int) -> Dict[str, object]:
        replacements: Dict[str, Tuple[str, Optional[str], str]] = {}
        if spec.target_entity:
            value, source_label = self._value_for(spec.target_entity, record_index)
            replacements["TARGET"] = (value, spec.target_entity, source_label)
        if spec.secondary_entity:
            value, source_label = self._value_for(spec.secondary_entity, record_index)
            replacements["SECONDARY"] = (value, spec.secondary_entity, source_label)
        if spec.noise_kind:
            replacements["NOISE"] = (
                self._noise_value(spec.noise_kind),
                spec.noise_entity,
                spec.noise_source_label or spec.noise_kind,
            )

        template = spec.template.replace("[[TIMESTAMP]]", self._timestamp(record_index))
        template = template.replace("[[EVENT_ID]]", f"evt-{record_index:08d}")
        template = template.replace("[[REQUEST_ID]]", self._request_id())
        parts: List[str] = []
        entities: List[List[object]] = []
        provenance: List[Dict[str, object]] = []
        cursor = 0
        text_length = 0
        for match in PLACEHOLDER_RE.finditer(template):
            literal = template[cursor:match.start()]
            parts.append(literal)
            text_length += len(literal)
            placeholder = match.group(1)
            if placeholder not in replacements:
                raise ValueError(f"Template {spec.family} has unresolved placeholder {placeholder}")
            value, label, source_label = replacements[placeholder]
            start = text_length
            parts.append(value)
            text_length += len(value)
            if label:
                entities.append([start, text_length, label])
                provenance.append({
                    "start": start,
                    "end": text_length,
                    "label": label,
                    "source_label": source_label,
                    "action": "kept" if source_label == label else "normalized",
                    "training_exclusion": False,
                })
            cursor = match.end()
        parts.append(template[cursor:])
        text = "".join(parts)
        # A deterministic operational sequence keeps repeated value pools from
        # creating duplicate texts. It is deliberately unannotated and carries
        # explicit non-sensitive log-sequence context.
        if spec.text_format == "json" and text.endswith("}"):
            text = text[:-1] + f',"log_sequence":{record_index}' + "}"
        elif spec.text_format == "xml" and "</" in text:
            closing = text.rfind("</")
            text = text[:closing] + f"<log_sequence>{record_index}</log_sequence>" + text[closing:]
        else:
            text += f" log_sequence={record_index}"
        entities.sort(key=lambda entity: (int(entity[0]), int(entity[1]), str(entity[2])))

        return {
            "text": text,
            "entities": entities,
            "meta": {
                "source": "securelogx_custom_logs",
                "source_record_id": f"synthetic-{record_index:08d}",
                "format": spec.text_format,
                "review_status": "synthetic_generated",
                "license_reviewed": True,
                "template_family": spec.family,
                "scenario_kind": spec.scenario_kind,
                "annotation_rationale": spec.rationale,
                "primary_entity": spec.target_entity or spec.noise_entity or "O",
                "generator_version": "securelogx-ml-v1",
                "entity_provenance": provenance,
            },
        }

    def generate_labeled_dataset(self, count: int) -> List[Dict[str, object]]:
        if count <= 0:
            raise ValueError("count must be greater than zero")
        catalog = self.template_catalog()
        records = [self._render(catalog[index % len(catalog)], index) for index in range(count)]
        self.rng.shuffle(records)
        return records

    def generate_unlabeled_dataset(self, count: int, start_index: int = 1_000_000) -> List[Dict[str, object]]:
        if count < 0:
            raise ValueError("count cannot be negative")
        catalog = self._hard_negative_catalog() + self._ordinary_negative_catalog()
        records: List[Dict[str, object]] = []
        for offset in range(count):
            rendered = self._render(catalog[offset % len(catalog)], start_index + offset)
            meta = dict(rendered["meta"])  # type: ignore[arg-type]
            meta["review_status"] = "raw_unlabeled"
            meta["entity_provenance"] = []
            records.append({"text": rendered["text"], "entities": [], "meta": meta})
        self.rng.shuffle(records)
        return records


def write_jsonl(path: str, records: Sequence[Dict[str, object]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate hardened SecureLogX ML-v1 synthetic logs")
    parser.add_argument("--count", type=int, default=25_000, help="Number of labeled records")
    parser.add_argument("--out-labeled", required=True, help="Output labeled JSONL path")
    parser.add_argument("--out-raw", required=True, help="Output raw/unlabeled JSONL path")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed")
    args = parser.parse_args()

    try:
        generator = SecureLogXDataGenerator(seed=args.seed)
        labeled = generator.generate_labeled_dataset(args.count)
        raw = generator.generate_unlabeled_dataset(args.count // 5)
        write_jsonl(args.out_labeled, labeled)
        write_jsonl(args.out_raw, raw)
    except (OSError, ValueError, AssertionError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    families = {record["meta"]["template_family"] for record in labeled}  # type: ignore[index]
    hard_negatives = sum(record["meta"]["scenario_kind"] == "hard_negative" for record in labeled)  # type: ignore[index]
    print(f"Labeled data: {len(labeled)} records across {len(families)} template families")
    print(f"Hard-negative records: {hard_negatives}")
    print(f"Raw data: {len(raw)} records (excluded from training merge)")
    print(f"Labeled output: {args.out_labeled}")
    print(f"Raw output: {args.out_raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
