#!/usr/bin/env python3
"""Generate ML-v1.3 real-structure / semi-synthetic SecureLogX records.

Skeletons follow public log FORMAT specifications.  No third-party log corpus
is downloaded or stored.  Injected entity values are synthetic.  The generator
never opens parent JSONL files or the sealed challenge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SOURCE = "securelogx_real_structure_v1_3"
GENERATOR_VERSION = "securelogx-ml-v1.3-real-structure-v1"
PARTITIONS = ("train_addition", "dev_challenge")
PAIR_COUNTS = {"train_addition": 12, "dev_challenge": 10}
EXPECTED_FAMILY_COUNTS = {"train_addition": 96, "dev_challenge": 24}
EXPECTED_RECORD_COUNTS = {
    partition: EXPECTED_FAMILY_COUNTS[partition] * PAIR_COUNTS[partition] * 2
    for partition in PARTITIONS
}
BUSINESS_ID_SUBTYPES = (
    "CUSTOMER_ID",
    "REQUEST_ID",
    "TRANSACTION_ID",
    "APPLICATION_ID",
    "ORDER_ID",
    "ACCOUNT_ID",
    "CASE_ID",
    "TICKET_ID",
    "REFERENCE_ID",
    "WORKFLOW_ID",
)
FAILURE_MODES = (
    "accountlike_o_vs_business_id",
    "auth_token_vs_business_id",
    "ssn_vs_itin",
    "ip_vs_technical_reference",
    "high_risk_identifier",
)
FOCUS = "{{FOCUS}}"
COMPANION = "{{COMPANION}}"
_SLUG_RE = re.compile(r"[^a-z0-9]+")

PUBLIC_SOURCES = {
    "spring_boot_application": {
        "name": "Spring Boot default logging pattern",
        "url": "https://docs.spring.io/spring-boot/docs/current/reference/html/features.html#features.logging",
        "license": "Apache-2.0 (documentation/format)",
        "license_reviewed": True,
        "raw_source_text_stored": False,
        "structure_retained": True,
        "classification": "public_format_specification",
    },
    "log4j_pattern": {
        "name": "Apache Log4j 2 PatternLayout",
        "url": "https://logging.apache.org/log4j/2.x/manual/layouts.html",
        "license": "Apache-2.0 (documentation/format)",
        "license_reviewed": True,
        "raw_source_text_stored": False,
        "structure_retained": True,
        "classification": "public_format_specification",
    },
    "json_application": {
        "name": "Elastic Common Schema JSON log conventions",
        "url": "https://www.elastic.co/guide/en/ecs/current/ecs-reference.html",
        "license": "Apache-2.0 (schema documentation)",
        "license_reviewed": True,
        "raw_source_text_stored": False,
        "structure_retained": True,
        "classification": "public_format_specification",
    },
    "nested_json": {
        "name": "JSON application logs with nested objects",
        "url": "https://www.elastic.co/guide/en/ecs/current/ecs-reference.html",
        "license": "Apache-2.0 (schema documentation)",
        "license_reviewed": True,
        "raw_source_text_stored": False,
        "structure_retained": True,
        "classification": "public_format_specification",
    },
    "key_value_audit": {
        "name": "Original key=value audit-line structure",
        "url": None,
        "license": "original SecureLogX research structure",
        "license_reviewed": True,
        "raw_source_text_stored": False,
        "structure_retained": True,
        "classification": "original_research_structure",
    },
    "rest_access": {
        "name": "Apache Combined Log Format",
        "url": "https://httpd.apache.org/docs/current/logs.html",
        "license": "Apache License 2.0 documentation; format is a public logging convention",
        "license_reviewed": True,
        "raw_source_text_stored": False,
        "structure_retained": True,
        "classification": "public_format_specification",
    },
    "authentication_security": {
        "name": "HTTP Authorization/Bearer header logging convention",
        "url": "https://datatracker.ietf.org/doc/html/rfc6750",
        "license": "IETF RFC 6750 (BSD-style)",
        "license_reviewed": True,
        "raw_source_text_stored": False,
        "structure_retained": True,
        "classification": "public_format_specification",
    },
    "kafka_client": {
        "name": "Apache Kafka client logging conventions",
        "url": "https://kafka.apache.org/documentation/",
        "license": "Apache-2.0",
        "license_reviewed": True,
        "raw_source_text_stored": False,
        "structure_retained": True,
        "classification": "public_format_specification",
    },
    "database_error": {
        "name": "JDBC/SQLException message structure",
        "url": "https://docs.oracle.com/javase/8/docs/api/java/sql/SQLException.html",
        "license": "original structure inspired by public JDBC exception API docs",
        "license_reviewed": True,
        "raw_source_text_stored": False,
        "structure_retained": True,
        "classification": "public_format_specification",
    },
    "stack_trace_adjacent": {
        "name": "Java stack-trace-adjacent log line",
        "url": None,
        "license": "original SecureLogX research structure",
        "license_reviewed": True,
        "raw_source_text_stored": False,
        "structure_retained": True,
        "classification": "original_research_structure",
    },
    "warn_error": {
        "name": "WARN/ERROR operational log line",
        "url": None,
        "license": "original SecureLogX research structure",
        "license_reviewed": True,
        "raw_source_text_stored": False,
        "structure_retained": True,
        "classification": "original_research_structure",
    },
    "mixed_structured": {
        "name": "Mixed JSON+text operational log",
        "url": None,
        "license": "original SecureLogX research structure",
        "license_reviewed": True,
        "raw_source_text_stored": False,
        "structure_retained": True,
        "classification": "original_research_structure",
    },
    "syslog_rfc5424": {
        "name": "RFC 5424 syslog",
        "url": "https://datatracker.ietf.org/doc/html/rfc5424",
        "license": "IETF RFC 5424",
        "license_reviewed": True,
        "raw_source_text_stored": False,
        "structure_retained": True,
        "classification": "public_format_specification",
    },
    "nginx_access": {
        "name": "nginx combined access log format",
        "url": "https://nginx.org/en/docs/http/ngx_http_log_module.html",
        "license": "BSD-2-Clause (nginx documentation/format)",
        "license_reviewed": True,
        "raw_source_text_stored": False,
        "structure_retained": True,
        "classification": "public_format_specification",
    },
    "kubernetes_container": {
        "name": "Kubernetes container log prefix convention",
        "url": "https://kubernetes.io/docs/concepts/cluster-administration/logging/",
        "license": "Apache-2.0 (Kubernetes documentation)",
        "license_reviewed": True,
        "raw_source_text_stored": False,
        "structure_retained": True,
        "classification": "public_format_specification",
    },
    "free_text_operational": {
        "name": "Free-text operational message",
        "url": None,
        "license": "original SecureLogX research structure",
        "license_reviewed": True,
        "raw_source_text_stored": False,
        "structure_retained": True,
        "classification": "original_research_structure",
    },
}

STRUCTURE_CATEGORIES = tuple(PUBLIC_SOURCES)


@dataclass(frozen=True)
class RoleSpec:
    context_role: str
    expected_label: str
    template: str
    companion_label: str
    companion_subtype: str | None = None
    focus_kind: str = "business_id"
    companion_kind: str = "business_id"


@dataclass(frozen=True)
class FamilySpec:
    template_family: str
    intended_split: str
    structure_category: str
    failure_mode: str
    business_id_subtype: str
    pair_count: int
    roles: tuple[RoleSpec, RoleSpec]


def _slug(value: str) -> str:
    return _SLUG_RE.sub("-", value.casefold()).strip("-")


def _digits(rng: random.Random, count: int) -> str:
    return "".join(str(rng.randint(0, 9)) for _ in range(count))


def _alnum(rng: random.Random, count: int) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(rng.choice(alphabet) for _ in range(count))


def _luhn_card(rng: random.Random) -> str:
    body = [rng.choice((4, 5))] + [rng.randint(0, 9) for _ in range(14)]
    total = 0
    for index, digit in enumerate(reversed(body)):
        value = digit * (2 if index % 2 == 0 else 1)
        total += value - 9 if value > 9 else value
    body.append((10 - (total % 10)) % 10)
    return "".join(str(digit) for digit in body)


def _ssn(rng: random.Random) -> str:
    area = rng.choice((123, 234, 321, 456, 512, 587, 612, 701, 734, 812))
    return f"{area:03d}-{rng.randint(10, 92):02d}-{rng.randint(1000, 9999):04d}"


def _itin(rng: random.Random) -> str:
    group = rng.choice((70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88))
    return f"9{rng.randint(10, 99):02d}-{group:02d}-{rng.randint(1000, 9999):04d}"


def _ipv4(rng: random.Random) -> str:
    return f"198.51.{rng.randint(0, 255)}.{rng.randint(1, 254)}"


def _version_quad(rng: random.Random) -> str:
    return f"{rng.randint(1, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 40)}.{rng.randint(0, 12)}"


def _business_id(subtype: str, rng: random.Random) -> str:
    prefixes = {
        "CUSTOMER_ID": "CUST",
        "REQUEST_ID": "REQ",
        "TRANSACTION_ID": "TXN",
        "APPLICATION_ID": "APP",
        "ORDER_ID": "ORD",
        "ACCOUNT_ID": "ACCT",
        "CASE_ID": "CASE",
        "TICKET_ID": "TCK",
        "REFERENCE_ID": "REF",
        "WORKFLOW_ID": "WF",
    }
    return f"{prefixes[subtype]}-{_alnum(rng, 4).upper()}{_digits(rng, 6)}"


def _service_account(rng: random.Random) -> str:
    stem = rng.choice(
        (
            "svc-payments",
            "svc-ledger",
            "deploy-account",
            "env-alias",
            "cache-redis",
            "db-migrate",
            "infra-ci",
            "system-ops",
            "batch-runner",
            "k8s-node",
        )
    )
    env = rng.choice(("prod", "staging", "qa", "dev", "ops"))
    return f"{stem}-{env}-{_digits(rng, 3)}"


def _auth_token(rng: random.Random) -> str:
    return "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9." + _alnum(rng, 24)


def _api_key(rng: random.Random) -> str:
    return "ak_live_" + _alnum(rng, 28)


def _value(kind: str, subtype: str, rng: random.Random) -> str:
    if kind == "business_id":
        return _business_id(subtype, rng)
    if kind == "service_account":
        return _service_account(rng)
    if kind == "auth_token":
        return _auth_token(rng)
    if kind == "api_key":
        return _api_key(rng)
    if kind == "ssn":
        return _ssn(rng)
    if kind == "itin":
        return _itin(rng)
    if kind == "ip":
        return _ipv4(rng)
    if kind == "version_quad":
        return _version_quad(rng)
    if kind == "card":
        grouped = _luhn_card(rng)
        return f"{grouped[:4]}-{grouped[4:8]}-{grouped[8:12]}-{grouped[12:]}"
    raise ValueError(kind)


def _wrap(category: str, unique: str, message: str) -> str:
    if category == "spring_boot_application":
        return (
            f"2026-05-11 07:22:18.441  WARN 18432 --- [http-nio-8080-exec-3] "
            f"c.s.{unique} : {message}"
        )
    if category == "log4j_pattern":
        return f"2026-05-11 07:22:18,441 [worker-7] ERROR com.securelogx.{unique} - {message}"
    if category == "json_application":
        return (
            '{"@timestamp":"2026-05-11T07:22:18.441Z","log.level":"WARN","service.name":"'
            + unique
            + '","message":"'
            + message
            + '"}'
        )
    if category == "nested_json":
        return (
            '{"event":{"kind":"log","dataset":"'
            + unique
            + '"},"securelogx":{"detail":"'
            + message
            + '"}}'
        )
    if category == "key_value_audit":
        return f"audit.ts=2026-05-11T07:22:18Z component={unique} {message} result=recorded"
    if category == "rest_access":
        return (
            f'10.0.0.8 - svc-{unique} [11/May/2026:07:22:18 -0400] '
            f'"GET /v1/{unique} HTTP/1.1" 401 218 "-" "{message}"'
        )
    if category == "authentication_security":
        return f"security.filter={unique} decision=deny {message}"
    if category == "kafka_client":
        return (
            f"[2026-05-11 07:22:18,441] WARN org.apache.kafka.clients.{unique}: {message}"
        )
    if category == "database_error":
        return (
            f"org.postgresql.util.PSQLException: ERROR: {message} "
            f"[SQLState=28000, component={unique}]"
        )
    if category == "stack_trace_adjacent":
        return (
            f"java.lang.IllegalStateException: {message}\n"
            f"\tat com.securelogx.{unique}.handle(Handler.java:88)"
        )
    if category == "warn_error":
        return f"ERROR {unique} operation-failed {message}"
    if category == "mixed_structured":
        return f"{unique} context={{ {message} }} flattened=true"
    if category == "syslog_rfc5424":
        return (
            f"<134>1 2026-05-11T07:22:18.441Z payments-api-1 {unique} 44192 "
            f"ID47 - {message}"
        )
    if category == "nginx_access":
        return (
            f'203.0.113.8 - - [11/May/2026:07:22:18 -0400] "POST /{unique} HTTP/1.1" '
            f'403 512 "{message}" "SecureLogX-Test/1.3"'
        )
    if category == "kubernetes_container":
        return f"2026-05-11T07:22:18.441444444Z stderr F [{unique}] {message}"
    if category == "free_text_operational":
        return f"ops note from {unique}: {message}"
    raise ValueError(category)


def _messages(failure_mode: str, subtype: str) -> tuple[tuple[str, RoleSpec], tuple[str, RoleSpec]]:
    field = subtype.replace("_", "").lower()
    if failure_mode == "accountlike_o_vs_business_id":
        return (
            (
                f"technicalAccount={FOCUS} customer{subtype.title().replace('_','')}={COMPANION} reason=non-human-principal",
                RoleSpec(
                    "technical_account",
                    "O",
                    "",
                    "BUSINESS_ID",
                    subtype,
                    "service_account",
                    "business_id",
                ),
            ),
            (
                f"customerAccount={FOCUS} requestId={COMPANION} reason=human-customer-principal",
                RoleSpec(
                    "customer_account",
                    "BUSINESS_ID",
                    "",
                    "BUSINESS_ID",
                    "REQUEST_ID",
                    "business_id",
                    "business_id",
                ),
            ),
        )
    if failure_mode == "auth_token_vs_business_id":
        return (
            (
                f"Authorization={FOCUS} {field}={COMPANION} authScheme=Bearer",
                RoleSpec(
                    "authorization_bearer",
                    "AUTH_TOKEN",
                    "",
                    "BUSINESS_ID",
                    subtype,
                    "auth_token",
                    "business_id",
                ),
            ),
            (
                f"opaqueBusinessKey={FOCUS} Authorization={COMPANION} authScheme=Bearer",
                RoleSpec(
                    "opaque_business_key",
                    "BUSINESS_ID",
                    "",
                    "AUTH_TOKEN",
                    None,
                    "business_id",
                    "auth_token",
                ),
            ),
        )
    if failure_mode == "ssn_vs_itin":
        return (
            (
                f"payrollSsn={FOCUS} employee{subtype.title().replace('_','')}={COMPANION} taxYear=2026",
                RoleSpec("payroll_ssn", "SSN", "", "BUSINESS_ID", subtype, "ssn", "business_id"),
            ),
            (
                f"itinNumber={FOCUS} filer{subtype.title().replace('_','')}={COMPANION} taxYear=2026",
                RoleSpec("tax_itin", "ITIN", "", "BUSINESS_ID", subtype, "itin", "business_id"),
            ),
        )
    if failure_mode == "ip_vs_technical_reference":
        return (
            (
                f"clientIp={FOCUS} {field}={COMPANION} source=load-balancer",
                RoleSpec("client_ip", "IP_ADDRESS", "", "BUSINESS_ID", subtype, "ip", "business_id"),
            ),
            (
                f"libraryVersion={FOCUS} {field}={COMPANION} source=dependency-coordinate",
                RoleSpec(
                    "technical_reference",
                    "O",
                    "",
                    "BUSINESS_ID",
                    subtype,
                    "version_quad",
                    "business_id",
                ),
            ),
        )
    if failure_mode == "high_risk_identifier":
        return (
            (
                f"apiKey={FOCUS} {field}={COMPANION} sink=payments-gateway",
                RoleSpec(
                    "gateway_api_key",
                    "API_KEY",
                    "",
                    "BUSINESS_ID",
                    subtype,
                    "api_key",
                    "business_id",
                ),
            ),
            (
                f"cardPan={FOCUS} {field}={COMPANION} sink=payments-gateway",
                RoleSpec(
                    "gateway_card",
                    "CREDIT_CARD_NUMBER",
                    "",
                    "BUSINESS_ID",
                    subtype,
                    "card",
                    "business_id",
                ),
            ),
        )
    raise ValueError(failure_mode)


def catalog() -> tuple[FamilySpec, ...]:
    families: list[FamilySpec] = []
    train_plan = (
        ("accountlike_o_vs_business_id", 24),
        ("auth_token_vs_business_id", 24),
        ("ssn_vs_itin", 16),
        ("ip_vs_technical_reference", 16),
        ("high_risk_identifier", 16),
    )
    dev_plan = (
        ("accountlike_o_vs_business_id", 6),
        ("auth_token_vs_business_id", 6),
        ("ssn_vs_itin", 4),
        ("ip_vs_technical_reference", 4),
        ("high_risk_identifier", 4),
    )
    for partition, plan in (
        ("train_addition", train_plan),
        ("dev_challenge", dev_plan),
    ):
        cursor = 0
        for failure_mode, count in plan:
            for index in range(count):
                category = STRUCTURE_CATEGORIES[(cursor + index) % len(STRUCTURE_CATEGORIES)]
                subtype = BUSINESS_ID_SUBTYPES[(cursor + index) % len(BUSINESS_ID_SUBTYPES)]
                unique = f"{partition[:5]}{failure_mode.split('_')[0]}{index:02d}{subtype[:4].lower()}"
                left_msg, right_msg = _messages(failure_mode, subtype)
                roles = []
                for message, role in (left_msg, right_msg):
                    template = _wrap(category, unique, message)
                    if template.count(FOCUS) != 1 or template.count(COMPANION) != 1:
                        raise AssertionError(template)
                    roles.append(
                        RoleSpec(
                            role.context_role,
                            role.expected_label,
                            template,
                            role.companion_label,
                            role.companion_subtype,
                            role.focus_kind,
                            role.companion_kind,
                        )
                    )
                slug = _slug(f"{failure_mode}_{category}_{unique}")
                families.append(
                    FamilySpec(
                        template_family=f"ml_v1_3_{partition}_{slug}_v1_3",
                        intended_split=partition,
                        structure_category=category,
                        failure_mode=failure_mode,
                        business_id_subtype=subtype,
                        pair_count=PAIR_COUNTS[partition],
                        roles=(roles[0], roles[1]),
                    )
                )
            cursor += count
    families.sort(key=lambda spec: spec.template_family)
    names = [spec.template_family for spec in families]
    if len(set(names)) != len(names):
        raise AssertionError("duplicate family names")
    templates = [role.template for spec in families for role in spec.roles]
    if len(set(templates)) != len(templates):
        raise AssertionError("duplicate templates")
    by_split = {partition: 0 for partition in PARTITIONS}
    for spec in families:
        by_split[spec.intended_split] += 1
    if by_split != dict(EXPECTED_FAMILY_COUNTS):
        raise AssertionError(by_split)
    return tuple(families)


def _span(text: str, value: str) -> tuple[int, int]:
    start = text.index(value)
    return start, start + len(value)


def _record(
    spec: FamilySpec,
    role: RoleSpec,
    group_index: int,
    focus: str,
    companion: str,
) -> dict[str, Any]:
    text = role.template.replace(FOCUS, focus).replace(COMPANION, companion)
    entities: list[list[Any]] = []
    provenance: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    focus_span = _span(text, focus)
    companion_span = _span(text, companion)
    if role.expected_label != "O":
        entities.append([focus_span[0], focus_span[1], role.expected_label])
        entry = {
            "start": focus_span[0],
            "end": focus_span[1],
            "label": role.expected_label,
        }
        if role.expected_label == "BUSINESS_ID":
            entry["business_id_subtype"] = spec.business_id_subtype
        provenance.append(entry)
    targets.append(
        {
            "start": focus_span[0],
            "end": focus_span[1],
            "expected_label": role.expected_label,
            "role": "focus",
        }
    )
    entities.append([companion_span[0], companion_span[1], role.companion_label])
    companion_entry = {
        "start": companion_span[0],
        "end": companion_span[1],
        "label": role.companion_label,
    }
    if role.companion_label == "BUSINESS_ID":
        companion_entry["business_id_subtype"] = (
            role.companion_subtype or spec.business_id_subtype
        )
    provenance.append(companion_entry)
    targets.append(
        {
            "start": companion_span[0],
            "end": companion_span[1],
            "expected_label": role.companion_label,
            "role": "companion",
        }
    )
    entities.sort(key=lambda item: (item[0], item[1], item[2]))
    public = PUBLIC_SOURCES[spec.structure_category]
    source_id = (
        f"ml-v1.3-{spec.intended_split}-{_slug(spec.template_family)}-"
        f"g{group_index:04d}-{_slug(role.context_role)}"
    )
    return {
        "text": text,
        "entities": entities,
        "meta": {
            "source": SOURCE,
            "source_record_id": source_id,
            "generator_version": GENERATOR_VERSION,
            "intended_split": spec.intended_split,
            "template_family": spec.template_family,
            "structure_family": spec.template_family,
            "structure_category": spec.structure_category,
            "failure_mode": spec.failure_mode,
            "contrast_category": spec.failure_mode,
            "context_role": role.context_role,
            "public_source_identifier": spec.structure_category,
            "public_source_name": public["name"],
            "public_source_url": public["url"],
            "license": public["license"],
            "license_reviewed": public["license_reviewed"],
            "raw_source_text_stored": False,
            "synthetic_value_injection": True,
            "review_status": "synthetic_generated",
            "format": (
                "json"
                if spec.structure_category in {"json_application", "nested_json"}
                else "key_value"
                if spec.structure_category == "key_value_audit"
                else "text"
            ),
            "entity_provenance": provenance,
            "contrast_targets": targets,
            "business_id_subtype": spec.business_id_subtype,
        },
    }


def generate_all(seed: int = 42) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    used: set[str] = set()
    by_partition: dict[str, list[dict[str, Any]]] = {partition: [] for partition in PARTITIONS}
    for spec in catalog():
        for group_index in range(spec.pair_count):
            values: dict[str, str] = {}
            for role in spec.roles:
                for kind, subtype_key in (
                    (role.focus_kind, spec.business_id_subtype),
                    (
                        role.companion_kind,
                        role.companion_subtype or spec.business_id_subtype,
                    ),
                ):
                    key = f"{kind}:{subtype_key}:{spec.template_family}:{group_index}:{role.context_role}"
                    while True:
                        candidate = _value(kind, subtype_key, rng)
                        if candidate not in used:
                            used.add(candidate)
                            values[key] = candidate
                            break
            for role in spec.roles:
                focus = values[
                    f"{role.focus_kind}:{spec.business_id_subtype}:{spec.template_family}:{group_index}:{role.context_role}"
                ]
                companion = values[
                    f"{role.companion_kind}:{role.companion_subtype or spec.business_id_subtype}:{spec.template_family}:{group_index}:{role.context_role}"
                ]
                by_partition[spec.intended_split].append(
                    _record(spec, role, group_index, focus, companion)
                )
    for partition, records in by_partition.items():
        records.sort(key=lambda record: record["meta"]["source_record_id"])
        if len(records) != EXPECTED_RECORD_COUNTS[partition]:
            raise AssertionError((partition, len(records)))
    return by_partition


def generate_partition(partition: str, seed: int = 42) -> list[dict[str, Any]]:
    return generate_all(seed)[partition]


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--partition", choices=PARTITIONS)
    args = parser.parse_args()
    generated = generate_all(args.seed)
    if args.partition:
        print(len(generated[args.partition]))
    else:
        print({name: len(rows) for name, rows in generated.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
