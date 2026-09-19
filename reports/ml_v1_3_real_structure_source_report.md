# ML-v1.3 Real-Structure Source Report

No third-party log corpus was downloaded. Structures are original skeletons
inspired by public logging format specifications. Only synthetic values are stored.

| Structure category | Source | URL | License | License reviewed | Raw text stored |
|---|---|---|---|---|---|
| `spring_boot_application` | Spring Boot default logging pattern | https://docs.spring.io/spring-boot/docs/current/reference/html/features.html#features.logging | Apache-2.0 (documentation/format) | True | False |
| `log4j_pattern` | Apache Log4j 2 PatternLayout | https://logging.apache.org/log4j/2.x/manual/layouts.html | Apache-2.0 (documentation/format) | True | False |
| `json_application` | Elastic Common Schema JSON log conventions | https://www.elastic.co/guide/en/ecs/current/ecs-reference.html | Apache-2.0 (schema documentation) | True | False |
| `nested_json` | JSON application logs with nested objects | https://www.elastic.co/guide/en/ecs/current/ecs-reference.html | Apache-2.0 (schema documentation) | True | False |
| `key_value_audit` | Original key=value audit-line structure | n/a | original SecureLogX research structure | True | False |
| `rest_access` | Apache Combined Log Format | https://httpd.apache.org/docs/current/logs.html | Apache License 2.0 documentation; format is a public logging convention | True | False |
| `authentication_security` | HTTP Authorization/Bearer header logging convention | https://datatracker.ietf.org/doc/html/rfc6750 | IETF RFC 6750 (BSD-style) | True | False |
| `kafka_client` | Apache Kafka client logging conventions | https://kafka.apache.org/documentation/ | Apache-2.0 | True | False |
| `database_error` | JDBC/SQLException message structure | https://docs.oracle.com/javase/8/docs/api/java/sql/SQLException.html | original structure inspired by public JDBC exception API docs | True | False |
| `stack_trace_adjacent` | Java stack-trace-adjacent log line | n/a | original SecureLogX research structure | True | False |
| `warn_error` | WARN/ERROR operational log line | n/a | original SecureLogX research structure | True | False |
| `mixed_structured` | Mixed JSON+text operational log | n/a | original SecureLogX research structure | True | False |
| `syslog_rfc5424` | RFC 5424 syslog | https://datatracker.ietf.org/doc/html/rfc5424 | IETF RFC 5424 | True | False |
| `nginx_access` | nginx combined access log format | https://nginx.org/en/docs/http/ngx_http_log_module.html | BSD-2-Clause (nginx documentation/format) | True | False |
| `kubernetes_container` | Kubernetes container log prefix convention | https://kubernetes.io/docs/concepts/cluster-administration/logging/ | Apache-2.0 (Kubernetes documentation) | True | False |
| `free_text_operational` | Free-text operational message | n/a | original SecureLogX research structure | True | False |

Local inventory classification:

- Gretel finance PII: external labeled prose (`license_reviewed=false`); not used as a log-structure source.
- SecureLogX ML-v1 generated logs: existing synthetic templates; not reused.
- ML-v1.1 / ML-v1.2 challenge templates: generated contrast templates; not reused.
- No suitable raw public log files were present locally.
