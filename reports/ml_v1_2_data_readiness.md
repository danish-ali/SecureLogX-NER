# ML-v1.2 Data Readiness

## Recommendation: READY FOR ML-v1.2 RETRAINING

| Gate | Result |
|---|---|
| `all_business_id_subtypes_retained` | PASS |
| `all_sensitive_counterexamples_present` | PASS |
| `canonical_ontology_unchanged` | PASS |
| `dev_record_count_within_declared_range` | PASS |
| `exact_text_leakage_zero` | PASS |
| `model_predictions_not_run` | PASS |
| `multi_entity_ratio_met` | PASS |
| `original_test_contamination_zero` | PASS |
| `parent_artifact_hash_mismatches_zero` | PASS |
| `sealed_challenge_hash_unchanged` | PASS |
| `sealed_challenge_never_opened` | PASS |
| `ssn_counterbalance_material` | PASS |
| `target_dev_family_count` | PASS |
| `target_dev_record_count` | PASS |
| `target_train_family_count` | PASS |
| `target_train_record_count` | PASS |
| `template_family_leakage_zero` | PASS |
| `train_record_count_within_declared_range` | PASS |
| `worst_category_families_present` | PASS |
| `validator.canonical_spans` | PASS |
| `validator.complete_paired_contrast_groups` | PASS |
| `validator.contrast_targets` | PASS |
| `validator.cross_split_exact_text_leakage` | PASS |
| `validator.cross_split_focus_value_hash_leakage` | PASS |
| `validator.cross_split_source_id_leakage` | PASS |
| `validator.cross_split_template_family_leakage` | PASS |
| `validator.dev_challenge_category_family_allocation` | PASS |
| `validator.dev_challenge_family_count` | PASS |
| `validator.dev_challenge_multi_entity_ratio` | PASS |
| `validator.dev_challenge_record_count` | PASS |
| `validator.dev_challenge_record_count_range` | PASS |
| `validator.dev_ip_technical_family_minimum` | PASS |
| `validator.entity_provenance_bijection` | PASS |
| `validator.family_metadata_consistency` | PASS |
| `validator.forbidden_parent_hard_negative` | PASS |
| `validator.inputs_readable` | PASS |
| `validator.intended_split` | PASS |
| `validator.o_target_non_overlap` | PASS |
| `validator.ontology_exact_25_entities_51_bio` | PASS |
| `validator.parent_exact_text_contamination` | PASS |
| `validator.parent_source_id_contamination` | PASS |
| `validator.parent_template_family_contamination` | PASS |
| `validator.parent_test_exact_text_contamination` | PASS |
| `validator.record_shape` | PASS |
| `validator.sealed_challenge_file_never_opened` | PASS |
| `validator.sealed_challenge_hash_unchanged` | PASS |
| `validator.template_catalog_catalogs_readable` | PASS |
| `validator.template_catalog_cross_split_normalized_skeleton_overlap` | PASS |
| `validator.template_catalog_forbidden_parent_template` | PASS |
| `validator.template_catalog_parent_normalized_skeleton_overlap` | PASS |
| `validator.template_catalog_v1_1_family_name_overlap` | PASS |
| `validator.template_catalog_v1_1_normalized_skeleton_overlap` | PASS |
| `validator.train_addition_category_family_allocation` | PASS |
| `validator.train_addition_family_count` | PASS |
| `validator.train_addition_multi_entity_ratio` | PASS |
| `validator.train_addition_record_count` | PASS |
| `validator.train_addition_record_count_range` | PASS |
| `validator.train_all_business_id_subtypes_retained` | PASS |
| `validator.train_customer_id_family_minimum` | PASS |
| `validator.train_ip_technical_family_minimum` | PASS |
| `validator.train_request_id_family_minimum` | PASS |
| `validator.train_sensitive_counterexample_minimum` | PASS |
| `validator.train_ssn_span_minimum` | PASS |
| `validator.unique_record_texts` | PASS |
| `validator.unique_source_record_ids` | PASS |
| `validator.v1_1_contamination_zero` | PASS |
| `validator.v1_2_metadata` | PASS |

Parent artifact mismatches before generation: **0** (129 files verified).
Parent artifact mismatches after generation: **0** (129 files verified).

Sealed challenge byte SHA-256: `6f22e8133e74e4cc7e63048b05acaf9004e9ff8c385f774298e07bd6b15ded9a` (required `6f22e8133e74e4cc7e63048b05acaf9004e9ff8c385f774298e07bd6b15ded9a`; MATCH). The sealed file was never opened: the hash was established through deterministic in-memory regeneration by the frozen v1.1 generator.

The canonical ontology remains exactly 25 entities and 51 BIO labels. ML-v1 train/dev/test, both model checkpoints and their per-epoch artifacts, the frozen v1.1 datasets, configs, and reports all match the pre-generation parent snapshot.

No model training, model evaluation, ONNX export, Java modification, sealed-challenge prediction, or original-test inference occurred in this phase.

Technical readiness does not clear the separate release constraint: inherited Gretel records still carry `license_reviewed=false`, so production or commercial release requires explicit license review.
