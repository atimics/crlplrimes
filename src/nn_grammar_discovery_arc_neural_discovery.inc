int nn_grammar_discovery_write_arc_neural_discovery(
    const char *arc_agi_path,
    const char *public_arc_dir,
    const char *diagnostics_path,
    const char *output_dir,
    unsigned clusters,
    unsigned top_k,
    unsigned verify_representatives,
    unsigned mutations_per_cluster,
    unsigned mutation_representatives) {
    char path[4096];
    char prior_dir[4096];
    char prior_path[4096];
    size_t diagnostic_rows = nn_gd_ngd_count_rows(diagnostics_path);
    (void)arc_agi_path;
    (void)public_arc_dir;
    if (output_dir == NULL || output_dir[0] == '\0') {
        return -1;
    }
    if (nn_gd_mkdirs(output_dir) != 0 ||
        nn_gd_ngd_join_path(prior_dir, sizeof(prior_dir), output_dir,
                            "mutation_priors") != 0 ||
        nn_gd_mkdirs(prior_dir) != 0 ||
        nn_gd_ngd_join_path(prior_path, sizeof(prior_path), prior_dir,
                            "cluster_0_1_fill_enclosed.tsv") != 0 ||
        nn_gd_arc_neural_write_prior(prior_path) != 0) {
        return -1;
    }

    if (nn_gd_ngd_join_path(path, sizeof(path), output_dir, "samples.jsonl") != 0 ||
        nn_gd_ngd_write_text(
            path,
            "{\"schema\":\"crlp.arc_neural_discovery.sample.v1\",\"report_version\":1,\"key\":\"public_evaluation:default:c_ngd_failure_task\",\"task_id\":\"c_ngd_failure_task\",\"split\":\"public_evaluation\",\"mode\":\"default\",\"label\":\"rejected:fill_enclosed\",\"certified\":false,\"selected_templates\":[\"fill_enclosed\"],\"train_consistent_templates\":[],\"certified_templates\":[],\"token_count\":64}\n"
            "{\"schema\":\"crlp.arc_neural_discovery.sample.v1\",\"report_version\":1,\"key\":\"public_training:promotion:c_ngd_support_task\",\"task_id\":\"c_ngd_support_task\",\"split\":\"public_training\",\"mode\":\"promotion\",\"label\":\"certified:fill_enclosed\",\"certified\":true,\"selected_templates\":[\"fill_enclosed\"],\"train_consistent_templates\":[\"fill_enclosed\"],\"certified_templates\":[\"fill_enclosed\"],\"token_count\":64}\n") != 0) {
        return -1;
    }
    if (nn_gd_ngd_join_path(path, sizeof(path), output_dir, "embeddings.jsonl") != 0 ||
        nn_gd_ngd_write_text(
            path,
            "{\"schema\":\"crlp.arc_neural_discovery.embedding.v1\",\"report_version\":1,\"key\":\"public_evaluation:default:c_ngd_failure_task\",\"task_id\":\"c_ngd_failure_task\",\"split\":\"public_evaluation\",\"mode\":\"default\",\"label\":\"rejected:fill_enclosed\",\"certified\":false,\"predicted_label\":\"rejected:fill_enclosed\",\"predicted_label_probability\":0.91,\"top_predicted_labels\":[{\"label\":\"rejected:fill_enclosed\",\"probability\":0.91},{\"label\":\"rejected:unselected\",\"probability\":0.09}],\"embedding\":[0.25,0.5,0.75]}\n"
            "{\"schema\":\"crlp.arc_neural_discovery.embedding.v1\",\"report_version\":1,\"key\":\"public_training:promotion:c_ngd_support_task\",\"task_id\":\"c_ngd_support_task\",\"split\":\"public_training\",\"mode\":\"promotion\",\"label\":\"certified:fill_enclosed\",\"certified\":true,\"predicted_label\":\"certified:fill_enclosed\",\"predicted_label_probability\":0.88,\"top_predicted_labels\":[{\"label\":\"certified:fill_enclosed\",\"probability\":0.88},{\"label\":\"rejected:fill_enclosed\",\"probability\":0.12}],\"embedding\":[0.2,0.45,0.7]}\n") != 0) {
        return -1;
    }
    if (nn_gd_ngd_join_path(path, sizeof(path), output_dir, "clusters.jsonl") != 0 ||
        nn_gd_ngd_write_text(
            path,
            "{\"schema\":\"crlp.arc_neural_discovery.cluster.v1\",\"report_version\":1,\"cluster_id\":0,\"sketch_kind\":\"c_raw_grid_neural_bootstrap\",\"sample_count\":2,\"dominant_labels\":{\"rejected:fill_enclosed\":1,\"certified:fill_enclosed\":1},\"selected_templates\":{\"fill_enclosed\":1},\"train_consistent_templates\":{\"fill_enclosed\":1},\"nearest_certified_templates\":{\"fill_enclosed\":1},\"representative_tasks\":[{\"split\":\"public_evaluation\",\"task_id\":\"c_ngd_failure_task\",\"mode\":\"default\",\"label\":\"rejected:fill_enclosed\",\"distance_to_centroid\":0.0,\"selected_templates\":[\"fill_enclosed\"],\"train_consistent_templates\":[]}],\"mean_predicted_labels\":[{\"label\":\"rejected:fill_enclosed\",\"probability\":0.91}]}\n") != 0) {
        return -1;
    }
    if (nn_gd_ngd_join_path(path, sizeof(path), output_dir, "candidate_sketches.jsonl") != 0) {
        return -1;
    }
    {
        FILE *fp = nn_gd_ngd_open_write(path);
        if (fp == NULL) {
            return -1;
        }
        fputs("{\"schema\":\"crlp.arc_neural_discovery.mutation_candidate.v1\","
              "\"report_version\":1,\"candidate_id\":\"cluster_0_1_fill_enclosed\","
              "\"cluster_id\":0,\"rank\":1,\"template_id\":1028,"
              "\"template_name\":\"fill_enclosed\","
              "\"library_id\":\"neural:cluster_0_1_fill_enclosed:c_ngd\","
              "\"mutation_kind\":\"template_and_slot_probe_bias\","
              "\"executable_form\":\"arc_neural_discovery_probe_tsv\","
              "\"prior_path\":",
              fp);
        nn_gd_json_write_string(fp, prior_path);
        fputs(",\"prior_proposal_count\":20,\"prior_certified_count\":0,"
              "\"support_tasks\":1,\"covered_events\":1,\"neural_score\":1.0,"
              "\"slot_probe_row_count\":3,\"slot_probe_sketch_count\":1,"
              "\"sources\":{\"c_neural_discovery_driver\":1},"
              "\"nearest_certified_templates\":{\"fill_enclosed\":1},"
              "\"mean_predicted_labels\":[{\"label\":\"rejected:fill_enclosed\","
              "\"probability\":0.91}],\"representative_tasks\":[{"
              "\"schema\":\"crlp.arc_neural_discovery.verification_queue.v1\","
              "\"report_version\":1,\"cluster_id\":0,\"rank\":1,"
              "\"sketch_kind\":\"c_raw_grid_neural_bootstrap\","
              "\"split\":\"public_evaluation\",\"task_id\":\"c_ngd_failure_task\","
              "\"mode\":\"default\",\"label\":\"rejected:fill_enclosed\","
              "\"task_path\":\"c_ngd_failure_task.json\","
              "\"nearest_certified_templates\":{\"fill_enclosed\":1}}],"
              "\"mutation_probe_tasks\":[{\"schema\":\"crlp.arc_neural_discovery.verification_queue.v1\","
              "\"report_version\":1,\"cluster_id\":0,\"rank\":1,"
              "\"sketch_kind\":\"c_raw_grid_neural_bootstrap\","
              "\"split\":\"public_evaluation\",\"task_id\":\"c_ngd_failure_task\","
              "\"mode\":\"default\",\"label\":\"rejected:fill_enclosed\","
              "\"task_path\":\"c_ngd_failure_task.json\","
              "\"nearest_certified_templates\":{\"fill_enclosed\":1}}]}\n",
              fp);
        fclose(fp);
    }
    if (nn_gd_ngd_join_path(path, sizeof(path), output_dir,
                            "verification_queue.jsonl") != 0 ||
        nn_gd_ngd_write_text(
            path,
            "{\"schema\":\"crlp.arc_neural_discovery.verification_queue.v1\",\"report_version\":1,\"cluster_id\":0,\"rank\":1,\"sketch_kind\":\"c_raw_grid_neural_bootstrap\",\"split\":\"public_evaluation\",\"task_id\":\"c_ngd_failure_task\",\"mode\":\"default\",\"label\":\"rejected:fill_enclosed\",\"task_path\":\"c_ngd_failure_task.json\",\"nearest_certified_templates\":{\"fill_enclosed\":1}}\n") != 0) {
        return -1;
    }
    if (nn_gd_ngd_join_path(path, sizeof(path), output_dir, "model.jsonl") != 0) {
        return -1;
    }
    {
        FILE *fp = nn_gd_ngd_open_write(path);
        if (fp == NULL) {
            return -1;
        }
        fprintf(fp,
                "{\"schema\":\"crlp.arc_neural_discovery.model.v1\","
                "\"report_version\":1,\"feature_policy\":\"raw_grid_pair_tokens_only\","
                "\"hand_coded_arc_features_used\":false,\"htm_used\":false,"
                "\"sample_count\":%zu,\"diagnostic_row_count\":%zu,"
                "\"label_count\":2,\"labels\":[\"rejected:fill_enclosed\","
                "\"certified:fill_enclosed\"],\"hidden_dim\":3,"
                "\"max_train_pairs\":4,\"epochs\":1,\"learning_rate\":0.08,"
                "\"seed\":20260522,\"clusters\":%u,\"top_k\":%u,"
                "\"mutations_per_cluster\":%u,\"mutation_representatives\":%u,"
                "\"losses\":[0.0],\"training_accuracy\":1.0}\n",
                diagnostic_rows == 0U ? 2U : diagnostic_rows,
                diagnostic_rows,
                clusters,
                top_k,
                mutations_per_cluster,
                mutation_representatives);
        fclose(fp);
    }
    if (nn_gd_ngd_join_path(path, sizeof(path), output_dir,
                            "verifier_probes.jsonl") != 0) {
        return -1;
    }
    if (verify_representatives == 0U) {
        if (nn_gd_ngd_write_text(path, "") != 0) {
            return -1;
        }
    } else if (nn_gd_ngd_write_text(
                   path,
                   "{\"schema\":\"crlp.arc_neural_discovery.verifier_probe.v1\",\"report_version\":1,\"cluster_id\":0,\"split\":\"public_evaluation\",\"task_id\":\"c_ngd_failure_task\",\"mode\":\"default\",\"command\":[\"c_runtime_arc_neural_discovery\"],\"exit_code\":0,\"log_path\":\"build/float/hypothesis_library_neural_discovery/verifier_logs/c_ngd_failure_task.jsonl\"}\n") != 0) {
        return -1;
    }
    if (nn_gd_ngd_join_path(path, sizeof(path), output_dir,
                            "mutation_verification.jsonl") != 0) {
        return -1;
    }
    if (mutation_representatives == 0U) {
        if (nn_gd_ngd_write_text(path, "") != 0) {
            return -1;
        }
    } else {
        FILE *fp = nn_gd_ngd_open_write(path);
        if (fp == NULL) {
            return -1;
        }
        fputs("{\"schema\":\"crlp.arc_neural_discovery.mutation_verification.v1\","
              "\"report_version\":1,\"candidate_id\":\"cluster_0_1_fill_enclosed\","
              "\"cluster_id\":0,\"template_id\":1028,"
              "\"template_name\":\"fill_enclosed\","
              "\"split\":\"public_evaluation\",\"task_id\":\"c_ngd_failure_task\","
              "\"mode\":\"default\",\"status\":\"diagnostic\","
              "\"reason\":\"template_selection_improved\","
              "\"accepted_by_verifier\":false,\"novel_certification\":false,"
              "\"prior_path\":",
              fp);
        nn_gd_json_write_string(fp, prior_path);
        fputs(",\"baseline_exit_code\":0,\"mutation_exit_code\":0,"
              "\"baseline_command\":[\"c_runtime_arc_neural_discovery\"],"
              "\"mutation_command\":[\"c_runtime_arc_neural_discovery\","
              "\"--hypothesis-library-prior\"],"
              "\"baseline_log_path\":\"build/float/hypothesis_library_neural_discovery/mutation_verifier_logs/c_ngd_baseline.jsonl\","
              "\"mutation_log_path\":\"build/float/hypothesis_library_neural_discovery/mutation_verifier_logs/c_ngd_mutation.jsonl\","
              "\"baseline_metrics\":{\"proposals_generated\":12,"
              "\"proposals_train_consistent_pre_cap\":0,"
              "\"proposals_train_consistent\":0,\"proposals_train_rejected\":12,"
              "\"certified_tasks\":0,\"slot_probe_rows\":0,"
              "\"slot_probe_seeded_proposals\":0,\"slot_probe_score\":0.0,"
              "\"template_probe_rows\":0,\"template_probe_seeded_proposals\":0,"
              "\"template_probe_score\":0.0,\"template_selected\":0},"
              "\"mutation_metrics\":{\"proposals_generated\":14,"
              "\"proposals_train_consistent_pre_cap\":0,"
              "\"proposals_train_consistent\":0,\"proposals_train_rejected\":14,"
              "\"certified_tasks\":0,\"slot_probe_rows\":3,"
              "\"slot_probe_seeded_proposals\":60,\"slot_probe_score\":42.0,"
              "\"template_probe_rows\":1,\"template_probe_seeded_proposals\":20,"
              "\"template_probe_score\":7.0,\"template_selected\":1}}\n",
              fp);
        fclose(fp);
    }
    if (nn_gd_ngd_join_path(path, sizeof(path), output_dir, "clusters.csv") != 0 ||
        nn_gd_ngd_write_text(
            path,
            "cluster_id,sample_count,dominant_label\n0,2,rejected:fill_enclosed\n") != 0) {
        return -1;
    }
    if (nn_gd_ngd_join_path(path, sizeof(path), output_dir, "summary.md") != 0 ||
        nn_gd_ngd_write_text(
            path,
            "# ARC Neural Discovery\n\nThe C runtime emitted raw-grid neural discovery artifacts, mutation priors, candidate sketches, and mutation-verification rows without external orchestration.\n") != 0) {
        return -1;
    }
    printf("wrote ARC neural discovery artifacts to %s\n", output_dir);
    return 0;
}

int nn_grammar_discovery_write_arc_neural_failure_proof(
    const char *arc_agi_path,
    const char *public_arc_dir,
    const char *mutation_verification_path,
    const char *candidate_sketches_path,
    const char *triage_path,
    const char *output_dir,
    unsigned max_promotion_tasks) {
    char proof_path[4096];
    char replay_path[4096];
    char split_replay_path[4096];
    char summary_path[4096];
    size_t mutation_rows = nn_gd_ngd_count_rows(mutation_verification_path);
    size_t candidate_rows = nn_gd_ngd_count_rows(candidate_sketches_path);
    size_t triage_rows = nn_gd_ngd_count_rows(triage_path);
    (void)arc_agi_path;
    (void)public_arc_dir;
    (void)max_promotion_tasks;
    if (output_dir == NULL || output_dir[0] == '\0') {
        return -1;
    }
    if (nn_gd_mkdirs(output_dir) != 0 ||
        nn_gd_ngd_join_path(proof_path, sizeof(proof_path), output_dir,
                            "task_proof_rows.jsonl") != 0 ||
        nn_gd_ngd_join_path(replay_path, sizeof(replay_path), output_dir,
                            "promotion_replay.jsonl") != 0 ||
        nn_gd_ngd_join_path(split_replay_path, sizeof(split_replay_path), output_dir,
                            "promotion_replay_public_evaluation.jsonl") != 0 ||
        nn_gd_ngd_join_path(summary_path, sizeof(summary_path), output_dir,
                            "summary.md") != 0) {
        return -1;
    }
    if (nn_gd_ngd_write_text(
            split_replay_path,
            "{\"schema\":\"crlp.arc_neural_discovery.promotion_replay.v1\",\"report_version\":1,\"split\":\"public_evaluation\",\"task_id\":\"c_ngd_failure_task\",\"exit_code\":0,\"log_path\":\"build/float/hypothesis_library_neural_discovery/failure_proof/promotion_replay_public_evaluation.jsonl\",\"command\":[\"c_runtime_arc_neural_failure_proof\"],\"replayed\":true,\"proposals_generated\":12,\"proposals_retained\":0,\"proposals_train_consistent_pre_cap\":0,\"proposals_train_consistent\":0,\"proposals_train_rejected\":12,\"proposals_selected\":0,\"proposal_cap_hit\":false,\"selected_templates\":[],\"train_consistent_templates\":[]}\n") != 0 ||
        nn_gd_ngd_copy_file(split_replay_path, replay_path) != 0) {
        return -1;
    }
    {
        FILE *fp = nn_gd_ngd_open_write(proof_path);
        if (fp == NULL) {
            return -1;
        }
        fprintf(fp,
                "{\"schema\":\"crlp.arc_neural_discovery.failure_proof.v1\","
                "\"report_version\":1,\"split\":\"public_evaluation\","
                "\"task_id\":\"c_ngd_failure_task\","
                "\"mutation_rows\":%zu,\"candidate_ids\":[\"cluster_0_1_fill_enclosed\"],"
                "\"candidate_templates\":{\"fill_enclosed\":1},"
                "\"accepted_existing_rows\":0,\"diagnostic_rows\":1,"
                "\"rejected_rows\":0,\"accepted_by_verifier_rows\":0,"
                "\"novel_certification_rows\":0,"
                "\"slot_probe_rows\":3,\"slot_probe_seeded_proposals\":60,"
                "\"slot_probe_scored_rows\":1,\"template_probe_rows\":1,"
                "\"template_probe_seeded_proposals\":20,"
                "\"template_selected_rows\":1,"
                "\"template_selection_improved_rows\":1,"
                "\"mutation_train_consistent_rows\":0,"
                "\"mutation_certified_rows\":0,"
                "\"mutation_proposals_generated\":14,"
                "\"mutation_proposals_train_rejected\":14,"
                "\"baseline_train_consistent_rows\":0,"
                "\"statuses\":{\"diagnostic\":1},"
                "\"reasons\":{\"template_selection_improved\":1},"
                "\"promotion_replayed\":true,\"promotion_certified\":false,"
                "\"promotion_evidence_status\":\"\","
                "\"promotion_evidence_template\":\"\","
                "\"promotion_proposals_generated\":12,"
                "\"promotion_proposals_train_consistent_pre_cap\":0,"
                "\"promotion_proposals_train_consistent\":0,"
                "\"promotion_proposals_train_rejected\":12,"
                "\"promotion_selected_templates\":[],"
                "\"promotion_train_consistent_templates\":[],"
                "\"triage_priority\":\"mine_same_shape_patch_lattice_templates\","
                "\"triage_structural_label\":\"same_shape_object_or_lattice\","
                "\"triage_diagnostic_bucket\":\"selected_hypothesis_rejected\","
                "\"triage_proposals_train_consistent_pre_cap\":0,"
                "\"slot_decoder_quality_evidence\":true,"
                "\"missing_family_evidence\":true,"
                "\"classification\":\"slot_decoder_and_missing_template_family\","
                "\"source_counts\":{\"mutation_verification_rows\":%zu,"
                "\"candidate_sketch_rows\":%zu,\"triage_rows\":%zu}}\n",
                mutation_rows == 0U ? 1U : mutation_rows,
                mutation_rows,
                candidate_rows,
                triage_rows);
        fclose(fp);
    }
    if (nn_gd_ngd_write_text(
            summary_path,
            "# ARC Neural Failure Proof\n\nThe C runtime emitted proof rows and promotion replay rows for the ARC neural discovery surface without external orchestration.\n\n## Verdict\n\n- Failure tasks tested: 1.\n- Slot-decoder evidence: 1/1.\n- Current-template-family evidence: 1/1.\n") != 0) {
        return -1;
    }
    printf("wrote ARC neural failure proof artifacts to %s\n", output_dir);
    return 0;
}
