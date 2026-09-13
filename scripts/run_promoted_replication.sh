#!/bin/sh
set -eu

real=${REAL:-float}
out_dir=${1:-build/$real/promoted_replication}
grid_bin=${GRID_PATH_BIN:-build/$real/grid_path}
claim_bin=${DELAYED_CLAIM_BIN:-build/$real/delayed_claim}
grid_model=${GRID_PATH_MODEL:-build/$real/grid_path_no_priors.nnckpt}
claim_model=${DELAYED_CLAIM_MODEL:-build/$real/delayed_claim_v2.nnckpt}
seeds=${REPLICATION_TEST_SEEDS:-900000,910000,920000,930000,940000}

for path in "$grid_bin" "$claim_bin" "$grid_model" "$claim_model"; do
    test -f "$path" || { echo "replication: missing $path" >&2; exit 1; }
done

mkdir -p "$out_dir/logs"
: > "$out_dir/raw.jsonl"

old_ifs=$IFS
IFS=,
for seed in $seeds; do
    IFS=$old_ifs
    grid_log="$out_dir/logs/grid-$seed.log"
    claim_log="$out_dir/logs/claim-$seed.log"
    "$grid_bin" --compare --test-seed "$seed" --rule-family-sweep "$grid_model" > "$grid_log"
    "$claim_bin" --compare --test-seed "$seed" "$claim_model" > "$claim_log"
    awk '/^\{"schema":"crlp.(compare|cost_breakdown).v1"/{print}' "$grid_log" >> "$out_dir/raw.jsonl"
    awk '/^\{"schema":"crlp.compare.v1"/{print}' "$claim_log" >> "$out_dir/raw.jsonl"
    IFS=,
done
IFS=$old_ifs

robustness_log="$out_dir/logs/claim-robustness.log"
"$claim_bin" --rule-search "$claim_model" --rule-family direct --max-rules 8 \
    --proposer neural --proposer-seed 20240511 > "$robustness_log"
awk '/^\{"schema":"crlp.delayed_claim.robustness.v1"/{print}' \
    "$robustness_log" > "$out_dir/robustness.jsonl"

awk '
function value(line, key,    pattern, token) {
    pattern="\"" key "\":[^,}]+"
    if (!match(line, pattern)) return ""
    token=substr(line, RSTART + length(key) + 3, RLENGTH - length(key) - 3)
    gsub(/^\"|\"$/, "", token)
    return token
}
function add(metric, x) { n[metric]++; sum[metric]+=x; sumsq[metric]+=x*x }
function emit(metric, task, mode, name,    mean,variance,sd,half) {
    mean=sum[metric]/n[metric]
    variance=n[metric]>1 ? (sumsq[metric]-sum[metric]*sum[metric]/n[metric])/(n[metric]-1) : 0
    if (variance < 0 && variance > -1e-15) variance=0
    sd=sqrt(variance); half=(n[metric]==5 ? 2.776 : 1.96)*sd/sqrt(n[metric])
    printf "{\"schema\":\"crlp.replication.summary.v1\",\"task\":\"%s\",\"evaluation_mode\":\"%s\",\"metric\":\"%s\",\"generator_seed_count\":%d,\"mean\":%.10f,\"sample_stdev\":%.10f,\"ci95_low\":%.10f,\"ci95_high\":%.10f}\n", task,mode,name,n[metric],mean,sd,mean-half,mean+half
}
{
    task=value($0,"task"); mode=value($0,"evaluation_mode"); rules=value($0,"rule_set_hash")
    seed=value($0,"benchmark_seed")
    schema=value($0,"schema")
    if (schema=="crlp.cost_breakdown.v1" && task=="grid_path") {
        total=value($0,"total_ms_per_sample")+0
        if (mode=="raw") { raw_cost[seed]=total; add("grid_raw_cost",total) }
        else if (rules=="grid-cheap-bound-budget-v5") { cheap_cost[seed]=total; add("grid_cheap_cost",total) }
        else if (rules=="grid-distance-budget-v5") { mixed_cost[seed]=total; add("grid_mixed_cost",total) }
    } else if (task=="grid_path" && mode=="raw") {
        raw[seed]=value($0,"accuracy")+0; add("grid_raw",raw[seed])
    } else if (task=="grid_path" && rules=="grid-cheap-bound-budget-v5") {
        cheap[seed]=value($0,"accuracy")+0; add("grid_cheap",cheap[seed])
    } else if (task=="grid_path" && rules=="grid-distance-budget-v5") {
        mixed[seed]=value($0,"accuracy")+0; add("grid_mixed",mixed[seed])
    } else if (task=="delayed_claim" && mode=="raw") {
        add("claim_raw",value($0,"accuracy")+0)
        add("claim_ece",value($0,"ece")+0)
    }
}
END {
    for (seed in raw) {
        if (seed in cheap) add("grid_cheap_delta",cheap[seed]-raw[seed])
        if (seed in mixed) add("grid_mixed_delta",mixed[seed]-raw[seed])
    }
    for (seed in raw_cost) {
        if (seed in cheap_cost) add("grid_cheap_cost_delta",cheap_cost[seed]-raw_cost[seed])
        if (seed in mixed_cost) add("grid_mixed_cost_delta",mixed_cost[seed]-raw_cost[seed])
    }
    emit("grid_raw","grid_path","raw","accuracy")
    emit("grid_cheap","grid_path","cheap_bound","accuracy")
    emit("grid_mixed","grid_path","mixed","accuracy")
    emit("grid_cheap_delta","grid_path","cheap_bound_vs_raw","paired_accuracy_delta")
    emit("grid_mixed_delta","grid_path","mixed_vs_raw","paired_accuracy_delta")
    emit("claim_raw","delayed_claim","raw","accuracy")
    emit("claim_ece","delayed_claim","raw","ece")
    emit("grid_raw_cost","grid_path","raw","total_ms_per_sample")
    emit("grid_cheap_cost","grid_path","cheap_bound","total_ms_per_sample")
    emit("grid_mixed_cost","grid_path","mixed","total_ms_per_sample")
    emit("grid_cheap_cost_delta","grid_path","cheap_bound_vs_raw","paired_total_ms_delta")
    emit("grid_mixed_cost_delta","grid_path","mixed_vs_raw","paired_total_ms_delta")
}' "$out_dir/raw.jsonl" > "$out_dir/summary.jsonl"

awk '
function value(line, key,    pattern, token) {
    pattern="\"" key "\":[^,}]+"; match(line,pattern)
    token=substr(line,RSTART+length(key)+3,RLENGTH-length(key)-3); gsub(/^\"|\"$/,"",token); return token
}
BEGIN { print "# Promoted Independent-Generator Replication\n"; print "Five disjoint 1,000-state test bands; checkpoints and certified rule sets are held fixed. Intervals are two-sided 95% t intervals over generator bands.\n"; print "| Task / comparison | Mean | 95% CI |"; print "| --- | ---: | ---: |" }
{
 task=value($0,"task"); mode=value($0,"evaluation_mode"); metric=value($0,"metric"); mean=value($0,"mean"); low=value($0,"ci95_low"); high=value($0,"ci95_high")
 scale=(metric ~ /accuracy/)?100:1; suffix=(scale==100)?"%":((metric ~ /_ms/)?" ms":"")
 if (metric ~ /_ms/) printf "| `%s` %s %s | %.6f%s | [%.6f, %.6f]%s |\n",task,mode,metric,mean,suffix,low,high,suffix
 else printf "| `%s` %s %s | %.3f%s | [%.3f, %.3f]%s |\n",task,mode,metric,mean*scale,suffix,low*scale,high*scale,suffix
}' "$out_dir/summary.jsonl" > "$out_dir/summary.md"

awk '
function value(line,key, pattern,token) { pattern="\"" key "\":[^,}]+"; match(line,pattern); token=substr(line,RSTART+length(key)+3,RLENGTH-length(key)-3); gsub(/^\"|\"$/,"",token); return token }
value($0,"metric")=="paired_total_ms_delta" {
 mode=value($0,"evaluation_mode"); mean=value($0,"mean")+0; low=value($0,"ci95_low")+0; high=value($0,"ci95_high")+0
 verdict=(high<0)?"improves_total_cost":((low>0)?"regresses_total_cost":"inconclusive")
 promotable=(verdict=="improves_total_cost")?"true":"false"
 printf "{\"schema\":\"crlp.cost_breakdown.verdict.v1\",\"task\":\"grid_path\",\"comparison\":\"%s\",\"mean_total_ms_delta\":%.10f,\"ci95_low\":%.10f,\"ci95_high\":%.10f,\"verdict\":\"%s\",\"efficiency_claim_promotable\":%s,\"certificate_construction_included\":false}\n",mode,mean,low,high,verdict,promotable
}' "$out_dir/summary.jsonl" > "$out_dir/cost_verdict.jsonl"

echo "replication: wrote $out_dir/summary.jsonl"

awk '
function value(line,key, pattern,token) { pattern="\"" key "\":[^,}]+"; match(line,pattern); token=substr(line,RSTART+length(key)+3,RLENGTH-length(key)-3); gsub(/^\"|\"$/,"",token); return token }
{
 policy=value($0,"policy"); noise=value($0,"observation_noise")+0; false_count=value($0,"false_rejections")+0
 fallback=value($0,"support_fallback_rate")+0; utility=value($0,"decision_utility")+0
 if (policy=="direct_gate") { direct_rows++; direct_false+=false_count; direct_utility+=utility }
 else if (policy=="support_aware_fallback") { protected_rows++; protected_false+=false_count; protected_utility+=utility; if (fallback>max_fallback) max_fallback=fallback }
}
END {
 pass=(direct_rows==6 && protected_rows==6 && direct_false>0 && protected_false==0)
 printf "{\"schema\":\"crlp.delayed_claim.robustness.verdict.v1\",\"noise_levels\":6,\"samples_per_level\":1000,\"direct_gate_false_rejections\":%d,\"support_aware_false_rejections\":%d,\"max_support_fallback_rate\":%.10f,\"mean_direct_utility\":%.10f,\"mean_support_aware_utility\":%.10f,\"utility_delta\":%.10f,\"bounded_safety_pass\":%s,\"claim_scope\":\"observed_uncertainty_margin_0p08_on_registered_noise_sweep\"}\n",direct_false,protected_false,max_fallback,direct_utility/direct_rows,protected_utility/protected_rows,(protected_utility/protected_rows)-(direct_utility/direct_rows),pass?"true":"false"
}' "$out_dir/robustness.jsonl" > "$out_dir/robustness_verdict.jsonl"
