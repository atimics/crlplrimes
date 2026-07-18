# Promoted Independent-Generator Replication

Five disjoint 1,000-state test bands; checkpoints and certified rule sets are held fixed. Intervals are two-sided 95% t intervals over generator bands.

| Task / comparison | Mean | 95% CI |
| --- | ---: | ---: |
| `grid_path` raw accuracy | 71.300% | [69.471, 73.129]% |
| `grid_path` cheap_bound accuracy | 87.500% | [86.106, 88.894]% |
| `grid_path` mixed accuracy | 92.180% | [91.197, 93.163]% |
| `grid_path` cheap_bound_vs_raw paired_accuracy_delta | 16.200% | [14.192, 18.208]% |
| `grid_path` mixed_vs_raw paired_accuracy_delta | 20.880% | [19.184, 22.576]% |
| `delayed_claim` raw accuracy | 96.020% | [94.685, 97.355]% |
| `delayed_claim` raw ece | 0.010 | [0.004, 0.016] |
| `grid_path` raw total_ms_per_sample | 0.003529 ms | [0.003439, 0.003619] ms |
| `grid_path` cheap_bound total_ms_per_sample | 0.002977 ms | [0.002850, 0.003104] ms |
| `grid_path` mixed total_ms_per_sample | 0.003471 ms | [0.003393, 0.003548] ms |
| `grid_path` cheap_bound_vs_raw paired_total_ms_delta | -0.000552 ms | [-0.000735, -0.000369] ms |
| `grid_path` mixed_vs_raw paired_total_ms_delta | -0.000059 ms | [-0.000171, 0.000054] ms |
