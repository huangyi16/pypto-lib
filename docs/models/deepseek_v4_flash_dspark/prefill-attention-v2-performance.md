# DSpark prefill attention: paired v2 evidence

Measured baseline: `7dc555059037bacced5a4f0225856202e100d3fa`. The optimized capture uses `3dc359fe57419506b42a19989ff072a7898a50ed` plus the six selected prefill changes, including ratio-128 pooling. The captured file hashes identify the measured source, before this PR was rebased. These results are historical evidence for that combined snapshot, not an incremental measurement of the rebased PR against current main. The measured snapshot includes output-projection column-view changes. Upstream #1814 independently adds overlapping output views and different cube tiling and L1 pipelines. This PR retains those upstream settings and integrates its remaining quantization/input views; the combined historical gains must not be attributed entirely to the remaining PR diff.

Both variants used Ascend A3 / 910C (8 physical, 16 logical devices), TP4 / DP4 / EP16, 16 local experts per rank, one prompt per DP group, the same frozen inputs and weights, and the same pinned toolchain. PyPTO `ce0eb7c`, simpler `d8f997fa`, PTOAS `0.67`, PTO ISA `b1d460ee`, CANN `9.0.0`.

The protocol discards two warmups and retains five measured traversals with all 16 raw rank sample arrays. Each traversal starts from empty caches and retains state over 37 consecutive chunks: at most 8192 tokens each, ending with 5088 tokens. Prefix totals are measured at 8192, 65536, 131072, and 300000 tokens. Both capture variants use the same host-only selector adapter for CSA L4; it changes no model or golden arithmetic.

Device effective latency is the device-domain orchestration/scheduler window union for each L2, accumulated on the same rank across L2s and chunks. It excludes host dispatch gaps. A representative layer includes both attention and the complete MoE L2. It is not attention-only latency, whole-model latency or serving wall time.

## 8192-token representative layers

Values are milliseconds. The primary statistic is the lowest per-rank median; +/- is the sample standard deviation over its five samples. The secondary statistic takes the maximum rank per round, then the median. Its start skew remains included.

| Layer | Baseline fastest median +/- SD | Optimized fastest median +/- SD | Reduction | Baseline max-rank median +/- SD | Optimized max-rank median +/- SD |
|---|---:|---:|---:|---:|---:|
| SWA / hash, L0 | 36.419 +/- 0.086 | 31.678 +/- 0.240 | 13.02% | 45.728 +/- 0.144 | 41.595 +/- 0.929 |
| CSA / hash, L2 | 50.735 +/- 0.584 | 42.645 +/- 0.225 | 15.95% | 59.476 +/- 0.321 | 51.309 +/- 0.323 |
| CSA / score, L4 | 49.428 +/- 0.701 | 42.030 +/- 0.156 | 14.97% | 58.288 +/- 0.992 | 51.387 +/- 0.233 |
| HCA / score, L3 | 61.194 +/- 0.237 | 36.702 +/- 0.431 | 40.02% | 70.771 +/- 0.331 | 45.312 +/- 0.506 |

## Independent-component weighted estimates

For each rank, weight representative-layer medians by 2 SWA/hash, 1 CSA/hash, 20 CSA/score and 20 HCA/score layers. Add head and tail medians once per chunk size, plus exactly one terminal bootstrap reading the last 128 tokens (B4 capacity, one active request per DP group). Only then select the minimum or maximum rank total. The table reports the minimum total rank. It does not sum independently selected fastest ranks.

| Complete prompt tokens | Baseline estimate ms | Optimized estimate ms | Estimated reduction |
|---:|---:|---:|---:|
| 8192 | 2374.882 | 1722.466 | 27.47% |
| 65536 | 19811.320 | 14460.891 | 27.01% |
| 131072 | 41559.302 | 30625.418 | 26.31% |
| 300000 | 106200.985 | 80176.104 | 24.51% |

These are estimates from independent component captures, not measured 43-layer or serving latency. They exclude target taps and host/serving overhead. They do not establish a 3x end-to-end speedup. The four representative-layer cases are the same newly measured cohort previously published before the auxiliary captures completed; this full-suite publication does not claim a second sampling of those cases.

## Validation of the captured snapshots and interpretation limits

- All four representative layers passed the original 32-token Torch goldens with the original comparators and tolerances.
- Both CSA cases passed cache-continuation checks for 64 slots on each of 16 ranks. The long replay checks finite values in 64 sampled output rows after the final chunk.
- The original bounded drafter and Markov goldens were frozen once and replayed. Bootstrap publication and finite/confidence contracts, head vocabulary contracts, and exact tail checks for both 8192 and 5088 rows passed.
- Full generic representative-layer compilation and the 32/8192/5088 argument ABI checks passed for the captured optimized source. These checks do not claim post-rebase validation or performance measurement.
- Native identities, all 16 x 5 timing samples, graph/timing ownership, and complete rank/task-data grids were checked. Auxiliary activity is reported as an entire-L2 interval union, without speculative mapping of dynamic graph task IDs to timing leaves. Optional missing profile native trees retain null identities and are bound through complete parent completion cohorts and graph/timing sidecars.
- No complete 300K Torch golden or real training checkpoint was used. Five samples are a quick comparison; dispersion is retained and is not a formal significance test. Small auxiliary differences are not claimed as optimizations.

## Rebased integration checks

The submitted combination is based on main `32b7eb1c76b70744995220e7b5a23db5aa25697a`. It retains upstream #1814 O-projection tiling, weight flattening, L1/L0 pipelines and padded-row initialization while preserving the remaining quantization/input column views. No additional tile tuning was performed.

After integration, all four representative layers were freshly compiled and passed the original 32-token Torch golden comparisons on TP4/DP4/EP16 with unchanged comparators and tolerances. Compiled argument ABI checks also passed at 32, 8192 and 5088 tokens. The same pinned toolchain listed above was used. The numerical-data file records the exact six model hashes checked.

These new checks do not rerun the long traversal, cache-continuity test, profiling or performance sampling. The check-only collector also reads 64 resident output rows, including padding after the 32 active rows; that is not a long-context finite-output validation. The performance tables above remain measurements of the earlier captured combination, not measurements of the rebased code.

## Audit data

[Numerical samples](prefill-attention-v2-samples.json) contains all 3840 original numerical samples from the 48 benchmark blocks and both fastest-rank and per-round-max summaries. Statistics and all four same-rank weighted estimates were independently recomputed. It contains no tensors, private machine paths or credentials.
