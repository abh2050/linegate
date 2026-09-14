# linegate scorecard

Gates: 0 pass · 1 pass · 2 pass · 3 pass · 4 RED · 5 pass · 6 pass · 7 pass. Holdout runs: 1. Policy: `policy-92773335dcb7`.

## Model

| Measure | Value |
|---|---|
| Baseline MCC (validation, v1 baseline) | 0.1559 |
| Baseline MCC (validation, baseline_v2, 3 seeds) | 0.2267 |
| Search reference MCC (baseline and placebo mean) | 0.2210 |
| Post-search MCC (validation) | 0.2210; accepted features: none |
| Holdout MCC at the validation threshold | 0.1100 (validation rehearsal 0.2267) |
| Holdout AUC / average precision | 0.5626 / 0.0297 |

## Search and agents

| Measure | Value |
|---|---|
| Proposal survival rate | 66.7% (14 of 21 decided proposals) |
| Quarantined | 9: mindate_id_diff, mindate_id_diff_reverse, l3_s30_spread, l3_s29_spread, l3_s29_dwell, l1_s24_variant_a, crossline_l3_source_mix, crossline_source_code, l1s24_block_presence_pattern |
| Token spend | $9.14 total (disposition $0.10, hypothesis $8.69, policy-watcher $0.00, warden-red-team $0.25, warden-search $0.09); prices in config/agents.yaml are assumptions |
| Token spend per MCC point | undefined: the search produced no MCC gain for $8.69 of search spend |
| Fabricated citations (agent dispositions) | 0 of 6 |
| Merge attempts refused | 2 |

## Decisions on the holdout (236,616 parts, 964 failures)

| Measure | Value |
|---|---|
| Committed share (shipped or inspected without review) | 74.3% |
| Abstain share (sent to human review) | 25.7% |
| Escaped errors (failures shipped below the band) | 629 of 964 |
| Failures in review / inspected | 307 / 28 |
| Expected cost at the policy threshold | $18,010 per shift vs $20,099 shipping everything |

## Honest vs leaky

| Model | MCC |
|---|---|
| Honest (holdout, this project) | 0.1100 |
| Leaky (2016 leaderboard, Id ordering leak) | 0.49 |

The leaky number cannot run on a line because it compares each part's Id with the next part that started at the same
moment, and on a live line that next part has not been built yet; the warden measured that feature's lift at
0.206, falling to 0.001 once Id order was randomized.
