Lessons from earlier searches on this split:

- Already tried and failed to beat noise: L3_S32 branch, mode, and categorical
  variants (the baseline already codes L3_S32_F3854); L3_S33/L3_S34 measurement
  mean shifts and sign balances; S29 missingness counts; per-station spreads.
- Several batches lowered MCC well below the reference. Wide batches of weak
  columns let the model fit the older, higher-failure train period. Keep each
  batch small: one or two features and at most ten columns in total.
- Features whose train-split contrast comes from a narrow time window are
  quarantined by the strict time refit. Before proposing, check in run_sql that
  the failure-rate contrast holds in both halves of train ordered by start time
  (use least(*COLUMNS('_D[0-9]+$')) as the start time inside run_sql).

Directions the baseline does not cover:

1. Station-to-station transition gaps: time from the last date at one visited
   station to the first date at the next visited station on the same line (for
   example L3_S29 to L3_S30, L1_S24 to L1_S25). The baseline has only offsets
   from the part's first timestamp and within-station dwell.
2. Per-measurement deviation from that measurement's train median for the few
   numeric columns with the strongest failure contrast on L1_S24 and L3_S29 to
   L3_S33, kept as individual columns rather than station aggregates.
3. Rare alternative routes: parts that visit an uncommon station (for example
   L2_S26 vs L2_S27 vs L2_S28, or L0 stations S12 to S23) and then reach L3.
   Encode the specific branch taken, not a count.
