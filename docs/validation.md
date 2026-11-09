# Validation status

Assembly date: 9 November 2026. The reference is the final integrated
manuscript. This is a reproducibility implementation under validation, not a
claim of a completed independent production rerun.

## Completed checks

- The manuscript include tree was traversed: all 61 scientific figures in
  Chapters 2–5 and Appendices A–B are indexed and their final reference PDFs
  are copied with SHA-256 identities. The two remaining tables are indexed.
- Nineteen passages referring to the accompanying project were extracted.
  Each has an explicit link to the corresponding detailed-method section.
  Sixteen technical calculation records and the numerical configurations
  supply the expanded settings and diagnostics.
- The standalone scientific test command passed **111 tests**. Tests cover
  event counts, boundaries and mechanism semantics; exact neighbour distances;
  metric definitions; equilibrium algebra and residuals; parameter-domain
  logic; conditional sampling; weighted tied ranks; separate bootstrap cohorts;
  summaries; and reconstruction/digitisation fixtures.
- The tracked bundle was copied to a separate location without the author's
  working data, caches or parent repository. The integrity check and all
  **111 tests passed again** there.
- The study-design table is generated from the final configurations. The
  eta-sensitivity table was recalculated from the original factorial summary;
  all six evaluated/lower/equal/higher rows match the final Appendix A table.
  This is a summary reconstruction, not a new factorial simulation.
- The independent original-catalogue selection check recreated **all 10,000
  real-landscape identifiers in their original order**, from 161,440 candidate
  records, using seeds 20260833 and 20260902. No dataset was added to Git.
  See [selection audit](validation/real_selection.json).
- Two real landscapes shown in the final supplementary figures were extracted
  again from the original JRC forest raster on the source machine. Both binary
  raster hashes matched exactly. See [raster audit](validation/real_rasters.json).
- One complete 88-level trajectory was freshly replayed for each removal
  mechanism (only random, edge random and access based; replicate 1, with
  rho = 0.999 for directed removal). Raster identities, cluster counts and
  event counters matched exactly. All six metric values matched, with the
  original ED denominator converted once to the manuscript units. See
  [random replay](validation/trajectory_replay.json) and
  [directed replays](validation/directed_replay.json). This checks 264 states,
  not the full 1,400-trajectory ensemble.
- The constructed hybrid-distance and majority-aggregation examples were
  generated using the public entry point and inspected visually. The former
  independently checked all 15 cluster pairs (nine exact neighbour edges and
  six centroid fallbacks); the latter checked the 50% forest tie rule.

## Limits of these checks

The 111-test result is from Python 3.12.3 on Linux with the versions recorded
by the environment check. Missing optional production libraries were installed
only in a temporary verification directory; the author's working virtual
environment was not changed. An installation in a completely new environment
and every remote download endpoint are not implied by that test result.

The selection check starts from the original catalogue. It does not rerun
the global candidate extraction. The two raster examples verify particular
extractions, not all 10,000 windows or the original source raster as a whole.
External data must be acquired in the documented versions. Source text
extraction for the published bird table remains an explicitly documented
manual external-input step.

The complete large simulation, parameter-domain search, equilibrium-vector
calculation, bootstrap and final 61-figure production sequence have **not**
been rerun from empty data directories for this assembly. Original source
contracts can reveal additional prerequisite/portability issues during that
full run. The public workflow includes the known producer and review sequence,
but its registration alone does not prove successful end-to-end execution.
Historical OSM context-line acquisition has not yet been independently retested.

The final thesis availability paragraph mentions processed data; the author's
publication policy instead distributes the source that generates these data.
The final discussion also contains a preliminary x/b-sensitivity statement
whose original numerical support is not established by the current records.
The verified b = 1 and x = 0.5 production settings are fully specified; the
additional preliminary claim is not silently treated as reproduced.

No test or documentation status should be relabelled as complete production
validation until the corresponding computation has actually passed.
