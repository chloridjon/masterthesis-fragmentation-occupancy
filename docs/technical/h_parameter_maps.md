> Technical calculation record from the thesis. Generated-table links refer to local outputs. Use the public workflow for current execution; earlier pilot replication does not replace the final 8/16 design.

# Habitat proportion and species-parameter contrast maps

These author-requested descriptive maps use the same audited256-profile conditional log-area design and8 Only-Random/16 directed-scenario seeds. No new simulations, bootstrap intervals or biological calibration were performed. First, for each profile and habitat proportion, we calculated the difference of landscape medians, directed scenario minus Only-Random. We then calculated the equal-design-weight arithmetic mean of these contrasts across profiles in each of16 fixed equal-width logarithmic parameter classes. This is a mean of profile-specific median differences, not a pooled landscape median or an effect at a fixed value of the omitted parameter.

For the r map, bin edges span log10(r)=-2.5 to5.75. For the dispersal map, bin edges span half-distance1 to512 cells. Every profile has weight1/256 in the original design; weights are renormalised within each class. Every one of the256 profiles belongs to one bin in each figure, with no curve-shape or contrast-based selection. Bin counts range4-31 for r and12-24 for half-distance; counts and memberships are saved. Four profiles in the sparsest r class provide limited parameter coverage. Shared dispersal groups are numerical design locations, not independent landscape replicates.

The other parameter varies over the portion of the adopted nonrectangular domain represented by the profiles in each class. Its conditional distribution therefore changes with the displayed parameter. Patterns describe the chosen joint design and cannot isolate the causal effect of changing r or dispersal while holding the other fixed. Averaging may also hide opposing profile-level contrasts.

All panels use the same85 evaluable H levels, .005-.925, requiring at least3 valid landscapes per scenario/H/profile before aggregation. The baseline has5/8 valid landscapes at H=.925; H=.975 has only2/8 and is withheld. H=0 and H=1 have undefined responses. Exclusions are not zero occupancy. Grey regions mark H outside the common domain. H decreases from1 to0. Display cells are centred on sampled H, bounded by adjacent midpoints and clipped to .005-.925; H spacing is not treated as uniform. Parameter classes are displayed at their exact logarithmic edges. No interpolation or smoothing is applied.

Both figures use a common signed colour scale -1 to+1 in occupancy units, with red positive, blue negative and neutral zero. The statistic remains pointwise in H; no integration is performed here. These colours encode no confidence intervals or significance test.

Independent scalar means reconstructed every plotted cell with maximum error2.22e-16; profile-count-weighted class means recover the overall mean with error5.55e-17. The CSV contrasts were separately checked against all source median-array differences, and landscape medians were recomputed from the audited saved response array. Cohort-aware whole-seed omissions preserve shared seeds1-8 and directed-only seeds9-16. Maximum omission changes are0.103591 for the r map and0.047276 for the dispersal map. Matched8-to-8/16 maximum changes are0.131723 and0.084673, respectively. These diagnostics do not establish universal stability or inferential coverage.

## Figure captions

**Extinction-to-colonization ratio.** Pointwise occupancy contrasts against Only-Random across habitat proportion and16 logarithmic classes of the ratio r. Each cell is the arithmetic mean across the sampled parameter profiles in that class of the difference between scenario and Only-Random median equilibrium patch occupancy. Profile medians use16 seeds for directed scenarios and8 for Only-Random, restricted to valid multi-patch landscapes. The other parameter, dispersal half-distance, varies among the sampled profiles within each class. Panels compare Edge-Random (left) and Access-based (right) at rho=.9, .99 and .999 (top to bottom). Red indicates higher and blue lower occupancy than Only-Random; grey marks the unevaluable H domain. The common colour scale is -1 to+1 occupancy units. The85 included H levels span .005-.925, shown in the direction of habitat loss; no smoothing or confidence interval is encoded. Counts range4-31 profiles per parameter class. Patterns are conditional on the adopted joint parameter distribution.

**Dispersal half-distance.** As above, with16 logarithmic classes of dispersal half-distance1-512 cells on the y-axis. The extinction-to-colonization ratio varies among the profiles represented within each class. Classes contain12-24 profiles; no value of r is held fixed. All remaining definitions, scenario panels and colour limits are identical.

## Reproduction

`.venv/bin/python -m src.tools.codex_run final-habitat-parameter-contrast`

After both PDFs are rendered and actually viewed, confirm with `--confirm-visual-review`.
