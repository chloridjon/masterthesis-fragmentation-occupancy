> Technical calculation record from the thesis. Generated-table links refer to local outputs. Use the public workflow for current execution; earlier pilot replication does not replace the final 8/16 design.

# Expanded self-colonization comparison and working parameter domain

The comparison now uses 8 complete shared seed blocks, all seven scenarios,
13 stored habitat proportions and the unchanged 65 parameter profiles.
The checked three-seed pilot is retained byte-for-byte. New q=1 responses use
the analytic local self-colonization solver, the same b=1, x=0.5, raw patch
areas, hybrid distances, multi-patch exclusion and residual tolerance 1e-12.
The model source and all original pilot artifacts remain unchanged.

Replication increased first to five seeds. The predeclared diagnostic gate
checked occupancy/contrast changes above 0.05, association changes above 0.1,
or domain omission shifts above 0.25 log10 decades or a changed dispersal
endpoint. A detected change authorised the configured final step to eight.
Stage comparisons and omission checks are saved. These tolerances are screening
conventions, not tests of significance or proof of sample adequacy.

## Working q=1 parameter domain

The sample-conditional working range is **r=0.031622777 to
5623413.3**, with dispersal half-distance **1 to
512 cell-coordinate units**.
The separate regular survey contains 481 fixed parameter combinations:
37 quarter-decade ratio levels from .01 to 10^7 and 13 doubling half-distances
from 1 to 4096. All seven scenarios and H=.025,.2,.4,.6,.825 enter selection.
No scenario-contrast, correlation or bird-fit criterion selects the domain.

The working range retains a central90% scenario-H median transition-bracket
envelope from occupancy .95 to .05, with an outward .25-decade margin. The
upper half-distance is the first plateau onset for which the95th percentile
of shifts in the .50 median crossing is <=.05 log10 decades for at least
two successive doubling steps and all later sampled steps. Both ratio
endpoints were then evaluated at every retained half-distance on every
eligible stored H. At least95% of scenario-H median cells per distance must
reach >=.95 at the lower-r endpoint and <=.05 at the upper-r endpoint. Any
failure expands the corresponding endpoint by .25 decades; the validation
history and exact eligible-cell counts are saved. The plateau diagnostic
remains conditional on the five selection H levels.

Complete-seed omission gives log10-r lower bounds
-1.500–-1.500, upper bounds
6.250–6.250, and dispersal
upper endpoints [512.0].
Sensitivity to plateau tolerances .025/.10, margins .125/.50 and a lower
half-distance of two is saved alongside the main decision. These numerical
bounds describe this landscape ensemble and are not biological species limits.
They define a separate sensitivity domain; no baseline domain or numerical
reference is replaced and no new parameter-space distribution is adopted.

## Results B-D at matched parameters

There are 728 sampled landscape states,
123 common multi-patch exclusions and
39325 valid paired profile-landscape responses.
For the unchanged B reference, the maximum absolute scenario-contrast change
is 0.481666; over sampled H<=.825 it is 0.205157. The high-H results
remain conditional on the fragmented subset of landscapes and are interpreted
with the saved scenario/H exclusion counts. This is mean patch occupancy,
not occupied habitat area or a measured persistence probability.

The median absolute q=0/q=1 change over paired Result-D coefficients is
0.057006;
463 coefficients change sign with both magnitudes
above .1. Equal total scenario weights are applied after paired exclusions.
Pooling scenarios does not control for scenario membership. Missing or
numerically constant cells remain undefined. Full species-parameter
associations and all six metric associations are saved, with stage and seed
omission diagnostics. Large local changes preclude a uniform robustness claim.

Requiring response ranges above0.05 under both models retains
2059 paired coefficients, with median absolute model
change 0.049528 and
288 sign reversals with both magnitudes above0.1.
Rounding occupancy to1e-8 changes 2.076%
of finite comparable coefficients by more than0.05 and makes
0 formerly defined coefficients unavailable.
These numerical/response-range diagnostics accompany interpretation and do
not replace the primary coefficients or establish general metric rankings.

No bootstrap confidence intervals or bird refits were added. The larger seed
sample improves the descriptive comparison but does not establish precision
for every parameter/H cell. B ribbons show5th–95th percentiles across generated
landscapes. They describe landscape variation, not confidence in a median.

## Figure captions

B: Median equilibrium mean patch occupancy and scenario-minus-Only-Random
median contrasts under q=0/q=1, unchanged reference domain131, 8 shared
landscape seeds. Ribbons show5th–95th percentiles across landscapes. Dashed
lines denote Access-based; colours identify rho. At least three valid
multi-patch landscapes are required for each median. H decreases left to right.

C: Marginal Spearman associations of pointwise contrasts with log10-r and
log10-half-distance across the same65 profiles at each H and scenario.
Panels compare q=0/q=1; styles follow B. Profiles are fixed, with no smoothing,
profile resampling or confidence intervals. Undefined cells remain missing.

D: Equal-scenario weighted within-H metric–occupancy rank associations for
the unchanged reference, comparing q=0/q=1. 8 complete shared seeds provide
landscapes; every scenario requires at least three valid paired observations.
Colours identify the six metrics. No confidence intervals are shown.

Range survey: Median q=1 occupancy at H=.4 for half-distances1,16,256, against
r on a log axis, seven scenarios and 8 shared seeds. Styles follow B. Lines
join coarse sampled ratios; no smoothing, extrapolation or CIs are used.

Domain: Central90% scenario-H transition-bracket envelope at each sampled
half-distance, using .95 and .05 occupancy crossings, plus .50-crossing shift
diagnostics. Points are the sampled doubling distances; lines connect them
without fitting a smooth boundary. The retained interval and validated
working ratio endpoints are marked. Envelope width describes variation
between sampled scenario-H response transitions, not a confidence interval.

## Reproduction

From repository root, run final-self-colonization-extension, then
final-self-colonization-extension-review --stage validation,
final-self-colonization-extension-diagnostics --full-audit, and
final-self-colonization-extension-review --stage export.
Both commands use the stable project runner. The original pilot and all
intermediate five-seed tables remain separate from this final stage.

## Final checked endpoint coverage

The seed-omission ratio bounds reported above refer to the initial transition
envelope on the five selection H levels. Validation at all eligible stored H
expanded its upper log10-r endpoint from6.25 to6.75. The final separate q=1
working rectangle is r=0.0316227766–5623413.252 and
half-distance1–512 cell-coordinate units.
At every retained distance,76 scenario-H cells were eligible under the full
eight-seed design: all76 met the lower endpoint target;75/76 met the upper
endpoint target. The remaining cell is saved explicitly and does not become
zero or disappear. Across complete-seed omissions, the minimum coverage was
100.0000% at the lower endpoint and98.6842% at the upper endpoint.
The final coverage rule is retained
at the declared95% threshold. This is sample-conditional coverage of median
responses, not a confidence interval or a biological species limit.

All five latest PDFs were rendered with Poppler and actually viewed. Font
embedding, editable SVG text, panel layout, interval legends, H direction,
parameter-axis transformations and missing cells were checked. The known
Poppler font-container warning has correct actual glyph rendering. Review at
the manuscript's eventual embedded size remains a later editorial task.

## Mechanistic check of the high-H reference response

In the directly checked Only-Random seed1 fixture at H=.925, patch areas are
36999 and1 cells. At the unchanged reference, q=0 patch occupancies are
[0.15121341702550536, 0.8579486296196953], while q=1 gives [0.9998793339563784, 0.9755727714839358].
The dominant patch obtains the largest direct benefit from its self term;
its greater occupancy increases external colonization of the one-cell patch.
The latter cannot persist on its self term alone at r=858.78>1. This corrects
the earlier three-seed pilot report's suggestion that the large mean shift
was caused by local self-persistence of the small isolated patch. Original
pilot artifacts remain immutable; this checked interpretation supersedes that
sentence. Mean patch occupancy weights both patches equally and differs from
occupied habitat area. This fixture does not establish a universal mechanism
for every scenario or H level.
