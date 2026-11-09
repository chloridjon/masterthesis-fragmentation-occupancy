# Detailed computational methods

This document supplies the methods delegated to the accompanying project in
the final thesis. [The promise index](repository_promises.csv) records the
original passages. Configuration files are executable specifications of
numeric grids, tolerances, seeds and output locations. The linked technical
records give the original, more extensive calculations and limitations.

<a id="software-and-landscape-conventions"></a>

## Software and landscape conventions

The implementation is Python. NumPy and SciPy implement numerical operations;
pandas manages tables; rasterio and shapely support empirical raster handling;
Matplotlib generates figures. Pinned dependencies are in `environment/`.
Latin Modern Roman 8 and Latin Modern Math are required for publication
graphics. The source checks their availability instead of substituting fonts.
Linux is the production platform: some orchestration uses `fcntl`, and final
SVG conversion uses fontconfig, librsvg and Cairo.

Generation operates on a 200 × 200 raster with periodic neighbours. Analysis
uses four-connected clusters on the open raster: opposite borders are not
joined. Forest clusters become the habitat patches of the occupancy model.
Simulation coordinates and areas are cell units and cell counts. They have no
fixed physical calibration. The full Forest-cover grid has 88 values; its
exact values are stored in the structural-response configuration.

There are seven fixed scenarios: only random at rho = 0, edge random and access
based at rho = 0.9, 0.99 and 0.999. The first removal nucleates non-forest
randomly. Subsequent events choose random removal with probability 1 − rho
and the prescribed edge mechanism with probability rho. The event controller
records forced initial, random and directed removals. Silent substitution of
one mechanism for another is not allowed. Corresponding replicate IDs use
the same seed across scenarios: base seed 20260924 plus replicate ID minus 1.

<a id="metric-definitions-and-undefined-cases"></a>

## Metric definitions and undefined cases

Edge density counts unlike orthogonal internal cell adjacencies and divides
by total raster-cell count. It does not divide by all possible adjacencies.
Largest cluster fraction is largest-cluster area divided by total forest area;
it is undefined if no forest remains. Susceptibility excludes exactly one
largest cluster, then divides the sum of squared remaining cluster areas by
their sum. It is undefined if fewer than two clusters remain.

MNN averages, over clusters, the minimum exact boundary-cell-centre distance
to a direct Voronoi neighbour. MIP averages pairwise distances over all
unordered cluster pairs. D50 is the smallest distance threshold at which a
connected component contains at least half the remaining forest area. It can
be zero when a single cluster already contains that area. Empty and
single-cluster cases, including differences between D50 and the two mean
distance metrics, are implemented in `landscape_stats.py` and covered by tests.
Undefined quantities remain missing, including constant-input correlations.

<a id="distance-and-graph-selection"></a>

## Distance and graph selection

The preliminary comparison has 25 removal scenarios, ten matched runs and
eight Forest-cover levels: 2,000 states, of which 1,970 are multi-cluster.
It compares centroid and exhaustive exact boundary distances, then the
Voronoi hybrid approximation with exhaustive exact distances. Both actual
values and within-H Spearman rankings are compared. Percentage differences
exclude zero reference values. Forest-cover ranges are descriptive spans,
not confidence intervals.

For the hybrid representation (Voronoi depth 1), exact distances are used for
direct neighbouring regions; all other pairs use centroid distance. MIP,
D50, capacity and occupancy consume the complete hybrid matrix. MNN takes
minima over exact direct-neighbour edges only. A centroid fallback cannot
create a nearest neighbour. Capacity differences do not establish equivalent
occupancy; the thesis did not compare occupancy under both distance matrices.
See [distance methods](technical/distance_methods.md) and
[representation decisions](technical/distance_decisions.md).

<a id="access-parameter-comparison-and-sensitivity"></a>

## Access parameter comparison and sensitivity

The factorial comparison uses eta = 0.5, 1, 2, 4; k = 1, 3, 5;
lambda = 0.1, 0.5, 1; the three focal positive rho values; ten matched runs;
and H = 0.01, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.99. Non-forest rings are
defined by breadth-first accessibility. At k = 1 the lambda values are
identical simulations, not independent replication.

The comparison decomposes variation in setting-level medians into factorial
main effects and interactions. At least eight defined runs per setting and
non-zero total variation are required. Omission diagnostics flag a share
change greater than 0.10, a different leading parameter or an undefined
decomposition. Shares are descriptive and depend on the tested ranges.
The reference is eta = 1, k = 3, lambda = 0.5. The eta = 2 sensitivity holds
k and lambda fixed and counts lower/equal/higher medians at the 24 rho–H
combinations, with metric-specific missingness.
See [access methods](technical/access_methods.md).

<a id="metric-redundancy"></a>

## Metric redundancy

Thirty matched runs per scenario are compared at eight Forest-cover values.
Spearman correlations pool scenarios separately within each H and are also
examined within individual scenarios. An absolute coefficient of 0.8 flags
closer inspection. Retention combines correlations with conceptual meaning
and their dependence on H and scenario. All six metrics remain in the final
set. Bootstrap resampling preserves complete matched runs.
See [redundancy methods](technical/redundancy_methods.md).

<a id="real-landscape-sampling"></a>

## Real-landscape sampling

JRC Global Forest Cover 2020 version 3 supplies external forest maps.
Candidate centres are on an equal-area grid (EPSG:6933) with 64 km spacing.
Centres in water and windows with less than 90% land are excluded using the
land-mask configuration. Windows of 2, 4, 8, 16 and 32 km are resampled to
200 × 200. At least 50% contributing forest makes an output cell forest.
Forest cover is computed after this aggregation.

One hundred candidates are randomly selected per 0.05-wide H bin per extent:
20 bins × 5 extents × 100 = 10,000 landscapes. This is a balanced structural
sample, not a worldwide prevalence estimate. The original procedure used a
low-H catalogue and a subsequent high-H extension; both source modules are
included. Selection seed, candidate availability, exclusions, selected IDs,
source version and raster hashes must be recorded by the local preparation.
The original selection manifest hash is recorded in `external_inputs.json`.
Exact recovery of the same sample must be checked against that identity;
merely drawing another balanced sample is not exact reproduction.

Metrics are recomputed from the selected binary rasters using the same
definitions as the simulation. A cell corresponds to 10–160 m depending on
extent, but comparisons use cell-coordinate distances. Example landscapes
are selected by minimum absolute distance to H = 0.5, with candidate ID as
the tie-breaker. The map uses Natural Earth land outlines.
See [external inputs](data_sources.md) and [coverage methods](technical/coverage_methods.md).

<a id="structural-responses-transitions-and-coverage"></a>

## Structural responses, transitions and coverage

Two hundred complete runs per scenario provide 123,200 recorded states.
For each metric and H, finite values give the median and linear 5th/95th
percentiles. Bands describe landscape variation. Missingness and stability
of quantile endpoints are reported separately.

H50 is the first downward crossing of median LCF below 0.5 along forest loss.
The transition width is H75 − H25. Crossing positions use linear interpolation
between neighbouring sampled levels. Crossings are also calculated for each
run. Bootstrap draws resample 200 whole matched runs, recalculate medians and
crossings, and form central 95% intervals. Initial/batch budgets, maximum draws
and convergence rules are fixed in the configuration. Additional diagnostics
compare cluster-size distributions, nearest-neighbour CDFs and matched
landscape geometries, with equal weight per landscape.

Real-landscape coverage uses simulation 5th–95th percentiles at each H-bin
midpoint, requiring at least 30 valid runs. Endpoints count as covered.
Overall coverage averages over H bins evaluable for every scenario, giving
each such bin equal weight. Union coverage counts a real window once even if
several scenarios cover it. The positive-D50 supplement pools positive-valued
real windows within extent; it has a different denominator. No scenario or
rho is fitted to the empirical data. Full grid resolution and seed-extension
diagnostics are generated by the structural summary and coverage scripts.
See [structural methods](technical/structural_methods.md) and
[coverage methods](technical/coverage_methods.md).

<a id="occupancy-model-and-numerical-checks"></a>

## Occupancy model and numerical checks

The baseline has no self-colonisation (q = 0), area exponent b = 1 and
extinction exponent x = 0.5. Variable parameters are r = e0/c and dispersal
half-distance ell, with alpha = log(2)/ell. Inter-patch weights decay as
exp(−alpha × distance). The full hybrid matrix is used without kernel cutoff.
Landscapes with fewer than two patches are excluded from the baseline domain.
The response is the unweighted mean of equilibrium patch probabilities.

The maximal equilibrium is approached from probabilities of one. Main
production uses tolerance 1e−12; an explicitly declared 1e−13 retry supports
independent comparisons with probability tolerance 2e−7. Iteration budgets,
continuation and independent-check subsets are in the respective configs.
Checks cover [0,1] bounds, original fixed-point residuals, rate balance,
area/distance identities and monotonicity in the two variable parameters.
Row-sum/capacity bounds can certify a unique zero equilibrium; near-zero
iteration alone does not prove extinction. Nonconverged rows are not treated
as valid equilibria. Saved patch vectors support independent rechecking.

<a id="parameter-domain-conditional-sample-and-reference"></a>

## Parameter domain, conditional sample and reference

The initial limit-based pilot derives broad r bounds from capacity and near-full
occupancy. A finer transition survey produces the rectangle
10^−2.5 ≤ r ≤ 10^5.75 and 1 ≤ ell ≤ 512 cells. The original 128-point sample
uses seed 2026100502. Its responses on 20 complete runs per scenario guide a
response-domain screen independently of empirical bird fitting.

For each parameter combination define L as the minimum landscape q05 across
eligible scenario–H cells, and U as the maximum q95. The retained domain
satisfies L < 0.95 and U > 0.05. Coordinate-wise occupancy monotonicity
certifies conservative inner strips; unresolved boundary gaps are excluded.
No interpolated saturation boundary is treated as a certificate.

Seed 2026101302 generates the fixed 256-point conditional sample: 64 jittered
equal-probability strata of marginal logarithmic area, each with four jittered
conditional log-r strata. Inverse cumulative area mapping makes density
uniform in log-area over the retained strips (69.29% of the preliminary
rectangle). Each point has weight 1/256. Sharing dispersal coordinates reuses
kernels and does not make this a rectangular independent LHS. Earlier pilot
and adaptive queries are not pooled into the final target distribution.

Reference eligibility requires each scenario median curve to reach 0.1 and
0.9. The choice balances occupancy span, intermediate-H width and scenario
differences as a response-informed Pareto compromise. It is an author decision
after inspecting the diagnostics, not the unique optimum of a weighted score.
The fixed choice `domain131` and its rationale are retained in
`10_reference_update.json`; reconstruction regenerates its diagnostic basis
and uses that same declared choice. The selected point is approximately
r = 858.78 and ell = 12.95; full precision is obtained from the fixed design.
It is selected before empirical bird comparison. See the records for the
[limit pilot](technical/parameter_limits.md), [boundary refinement](technical/parameter_boundary.md)
and [conditional sampling](technical/parameter_sampling.md).

<a id="occupancy-summaries-and-uncertainty"></a>

## Occupancy summaries and uncertainty

Reference curves use 20 runs per scenario (12,320 states). The 256-point
design uses eight only-random runs and 16 runs in each directed scenario:
104 trajectories, 9,152 states and 2,342,912 parameter–state rows before
domain exclusions. A median requires at least three valid landscapes.
Scenario contrasts subtract the only-random median at the same H. A within-
scenario forest-loss change is median occupancy at H = 0.4 minus H = 0.6.

Reference uncertainty resamples 20 complete matched runs. The unequal design
resamples seeds 1–8 jointly across all seven scenarios and seeds 9–16 jointly
across the six directed scenarios as a separate cohort. The 256 parameter
points remain fixed. Absence of only-random seeds 9–16 is structural absence,
not an ecological exclusion. Parameter correlations are separate Spearman
relationships with scenario contrast at each H; they do not isolate either
parameter independently of the conditional sample dependence.

Integrated contrasts, their H support, weights, missingness and numerical
integration are specified in [integrated contrasts](technical/integrated_contrasts.md).
See also [parameter correlations](technical/parameter_correlations.md),
[H-by-parameter maps](technical/h_parameter_maps.md), and
[interpolation](technical/interpolation.md). Interpolation is linear on
logarithmic parameter axes and is restricted to sampled support; it does not
add independent observations or resample the target distribution.

<a id="empirical-bird-reconstruction-digitisation-and-grouping"></a>

## Empirical bird reconstruction, digitisation and grouping

The empirical source is Morante-Filho, Benchimol and Faria (2021), DOI
[10.1007/s10980-020-01121-6](https://doi.org/10.1007/s10980-020-01121-6).
Table 1 contains rounded occupancy predictions, not detection histories.
All 15 rows classified Forest are included from the 19 published rows.
Species IDs preserve source-name discrepancies without imposing new taxonomy.
Occupancy is presence at a surveyed forest site, while forest cover is measured
within a 600 m radius. These quantities differ from mean model patch occupancy.

For each species, a logit line through published predictions at H = 0.2 and
0.4 reconstructs a logistic curve. The published H where occupancy reaches
0.6 is held out for checking. Rounding envelopes are compatibility diagnostics,
not statistical confidence intervals. Original coefficients and covariance
matrices are unavailable. The common fit grid has 77 levels H = 0.07–0.925.
RMSE compares reconstructed central predictions and each model median curve
with equal weight per H. Best individual scenario and best summed seven-scenario
RMSE are distinct objectives. Empirical curves do not change the parameter
domain or reference choice.

Publisher Figure 2 is digitised from a hash-checked PNG. Pixel-frame bounds,
tick calibration, colour masks, occlusion rectangles and sensitivity masks are
in the digitisation configuration. The central trace checks reconstruction;
grey-band limits only illustrate assumed pointwise 95% intervals. Their
coverage and construction are unverified. The source-review correction for
G. spirurus is implemented in `04_bird_band_source_review.py`; missing limits
and the display treatment are retained explicitly. All fits use central
reconstructed curves, never the illustrative bands.

Descriptive bird grouping uses average-linkage clustering of pairwise whole-
curve RMSEs over the 77 H levels without within-species standardisation.
Silhouettes are checked for two, three and four groups; the three-group cut is
illustrative. Group numbers follow increasing median H at occupancy 0.6.
The examples mf2021_07, mf2021_08 and mf2021_09 represent different response
shapes, not inferred biological classes. For L. vociferans, the additional
near-optimum check uses summed RMSE at most 0.1 above the minimum.

<a id="weighted-metric-occupancy-associations-and-regions"></a>

## Weighted metric–occupancy associations and regions

Associations are calculated separately within H for every fixed parameter
combination. Each of seven scenarios receives probability mass 1/7 after
pairwise exclusion. Observations within scenario share its mass. Both marginal
ranks and their Pearson correlation use these same weights: mid-distribution
ranks assign half of the tied probability mass. Ordinary unweighted ranks
followed by a weighted correlation are a different statistic.

Every scenario must have at least three paired valid observations. Constant
metrics or occupancy range ≤ 1e−10 are undefined. Medians across evaluable H
levels weight those levels equally. Additional maps report median absolute
association, sign consistency and informative fractions. A response-range
screen of 0.05 is a sensitivity analysis, not a redefinition of the main sample.
Bootstrap ranks and scenario weights are recomputed for each matched draw.
Adaptive interval budgets, availability and convergence rules are in config;
insufficient intervals remain missing. Full-seed omission and rounded-rank
sensitivity assess replication and numerical ties separately.

The final A–D regions are exploratory descriptions adopted after inspecting
parameter maps. A: signed median MNN < 0 and ell ≤ 16; B: signed median LCF ≥
0.5; C: above the adopted C lower boundary within defined LCF support;
D: ell > 16, signed median MNN ≤ −0.3, below C. The C boundary follows the
upper B contour at shorter dispersal and the chi ≥ 0.7 contour at ell ≥ 128,
with the coded log interpolation of gaps. Region membership is evaluated at
sampled coordinates; unassigned combinations remain unassigned. Regional
5th–95th percentiles describe variation among equally weighted parameter
combinations, not confidence intervals. See [weighted associations](technical/metric_associations.md)
and `15_regions_ABCD_20261104.json` for the complete rules.

<a id="self-colonisation-and-completed-sensitivity-scope"></a>

## Self-colonisation and completed sensitivity scope

Full local contribution q = 1 is compared with q = 0 at identical parameters
and landscapes: 65 combinations, eight matched runs per scenario and 13 H
levels. This isolates the change in model formulation. A separate response-
domain survey uses capacity/transition checks for q = 1. Its working rectangle
is r = 10^−1.5–10^6.75 and ell = 1–512, with endpoint and seed-omission checks.
The subsequent screen uses 10 dispersal values × 34 r values = 340 combinations
at five H levels; 235 meet the same response-envelope inequalities. These
grid counts are not estimates of continuous domain area. Five-H witnesses
certify retained points, whereas failure at those five levels leaves the
all-H criterion unresolved. No full new 256-point q = 1 sample or new bird fit
is implied. See [eight-run sensitivity](technical/self_colonisation_extension.md)
and [response-domain screen](technical/self_colonisation_domain.md).

<a id="scope-of-reconstruction-and-remaining-checks"></a>

## Scope of reconstruction and remaining checks

Reference PDFs identify what was included in the final manuscript. Rebuilding
them requires the complete producer sequence, including presentation steps.
Original source hashes and new-run provenance are distinct: a newly generated
table must never be accepted merely by copying an old audit receipt.

The final availability paragraph mentions processed data, while the agreed
repository policy distributes their generation code instead. Those processed
data are generated locally. This distinction is documented here without
silently changing the submitted manuscript. The final discussion also mentions
preliminary x/b effects whose original evidential trace is not yet established;
the verified b = 1, x = 0.5 main design does not itself reproduce that claim.
See [validation status](validation.md) for unresolved exact-reconstruction checks.
