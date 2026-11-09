> Technical calculation record from the thesis. Generated-table links refer to local outputs. Use the public workflow for current execution; earlier pilot replication does not replace the final 8/16 design.

# Analysis 1.1.3: methods and interpretation rules

## Generated sample

All landscapes are newly generated from complete 200 × 200-cell forest grids
with the active single-cell sequential deforestation generator. One forced
random removal starts each trajectory. The Only-Random reference has
$\rho=0$. Thirteen targets for the expected cumulative number of random
removals are geometrically spaced from 1 to 39,600 at the final snapshot
$H=0.01$. The first twelve targets give fixed positive $\rho$ values for both
Edge-Random and Access-based trajectories; the last is the Only-Random
reference. With $N_{\rm removed}=39{,}600$, a target count $R$ maps to
$\rho=1-(R-1)/(N_{\rm removed}-1)$. The twelve positive values range from
approximately 0.58615 to 1. Each
trajectory supplies snapshots at the eight habitat proportions listed in
`../../../../../config/01_landscape/01_01_analysis_framework/01_01_03_distance_graph_selection/01_distance_graph_selection.json`. Only snapshots with at least
two four-neighbour habitat patches enter metric comparisons. Generation uses
periodic boundaries and metric calculation uses open boundaries, matching the
most recent preliminary landscape analysis. Simulated cell distances have no
physical unit. The exact endpoints $H=0$ and $H=1$ have fewer than two patches
and are outside the defined comparison domain; the observed gradient runs from
$H=0.01$ to $H=0.99$.

The ten seed blocks are the independent resampling units. All comparisons are
paired on exactly the same landscape grid, and bootstrap resamples copy whole
seed blocks including all available methods and $\rho$ values. Exclusions and
bootstrap intervals are recalculated for each metric and habitat proportion.
Figures and representation decisions include $H$ levels where no more than
25% of generated landscapes have fewer than two patches. All eight sampled
levels qualify: 250 landscapes are valid at each level through $H=0.9$, and
220/250 are valid at $H=0.99$. However, all ten Only-Random landscapes at
$H=0.99$ have fewer than two patches, so Only-Random-relative capacity
scenario contrasts are unavailable there. The decision diagnostics also
require a defined Spearman correlation. For $D_{50}$, this leaves $H=0.01$
through $H=0.5$: all valid landscapes have $D_{50}=0$ at $H\geq0.7$, so
rankings and percentage differences are undefined there even when the
representations agree exactly.

## Distance and graph representations

- `centroid_complete`: Euclidean distances between patch centroids for every
  pair.
- `edge_complete`: minimum distance between boundary-cell centres for every
  pair. The exact accelerated calculation was checked against the existing
  `edge_min` implementation on several small generated landscapes.
- `voronoi_hybrid`: raster Voronoi contacts identify direct neighbours;
  their weights are then recalculated with the same exact edge-to-edge method
  as the complete matrix. Other pairs retain the centroid-distance fallback.
  This is `voronoi_dt` mode with depth 1.
- `voronoi_graph`: only direct Voronoi neighbour edges, **reweighted by exact**
  edge-to-edge distances taken from the complete edge matrix. This separates
  the information retained by graph topology from errors in raster contact
  weights. It is compared for mean nearest-neighbour distance
  and area-weighted MST $D_{50}$. Its stored mean edge length is a local
  neighbour statistic and is not compared as an estimator of all-pair mean
  inter-patch distance. A graph-only capacity would remove long-range
  dispersal terms and thereby change the model, so no such capacity is
  calculated.

For the selected Mean Nearest-Neighbour Distance, take each patch's minimum
over the exact weighted edges of `voronoi_graph`, then average across patches.
The `voronoi_hybrid` NN column in the saved comparisons instead takes the
minimum over every entry of the hybrid matrix. It remains a diagnostic of
the matrix fallback: a non-neighbour centroid distance can be shorter than
a direct edge. The selected NN is undefined for fewer than two patches.

A two-patch diagnostic guards against the former contact-weighting error:
diagonally adjacent single-cell patches have exact boundary-cell centre
distance $\sqrt{2}\approx1.414$, whereas the four-neighbour raster contact
estimate is 2. The corrected hybrid matrix and its Voronoi graph both store
$\sqrt{2}$. The separate graph representation in this analysis is still
reweighted from the complete matrix to isolate graph topology; its recorded
construction time is therefore not a standalone cost estimate for a sparse
exact-edge implementation.

The saved `01_computation_cost.csv` records timings from this run with the
current accelerated implementation. It extracts patch boundaries once per
raster and calculates selected exact boundary distances in a Numba kernel.
Timings are descriptive for this machine and workload; the Voronoi graph uses
weights copied from the complete edge matrix, so its recorded construction
time is not a standalone sparse-algorithm benchmark.

Capacity uses $M_{ij}=A_i^{0.5}\exp(-\alpha d_{ij})A_j$ for $i\ne j$ and a
zero diagonal, with the three $\alpha$ values documented in
`../../../../../references/notes/01_alpha_profile_source.md`. Kernel weights below
$10^{-4}$ are omitted, consistent with the latest preliminary calculations.
The symmetric eigenvalue iteration has relative convergence tolerance
$10^{-7}$ and a 3,000-iteration cap. Nonconverged values are missing and their
counts are logged. On one full edge-distance random landscape at $H=0.3$,
the relative capacity change from this kernel cutoff was 0.0087–0.0118%
across the three $\alpha$ values; this is a diagnostic, not a global error
bound.

## Comparison statistics

The complete edge representation is the reference. For each candidate and
metric, `difference = candidate - reference`, and percentage difference
magnitude is $100|candidate-reference|/|reference|$ when the reference is
nonzero. The
summary reports Spearman landscape-ranking correlation, median difference,
median candidate-to-reference ratio, 5th–95th percentile difference and
percentage difference magnitude, and a
Spearman association between the signed difference and mean or largest patch
area.
These statistics are reported by $H$, by method and $H$, and by method,
$\rho$, and $H$. Bootstrap 95% confidence intervals for pooled $H$-specific
Spearman correlations and median differences use adaptive whole-seed percentile
resampling, configured separately in 01_summary_stability.json. The algorithm
starts with 500 draws, adds batches of 500, and stops after three consecutive
stable comparisons or 10,000 draws. For both 2.5th and 97.5th percentiles and
the bootstrap median, changes must be at most 0.005 in correlation units and
0.01 of the reference scale for the signed difference. The scale is the maximum
of the median and 95th percentile of absolute reference values and 1e-12.
Undefined correlations on constant metrics remain missing; difference
intervals can still converge. The batch log retains every convergence check.
Unconverged intervals are retained as raw bounds in the table but excluded
from the plotted CI columns.
No correlation is reported when fewer than four finite landscapes or fewer
than two distinct values are available.

For hybrid versus complete-edge capacity, a separate descriptive summary
compares each positive-$\rho$ scenario's median capacity with the Only-Random
median at the same $H$ and dispersal profile. Log scenario contrasts are
$\log(\tilde\lambda_s/\tilde\lambda_0)$, where the tildes are scenario
medians and subscript 0 denotes Only-Random. Their Spearman
agreement, sign agreement, and median absolute gap quantify preservation of
relative scenario responses. These statistics cover 24 directed scenarios
at each $H$ with a valid Only-Random baseline; none can be formed at
$H=0.99$. They have no additional bootstrap interval and are descriptive.

Results of this preliminary method comparison are conditional on
$\eta=1$, $k=3$, and $\lambda=0.5$ for Access-based deforestation. Analysis 1.2.1 retained these
reference access parameters.

The selected representation for subsequent all-pair distance metrics,
capacity, and metapopulation runs is the depth-1 `voronoi_hybrid` matrix. The
all-pair mean uses every entry of this matrix, and the metapopulation dispersal
kernel retains every patch pair. Mean nearest-neighbour distance uses only
the exact direct Voronoi edges as specified above. The complete edge
matrix remains the benchmark in the present comparison, not the future
baseline. A full-edge sensitivity analysis is optional and not currently
planned; switching distances for occupancy would require reconsidering the
species parameterisation. This choice does not imply that the hybrid and
complete-edge capacity values are numerically equivalent.

## Reproduction

From the repository root, using its existing local virtual environment:

```sh
NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  Masterarbeit_final/code/analyses/01_landscape/01_01_analysis_framework/01_01_03_distance_graph_selection/01_distance_graph_selection.py --workers 4
```

The command skips completed trajectory CSV files and checks the saved
configuration before continuing. To regenerate only tables and figures after
all trajectories exist, add `--summarize-only`. `--config`, `--processed-dir`,
and `--results-dir` permit explicit input and output locations. The script
never writes to `data/raw`.

## Output files

- `../../../../../data/processed/01_landscape/01_01_analysis_framework/01_01_03_distance_graph_selection/run_config.json`:
  complete configuration including generated $\rho$ and $\alpha$ values.
- `../../../../../data/processed/01_landscape/01_01_analysis_framework/01_01_03_distance_graph_selection/trajectories/*.csv`:
  resumable trajectory-level metric data with landscape hashes and seeds.
- `../../../../../data/processed/01_landscape/01_01_analysis_framework/01_01_03_distance_graph_selection/landscape_metrics.csv`:
  combined machine-readable metric table.
- `../../../../../data/processed/01_landscape/01_01_analysis_framework/01_01_03_distance_graph_selection/paired_comparisons.csv`:
  paired landscape comparisons underlying all summaries and figures.
- `../tables/01_comparison_by_h.csv`, `01_comparison_by_method_h.csv`, and
  `01_comparison_by_rho_h.csv`: summary statistics.
- `../tables/01_capacity_scenario_medians.csv` and
  `01_capacity_scenario_contrast_preservation.csv`: capacity levels and
  Only-Random-relative scenario contrasts for hybrid versus complete-edge
  distances.
- `../tables/01_decision_diagnostics.csv`: across-$H$ diagnostic summary
  restricted to levels with at most 25% patch-count exclusions and a defined
  landscape-ranking correlation.
- `../tables/01_exclusions_by_scenario_h.csv`: generated and excluded
  landscape counts.
- `../tables/01_graph_exactness.csv`: number of landscapes with exact
  agreement between the neighbour graph and complete edge representation.
- `../tables/01_hybrid_local_metric_exactness.csv`: exact agreement counts and
  worst remaining gaps for the corrected hybrid nearest-neighbour and
  $D_{50}$ metrics.
- `../tables/01_capacity_shortfall_by_rho_h.csv`: paired percentage capacity
  shortfall by method, $\rho$, $H$, and dispersal profile, with the median
  and 5th–95th percentile range among landscapes.
- `../tables/01_selected_representations.csv` and
  `01_representation_decisions.md`: explicit metric-specific selections and
  their numerical justification.
- `../figures/01_*.svg` and matching PDFs: figures.

## Seed influence and Monte Carlo stability

All ten seeds were omitted in turn, keeping each remaining seed's scenarios
and representations together. A correlation change above 0.05 or a signed
median-difference change above 0.10 of the fixed reference scale flags seed
sensitivity. These thresholds were specified before computing this update.
They describe influence within the observed sample; they do not establish
population precision or justify interpreting 250 scenario trajectories as
250 independent seeds. Sensitive estimates remain descriptive diagnostics.
Increasing the number of bootstrap draws cannot supply missing independent
landscapes.

Of 112 comparison groups, 111 reached the stopping criterion. The remaining
group is centroid nearest-neighbour distance at H=0.99. All 48 hybrid groups
and all 16 exact-neighbour graph groups passed the seed-influence thresholds;
11 centroid groups did not. These results support retaining the previously
chosen distance definitions, conditional on the sampled scenarios and seeds.
