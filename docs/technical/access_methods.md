> Technical calculation record from the thesis. Generated-table links refer to local outputs. Use the public workflow for current execution; earlier pilot replication does not replace the final 8/16 design.

# Analysis 1.2.1: selected focal rho rerun

This is the current Access-parameter sensitivity sample. The user-selected
positive focal values are rho=0.9, 0.99, and 0.999 for both Access-based
and Edge-Random analyses. Only Access-based landscapes are generated here;
the three Edge-Random counterparts belong to the later method comparison.
The earlier 0.929/0.995/1 sample is retained separately as provenance.

The full factorial varies eta in {0.5, 1, 2, 4}, k in {1, 3, 5}, and
lambda in {0.1, 0.5, 1}. Ten seed blocks and eight exact habitat fractions
H={0.99, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1, 0.01} are used. Each trajectory
starts from a full 200 × 200 grid. Generation has periodic boundaries,
while four-neighbour patches and metrics use open boundaries. The first
removal is forced random; each later removal is directed with fixed
probability rho. Expected cumulative random removals at H=0.01 are
3960.9, 396.99, and 40.599, respectively.

At k=1, lambda has no mathematical effect. The 36 factorial labels
per rho therefore represent 28 distinct rules. The script generates
840 distinct parameter-by-seed trajectories and expands the exact
k=1 duplicates to 1080 labelled trajectories without counting them
as extra independent runs.

The six structural metrics are edge density, largest-cluster fraction,
susceptibility chi, mean nearest-neighbour distance, mean inter-patch
distance, and complete-MST D50. The selected depth-1 Voronoi hybrid
provides exact edge-to-edge weights for direct Voronoi neighbours and
centroid fallback distances otherwise. Nearest-neighbour distance uses
only exact direct Voronoi edges. Chi excludes exactly one largest patch.
Chi and the two distance means are undefined for fewer than two
patches. D50 is zero when a single patch already contains at least
half of total habitat area.

For each rho, H, parameter cell, and metric, the table reports the
median, mean, and 5th–95th percentiles among ten simulated landscapes,
with the valid count. Factorial main and interaction shares decompose
variation among the 36 cell medians. The figures display a share only
if every parameter cell has at least eight finite values; undefined
or zero-variation shares are omitted. Shares are descriptive, not
significance tests. No real landscapes or metapopulation capacity are
included.

Reproduce from the repository root using the existing virtual environment:

    MPLCONFIGDIR=/tmp/mpl-access-focal NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python Masterarbeit_final/code/analyses/01_landscape/01_02_preliminary_analyses/01_02_01_access_parameter_selection/02_access_parameter_selection.py --config Masterarbeit_final/config/01_landscape/01_02_preliminary_analyses/01_02_01_access_parameter_selection/03_focal_rho_090_099_0999.json --processed-dir Masterarbeit_final/data/processed/01_landscape/01_02_preliminary_analyses/01_02_01_access_parameter_selection/focal_rho_090_099_0999 --results-dir Masterarbeit_final/results/01_landscape/01_02_preliminary_analyses/01_02_01_access_parameter_selection/focal_rho_090_099_0999 --workers 4

Each trajectory is saved atomically with seed, grid hash, event counts,
and a producer fingerprint. Completed trajectories are validated and
reused on restart.

## Summary update: independent seeds and default run

The default entry point now selects 03_focal_rho_090_099_0999.json and the
matching focal data and results directories. Summary settings are stored in
04_summary_stability.json, separately from the data-producing configuration;
changing a summary diagnostic does not invalidate the trajectory cache.

The ten independent seed blocks were each omitted once, consistently across
all parameter cells and habitat snapshots. The entire median-based factorial
decomposition was recomputed after each omission. A maximum main-effect-share
change above 0.10, a change in the factor with the largest main-effect share,
or an undefined omission flags sensitivity. The diagnostic is conditional on
the observed seeds. The previous eight-finite-value rule remains a missingness
screen and must not be described as a sample-size or precision guarantee.
Sensitive decompositions remain explicitly marked descriptive results.

All 24 edge-density decompositions pass the influence criteria; 23 of 24
largest-cluster-fraction decompositions pass. The latter is sensitive at
rho=0.999 and H=0.5, where a nearly saturated response makes the relative
factor shares unstable. Five susceptibility, six nearest-neighbour,
three inter-patch-distance, and five D50 decompositions pass. Eight displayed
points in total are sensitive, and 70 remain unavailable under the existing
support/definedness rule. At least one additional seed would be required to
study stability as the sample grows; leave-one-out diagnostics alone cannot
establish that new trajectories would preserve every numerical estimate.
