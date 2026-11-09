> Technical calculation record from the thesis. Generated-table links refer to local outputs. Use the public workflow for current execution; earlier pilot replication does not replace the final 8/16 design.

# Confirmed-domain sampling protocol

The user authorised overnight sampling and equilibrium calculations on
13 October 2026. We chose 256 points in the monotonically certified inner
response region within the previously justified main rectangle: half-distance
1–512 cells and log10(r)=-2.5 to 5.75. The expanded 1–4096-cell boundary search
was kept as a diagnostic. Its low-r tail is driven by isolated patches at
H=.01 and was not automatically adopted as the main sampling distribution.
The dispersal plateau remains outside the main domain.

The target measure is uniform area in (log half-distance, log r), conditional
on both retention inequalities being certified. Exact stepwise strips describe
the inner region without interpolation. Their total logarithmic area is
82.113, or 69.291% of the original rectangle. We sampled
one independently jittered marginal-area coordinate in each of 64 equal-area
strata, then four independently jittered log-r quartiles conditional on each
dispersal coordinate. The inverse-area mapping weights wide parts of the
region proportionally to their area rather than assigning equal weight to
arbitrary dispersal columns. Every parameter point has design weight 1/256.
The random seed is 2026101302. Conditional stratification shares a kernel
among four ratios while retaining one fixed parameter combination per
landscape trajectory. This is a conditional stratified design, not a
two-dimensional Latin-hypercube design. Its point density is
2.89 times the original 128-point rectangular design.

All 256 points were checked against the same L<.95/U>.05 quantile-envelope
criterion. Unresolved boundary gaps are excluded from this conservative inner
domain and remain documented in the boundary pilot. We did not select points
by bird RMSE or association strength. Earlier LHS points and adaptively
selected boundary queries remain separate; they are not pooled unweighted
into the new main parameter summaries. Thus inference is conditional on
this explicitly defined target region. This selection does not identify
biological species parameters or require a complete 1-to-0 habitat response.

The occupancy run starts with five matched landscape seeds across all seven
scenarios and all 88 habitat levels. Further complete matched seed blocks
are added only after runtime and seed-stability checks, up to 20. Parameter
resolution and landscape replication are distinct; five seeds remain a
preliminary dataset, not evidence of precise intervals in every cell.
