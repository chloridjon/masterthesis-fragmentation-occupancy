> Technical calculation record from the thesis. Generated-table links refer to local outputs. Use the public workflow for current execution; earlier pilot replication does not replace the final 8/16 design.

# Result A, section 1.3.1: methods and findings

## Design

Two hundred independent matched seed blocks each generated seven complete
deforestation trajectories on a dimensionless $200\times200$ grid. The fixed
scenarios were Only-Random at $\rho=0$, and Edge-Random and Access-based each
at $\rho=0.9$, $0.99$, and $0.999$. Each trajectory held $\rho$ constant.
Access-based removal used the selected $\eta=1$, $k=3$, and $\lambda=0.5$.
Generation used periodic boundaries. Structural metrics used open boundaries,
four-neighbour habitat patches, and the selected depth-1 Voronoi hybrid
representation. Mean Nearest-Neighbour Distance used exact distances between
direct Voronoi neighbours only.

The 88 habitat levels cover $H=0$ and $H=1$, all 20 midpoints of the 0.05-wide
bins used for real-landscape coverage, every 0.01 step from $H=0.05$ to $0.65$,
and additional levels $0.005$, $0.01$, $0.02$, $0.03$, and $0.04$. The middle
grid resolves the sharp Largest Cluster Fraction transition in the random
reference and its shifts across directed scenarios. At $H=0$, Edge Density is
zero and the five patch-based metrics are undefined. Susceptibility and the
patch-distance means are undefined for one-patch landscapes, including $H=1$.
Undefined values remain missing.

Response tables report medians and 5th–95th percentiles across independent
complete trajectories at each scenario and $H$. These intervals describe
variation among simulated landscapes, not uncertainty in a median. Figures
use original metric scales. Complete-MST $D_{50}$ and the two distance means
are in grid cells; simulated landscapes have no physical distance scale.

For Largest Cluster Fraction, $H_{25}$, $H_{50}$, and $H_{75}$ are linearly
interpolated from the median response curve. Each is the first downward
crossing as habitat is lost from $H=1$. This convention prevents a rebound of
the fraction of *remaining* habitat in the largest patch at very small $H$
from redefining the main transition. Width is $H_{75}-H_{25}$ and remains
missing when either crossing is absent. Confidence intervals are 95%
percentile bootstrap intervals from complete seed-block resampling, using
the same resampled seeds at every $H$. Batches start at 500 draws and grow by
500 to at most 6,000. The bootstrap 2.5th, 50th, and 97.5th quantiles must
change by no more than 0.002 in $H$ for three successive batches. These
intervals condition on the sampled $H$ grid and linear interpolation. A
separate leave-one-seed-out check uses a prespecified 0.02-$H$ tolerance.

## Main findings

The mechanisms created large configuration contrasts at the same habitat
amount. At $H=0.525$, median Edge Density was 0.499 for Only-Random, 0.273
for Edge-Random at $\rho=0.9$, and 0.191 for Access-based at the same $\rho$.
At $\rho=0.999$, the two directed medians were 0.0509 and 0.0284. Median
Largest Cluster Fraction at this $H$ was 0.0259 for Only-Random, 0.490 and
0.898 at $\rho=0.9$, and 0.967 and 0.991 at $\rho=0.999$ (Edge-Random, then
Access-based in each pair). Access-based medians had less habitat–matrix
boundary and a larger dominant patch than Edge-Random at all 86 sampled
interior $H$ levels for each matched positive $\rho$. These are descriptive
properties of the simulated median curves; exhaustive pairwise significance
tests were not performed.

The main connectivity transition moved to lower habitat proportion as the
directed contribution increased. Estimated $H_{50}$ was 0.596 (95% seed-block
bootstrap CI 0.595–0.597) for Only-Random. At $\rho=0.9$ it was 0.526
(0.523–0.530) for Edge-Random and 0.445 (0.440–0.450) for Access-based;
at $\rho=0.99$ it was 0.406 (0.396–0.415) and 0.313 (0.299–0.327).
Edge-Random at $\rho=0.999$ crossed at 0.124 (0.086–0.153). The Access-based
$\rho=0.999$ median did not cross 0.5 over sampled positive $H$, so its
point $H_{50}$ is unavailable. In 17.9% of 2,000 complete-seed bootstrap
curves it did cross. The presence of this threshold therefore remains
sample-sensitive; a confidence interval conditional only on crossing curves
is withheld.

The $H_{75}-H_{25}$ width was 0.0250 (0.0238–0.0259) for Only-Random, 0.0700
(0.0656–0.0745) for Edge-Random at $\rho=0.9$, and 0.0876 (0.0804–0.0922)
for Access-based at that $\rho$. Widths are unavailable for both $\rho=0.999$
directed scenarios because their median curves lack $H_{25}$ crossings.
Only 0.84% of bootstrap curves for Edge-Random at $\rho=0.999$ produced an
$H_{25}$ crossing. The fine $H$ grid changed the Only-Random width estimate
from 0.0341 on the bin-midpoint grid to 0.0250. The bootstrap intervals do
not include uncertainty about response shapes between the sampled levels.

The additional low-$H$ levels reveal a different apparent ordering from the
main random transition. At $H=0.025$, median Largest Cluster Fraction was
0.003 in Only-Random, 0.394 in Edge-Random at $\rho=0.999$, and 0.548 in
Access-based at the same $\rho$. The Access-based median was already 0.638
at $H=0.005$. The denominator is *remaining habitat area*: a high fraction
at very low $H$ does not imply a large absolute patch or population occupancy.

Susceptibility $\chi$ peaked at different sampled $H$ levels across scenarios,
with wide replicate ranges around several peaks. The two inter-patch distance
summaries and Complete-MST $D_{50}$ did not share one stable scenario ordering
across the gradient. Mean Inter-Patch Distance remained near 100 grid cells
over much of the sampled domain even as Edge Density and Largest Cluster
Fraction separated strongly. Complete-MST $D_{50}$ was often exactly zero
in connected directed landscapes and at high $H$; these are defined zeros,
unlike missing metrics at $H=0$. The selected metrics therefore describe
different aspects of spatial configuration rather than one universal
fragmentation score.

Medians stabilised more quickly than the 5th–95th percentile endpoints. From
the first 150 seed blocks to all 200, 0.36% of informative scenario–$H$–metric
cells had a median shift exceeding 10% of the final 5th–95th percentile span;
8.29% had at least one interval endpoint shift exceeding that threshold. The
prespecified 5% frequency limit for endpoint stability was exceeded overall
and for four individual metrics. Local interval boundaries remain descriptive,
especially for Largest Cluster Fraction, susceptibility, and Complete-MST
$D_{50}$. Section 1.3.2 separately tests the sensitivity of the *coverage
summaries* to the number of simulated seed blocks.

The transition, seed-extension, and resolution checks are saved in the
corresponding tables under `../tables/`.

## Scope

These findings describe the generated scenario spectrum. The seven scenarios
were fixed before comparison with real landscapes. Section 1.3.2 reports
descriptive coverage using the JRC 2020 window sample. Neither analysis
validates a real deforestation process.
