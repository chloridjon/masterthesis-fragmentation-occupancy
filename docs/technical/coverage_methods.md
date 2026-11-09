> Technical calculation record from the thesis. Generated-table links refer to local outputs. Use the public workflow for current execution; earlier pilot replication does not replace the final 8/16 design.

# Result A, section 1.3.2: methods and findings

## Design

The question is which structural values in a balanced sample of real forest
windows are represented by the seven *fixed* deforestation scenarios. The
sample comprises 10,000 JRC Global Forest Cover 2020 windows previously
selected on a global equal-area candidate grid with 64 km centre spacing.
Candidate centres in water and windows with less than 90% land were excluded.
The selected sample contains 100 windows in each of 20 habitat-proportion
($H$) bins of width 0.05 for each of five square extents: 2, 4, 8, 16, and
32 km. Each original window was aggregated to a $200\times200$ binary grid
using a 50% forest-majority rule. We used the original selected binary grids
on Alder and did not resample the global source raster in this analysis.

The older offload metric table differs from the final thesis definitions for
several structural metrics. We recalculated Edge Density, Largest Cluster
Fraction, susceptibility $\chi$, Mean Nearest-Neighbour Distance, Mean
Inter-Patch Distance, and Complete-MST $D_{50}$ from all 10,000 binary grids
with the same frozen code used for the simulated landscapes. Structural
metrics use open boundaries, four-neighbour patches, and the selected
depth-1 Voronoi hybrid representation. Nearest-neighbour distance uses exact
direct Voronoi-neighbour edges only. Distances are in cells of each
$200\times200$ grid; the physical cell width varies with real window size.
One-patch landscapes have undefined susceptibility and patch-distance
means, which remain missing.

For each scenario and $H$-bin, exactly one snapshot from each complete
simulation trajectory was selected: the snapshot closest to the bin
midpoint. The prespecified simulation grid contains every midpoint exactly.
Across valid independent trajectories, the 5th and 95th percentiles define
the scenario's central 90% interval. An interval is withheld when fewer
than 30 of the 200 trajectories have a defined metric. A real window is
covered when its metric value lies inside the interval, including the
endpoints. Each bin records covered count, valid real-window denominator,
and valid simulation-trajectory count.

Overall coverage is the unweighted mean of bin-specific coverage fractions.
For a given metric and window size, every scenario uses the same bins:
bins with valid real values and defined intervals for all seven scenarios.
The analogous “any scenario” statistic counts each real window once when
its value lies inside at least one of the seven intervals. It is not the sum
of the seven scenario fractions. All 20 bins contribute for Edge Density,
Largest Cluster Fraction, and Complete-MST $D_{50}$. The highest bin is
excluded for susceptibility and the two patch-distance means because only
9 Only-Random trajectories have more than one patch at $H=0.975$, below the
prespecified minimum of 30.

## Main findings

The fixed scenarios cover different portions of the real structural range.
The table reports the equal-bin mean fraction of valid real windows inside
*at least one* scenario's central 90% interval. The full scenario-specific
fractions and bin counts are retained in the tables.

| Metric | 2 km | 4 km | 8 km | 16 km | 32 km |
|---|---:|---:|---:|---:|---:|
| Edge Density | 0.624 | 0.630 | 0.481 | 0.356 | 0.308 |
| Largest Cluster Fraction | 0.765 | 0.796 | 0.800 | 0.776 | 0.757 |
| Susceptibility $\chi$ | 0.731 | 0.776 | 0.790 | 0.793 | 0.794 |
| Mean Nearest-Neighbour Distance | 0.284 | 0.394 | 0.637 | 0.798 | 0.864 |
| Mean Inter-Patch Distance | 0.621 | 0.769 | 0.882 | 0.922 | 0.938 |
| Complete-MST $D_{50}$ | 0.948 | 0.948 | 0.950 | 0.952 | 0.964 |

Edge Density shows a gap despite the large structural differences among
the simulated scenarios. At 32 km, 1,383 of 2,000 real windows fell outside
all seven scenario intervals. Of these, 1,299 fell *between* the disjoint
intervals rather than below or above the complete scenario range. The
equal-bin “any scenario” coverage declined from 0.624 at 2 km to 0.308 at
32 km. At the bin midpoint $H=0.525$, the observed median Edge Density was
0.033 for 2 km windows and 0.147 for 32 km windows. The balanced sample thus
contains different edge-structure distributions at these two extents even
within the same habitat bin.

Largest Cluster Fraction had “any scenario” coverage between 0.757 and
0.800 across sizes. At 32 km, 410 of 2,000 real windows were below every
scenario's central interval at their respective $H$-bins. This is a
descriptive gap in dominant-patch structure; it does not show that any
particular deforestation process is absent from the real world.

Coverage of the two distance means increased with window extent when
distances were expressed in grid cells. For Mean Nearest-Neighbour Distance,
the “any scenario” fraction rose from 0.284 at 2 km to 0.864 at 32 km.
Among the 1,829 valid 2 km windows in the 19 comparable bins, 1,007 had
values above every scenario interval. Mean Inter-Patch Distance rose from
0.621 to 0.938. Susceptibility coverage was 0.731–0.794, and Complete-MST
$D_{50}$ coverage was 0.948–0.964. Between 64.6% and 82.8% of real windows
had a defined Complete-MST $D_{50}$ of zero, depending on window size; all
seven simulated scenarios also had zero at the highest $H$-bin midpoint.
High coverage for this metric thus partly reflects a shared point mass at
zero. More generally, broad or overlapping simulation intervals can yield
high coverage without a close match between entire distributions.

The headline equal-bin coverage means were stable under the last seed
extension: from 150 to 200 simulation seeds, none of 240
metric–size–coverage summaries (seven scenarios and their union) changed by more than the prespecified
absolute coverage fraction of 0.02. Local bin fractions can be more
sensitive. Among 4,680 comparable bin–metric–size–scenario cells, 31
changed by more than 0.10, with a maximum change of 0.434. The detailed
bin sensitivity table identifies these cases; fine local rankings,
especially for susceptibility, should remain descriptive.

## Scope

The sample is balanced across $H$ and window size; these coverage fractions
are not estimates of global forest-landscape prevalence. Coverage is
calculated separately for each structural metric. It does not imply that
one scenario covers all metrics of the same real window simultaneously.
The seven scenarios were fixed before comparison, and coverage is not used
to select a “best” $\rho$ or infer a real deforestation mechanism.
