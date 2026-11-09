> Technical calculation record from the thesis. Generated-table links refer to local outputs. Use the public workflow for current execution; earlier pilot replication does not replace the final 8/16 design.

# Completed first boundary refinement

The expanded pilot classified 83 additional parameter queries at
half-distances 1–4096 cells, with log10(r) endpoints [-44.000000000000014, 7.0].
The same seven scenarios, 20 matched seed blocks and 601 eligible scenario–H
cells were retained. Both rate endpoints were certified; the original 128-point
LHS and its response dataset remain unchanged. 4274
query/cell combinations required direct quantiles, based on 85466
valid patch-probability vectors and 14 exclusions.
Other cells were bounded by verified entrywise kernel/r comparisons; their
complete occupancy curves were not simulated or interpolated.

The prespecified four bisection rounds resolved 18
of 26 vertical brackets to the 0.25-decade target. 8
remain wider, up to 2.479 decades. Rate coverage, vertical bracket
precision and precision between dispersal anchors are separate issues.
Hatching in the figure includes uncertainty between anchors; it is not a
bootstrap confidence interval. A long-distance tail remains an exploratory
diagnostic, not evidence that arbitrarily large dispersal changes occupancy.

The independent audit reloaded all direct vectors and checked means and
original-model residuals (maximum 9.999e-11),
grid and geometry identities, quantiles, conservative interval certificates
and boundary witnesses. No bird RMSE or correlation-based selection was used.
These queries are not a random/LHS sample and do not increase the main
inferential sample size. A denser main-domain sample still requires its own
target-density decision.

## Methods paragraph for later manuscript integration

We refined the boundaries of the parameter region excluded by nearly uniform
low or high equilibrium occupancy. The search covered dispersal half-distances
from one to 4096 raster cells. At each of 13 doubling half-distances, we
queried geometric midpoints of rate brackets delimiting globally near-one
and near-zero responses. Eligibility was defined using the minimum landscape
5th percentile and maximum 95th percentile across seven scenarios and eligible
habitat levels, retaining profiles whose two summaries were below 0.95 and
above 0.05, respectively. We used the same 20 matched landscape seeds as in
the initial Latin-hypercube survey. Conservative entrywise comparisons of
the dispersal kernel divided by the extinction-to-colonization ratio supplied
bounds on occupancy and landscape quantiles. Where these bounds did not
determine a profile's classification, we recomputed the quantiles from all
matched seeds in the required scenario–habitat cells. The first refinement
used at most four rounds and a target bracket width of 0.25 logarithmic
decades; wider remaining brackets were reported. The adaptive boundary points
were kept separate from the main parameter sample.

## Figure caption

Expanded boundary search for the q=0 model. (a) Full search window.
(b) Detail spanning log10(r)=-3.5 to 6.5, resolving the
main sampling window. The rare low-rate tail remains visible in panel (a).
Circles show the original 128 LHS profiles;
diamonds mark 83 additional query certificates, which may use directly
evaluated quantiles, verified monotone bounds or universal endpoint bounds.
Blue denotes globally near-one response, orange globally near-zero response,
and green profiles retained under the L<.95/U>.05 rule. Classification uses
landscape 5th–95th percentiles, seven scenarios and 20 matched seeds; only cells
with at least three valid multi-patch landscapes are eligible. Pale areas
show coordinate-order implications of these certificates; hatched gaps
remain unresolved and are not confidence intervals. The outlined rectangle
is the original main search window. Both axes are logarithmic; half-distance
is measured in raster-cell coordinates and is not calibrated to physical units.
Standalone numerical/export review is complete; manuscript-size review and
final sampling-domain adoption remain pending.
