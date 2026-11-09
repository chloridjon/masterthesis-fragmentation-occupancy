> Technical calculation record from the thesis. Generated-table links refer to local outputs. Use the public workflow for current execution; earlier pilot replication does not replace the final 8/16 design.

# Limit-based species-parameter preliminary analysis

## Status after the selection decision on 2026-10-05

This record describes the completed broad diagnostic survey. Its rectangles,
profiles and Latin hypercubes are not adopted for Results B-D. The current
main-domain strategy focuses on the $q=0$ diagonal occupancy transition and
excludes the large-half-distance plateau. Final bounds require a finer pilot.
See [the subsequent selection decision](../09_transition_domain_decision.md)
for the rationale, the status of outstanding numerical choices and the later
matched-parameter self-colonization comparison. The procedures below describe
the survey that produced the existing numerical outputs.

## Completed survey methods

The revised analysis examines possible model responses systematically. The
shared dispersal half-distance interval is 0.1767817107–19409.43799 cell-coordinate
units, derived from kernel weights of 0.01 at the shortest positive distance
and 0.99 at the largest distance. The intervals approximate kernel limits;
they do not imply a one-percent error bound on occupancy. No off-diagonal
zero distances were found in this ensemble.

| Self-colonization q | Lower r | Upper r |
|---|---:|---:|
| 0 | 5.15844896429e-72 | 8.73674235883e+5 |
| 1 | 1.02942647688e-2 | 7.77860353883e+6 |

Each lower ratio is the largest numerically bracketed value achieving mean
patch occupancy at least 0.99 in every valid pilot landscape at the shortest
half-distance. Feasible endpoints are retained, with a logarithmic bracket
width at most 1e-05. The upper endpoint is
1.01 times the largest eigenvalue upper bracket at the longest half-distance.
The one-percent upper margin protects the strict extinction-boundary inequality;
it is distinct from both the kernel tolerance and the occupancy target.

Five complete matched seed blocks generated seven focal scenarios and ten H
levels. All 350 rasters were replayed and compared with their Result A hashes
and recomputed patch counts. The same 74 zero- or one-patch landscapes are
excluded for q=0 and q=1. The rectangular pilot uses 25 logarithmic ratios and
17 logarithmic half-distances per formulation. It retains saturated responses.
No marginal intermediate-response filter or scenario-effect optimization was
used. Profiles occupy the specified quarter and midpoint positions. Separate
independent 128-point logarithmic Latin hypercubes were generated after the
boundary calculation; their occupancy responses have not been evaluated.

Areas remain raw habitat-cell counts and distances have no physical calibration.
The full depth-1 Voronoi hybrid matrix is retained. The kernel is unnormalised
and untruncated, with b=1 and x=0.5. Self-colonization adds q A_i^1.5 to the
capacity-matrix diagonal. At short dispersal the self term remains active.
The same half-distance interval is shared, but r endpoints are independently
determined. Equal relative positions in the two rectangles are not identical
species parameters; their comparison addresses conclusions across model
formulations rather than the isolated effect of changing q at fixed r.

Numerical equilibria were iterated from full occupancy. Algebraic row scaling
uses B_ij=M_ij/Q_i and retains log(Q_i), where Q_i=sum_j M_ij. The map is
(Bp)_i/((Bp)_i+r/Q_i), exactly the original unnormalised model in exact arithmetic.
This prevents entire weak-source rows from disappearing through underflow.
It does not replace the model with a row-normalised dispersal kernel. Ratios
are stored as log(r) and scientific strings, with float values only where
representable. Relative weights below machine range may still round to zero;
absolute probability bounds, residuals and independent comparisons are audited.
All probability vectors and row offsets are retained. Nonconverged rows remain
missing in summaries. 0 response rows remained nonconverged.

The domain is conditional on the pilot ensemble and its extremes. Leave-one-
seed-out lower-bound changes at fixed dispersal endpoints reached
20.368 logarithmic decades. Separate geometry diagnostics show how
omitting seeds changes the dispersal extrema; the fixed-dispersal boundary
checks do not recompute all bounds under those altered extrema. These checks
are not a proof that five seeds capture every possible landscape. Endpoint
coverage must be rechecked when adding landscapes. The wide log-r interval
may allocate many equally spaced pilot or LHS points to saturated responses;
category fractions and profile responses are retained rather than selecting
a different domain to make scenario effects larger.

Transition tables report adjacent-level linear crossings in H or log(r),
without connecting missing values. Multiple crossings are counted. The broad
pilot grid may jump across a transition without sampling any intermediate
response, particularly for q=0. Interpolated widths are descriptive grid
diagnostics, not resolved ecological thresholds. Mixed finite differences in
log(r) and log(half-distance) describe interactions on the sampled grid.

Reproduce from the repository root:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/ufz-limit-mpl .venv/bin/python -m src.tools.codex_run final-species-parameter-limits --workers 4
```

The configuration, geometry and numerical-source fingerprints, kernel extrema,
individual lower-bound brackets and capacity brackets are saved with the run.
Original and expanded historical pilot outputs are preserved separately.
