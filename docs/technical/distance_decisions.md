> Technical calculation record from the thesis. Generated-table links refer to local outputs. Use the public workflow for current execution; earlier pilot replication does not replace the final 8/16 design.

# Analysis 1.1.3: selected distance and graph representations

The comparison reference is the complete matrix of minimum distances between
patch boundary-cell centres. Subsequent analyses use the **depth-1 Voronoi
hybrid matrix** for all-pair metrics, metapopulation capacity, and the
metapopulation model. Mean nearest-neighbour distance takes the minimum over
exact direct Voronoi-neighbour edges. The sample contains 250 complete
trajectories:
10 independent seed blocks for each of 25 scenarios (Only-Random at
$\rho=0$ and two directed methods at twelve positive $\rho$ values). Those
values are derived from geometrically spaced expected cumulative random
removal counts at $H=0.01$, not from a logarithmic grid in $\rho$ itself.
All eight sampled habitat levels $H=0.01$–$0.99$ satisfy the prespecified
maximum 25% patch-count exclusion rule. There are 1,970 comparable
landscapes with at least two patches; 30 of the 250 at $H=0.99$ are excluded.
No real landscapes were used.

The corrected `voronoi_dt` hybrid identifies direct neighbours from raster
Voronoi contacts and calculates their weights with the same exact boundary
method as the complete edge matrix. Non-neighbour pairs retain centroid
distances. The evaluated Voronoi graph has the same neighbour topology and
exact boundary weights copied from the complete edge matrix, isolating the
effect of topology. This graph is the selected source for nearest-neighbour
distance; it is otherwise a diagnostic rather than the common matrix. Its
recorded time is not the standalone cost of building a sparse exact-edge
graph.

| Metric | Selected representation | Evidence |
| --- | --- | --- |
| Mean nearest-neighbour distance | Minimum exact direct Voronoi edge per patch, then mean across patches | The topology-isolating graph, weighted from the complete edge matrix, equals complete-edge NN in all 1,970 valid landscapes. Two regenerated affected landscapes independently confirm the production graph's edge weights. The hybrid-matrix minimum agrees in only 1,957/1,970; a centroid fallback undercuts an exact neighbour edge in the other 13. The largest observed gap is 0.00973 cells or 0.446%. Hybrid-matrix NN remains a diagnostic, not the selected metric. |
| Mean inter-patch distance | All-pair mean of the Voronoi hybrid matrix | A mean over Voronoi graph edges measures a different local property. The hybrid all-pair mean has a maximum $H$-specific median difference magnitude of 0.975%; its largest $H$-specific 95th percentile is 3.74%. At $H=0.99$, the centroid-only all-pair version has 131% median difference magnitude and Spearman 0.218 against the edge reference. |
| Complete-MST $D_{50}$ | MST computed from the Voronoi hybrid matrix | Both the graph and corrected hybrid match the complete edge result in all 1,970 valid landscapes. For $H\geq0.7$, $D_{50}$ is zero in every valid landscape, so rank correlation and percentage difference are undefined despite exact equality. |
| Metapopulation capacity | Voronoi hybrid matrix for every patch pair in the dispersal kernel | The hybrid preserves landscape rankings fairly well (minimum $H$-specific Spearman 0.920 across the three dispersal profiles) but changes capacity magnitudes. At $H=0.3$, its pooled median paired shortfalls are 3.67%, 10.60%, and 19.61% for increasing dispersal half-distance. The 95th percentile among landscapes reaches 80.75% for the longest profile. A graph-only capacity would discard non-neighbour dispersal terms and change the model. |
| Metapopulation equilibrium occupancy | Voronoi hybrid matrix for every patch pair in the dispersal kernel | This uses the same baseline distance definition as capacity. Occupancy was not evaluated in analysis 1.1.3, so no numerical occupancy-equivalence claim follows from this comparison. |

The common hybrid matrix is a methodological choice despite the capacity
magnitude differences above. All-pair uses share the same distance definition,
while the local NN metric uses its Voronoi edges only. The complete exact-edge
matrix remains available only for a possible sensitivity analysis; that analysis is not currently
planned, and a switch in the dispersal distances may require renewed species
parameterisation before occupancy results can be compared. The 1.1.3
comparison tables and figures continue to show the exact-edge reference.

Centroid distances also confound separation with patch size. At $H=0.5$,
the Spearman association between mean patch area and the centroid-minus-edge
gap is 0.712 for nearest-neighbour distance and 0.551 for mean inter-patch
distance. These are descriptive associations within the sampled landscape
ensemble.

The capacity shortfall depends strongly on the deforestation scenario. At
$H=0.5$, the Only-Random median paired shortfall is 25.71%, 38.46%, and
44.93% across the three profiles. For Edge-Random, the largest shortfall
among the twelve sampled $\rho$ values is 8.52%, 15.92%, and 28.06%; for
Access-based deforestation it is 3.41%, 5.86%, and 9.41%. The maximum in
other parts of the sampled domain can be larger: Edge-Random reaches a
68.01% median shortfall for the longest profile at $H=0.3$ and
$\rho\approx0.999878$. Across the 24 directed scenarios at each available
$H$, the Spearman agreement of Only-Random-relative log-capacity contrasts
is at least 0.951, but their magnitudes differ. Sign agreement falls to
21/24 (87.5%) for the longest profile at $H=0.5$. Thus ranking agreement
alone does not establish interchangeable capacity values or scenario
responses.

The profiles use $\alpha=0.172145$, $0.106955$, and $0.066452$ per cell,
from the latest preliminary analysis. Access-based generation uses provisional
$\eta=1$, $k=3$, and $\lambda=0.5$ pending analysis 1.2.1. The exact graph
identities are mathematical properties of the sampled patch geometry;
empirical approximation gaps may change under a different landscape ensemble.

Supporting data: `../tables/01_graph_exactness.csv`,
`../tables/01_hybrid_local_metric_exactness.csv`,
`../tables/01_comparison_by_h.csv`, `../tables/01_comparison_by_rho_h.csv`,
`../tables/01_capacity_shortfall_by_rho_h.csv`,
`../tables/01_capacity_scenario_contrast_preservation.csv`, and
`../tables/01_selected_representations.csv`. Figure definitions are in
`../figures/01_captions.md`.
