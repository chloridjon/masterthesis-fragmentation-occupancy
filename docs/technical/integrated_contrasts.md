> Technical calculation record from the thesis. Generated-table links refer to local outputs. Use the public workflow for current execution; earlier pilot replication does not replace the final 8/16 design.

# Supplementary integrated occupancy contrasts

This author-requested descriptive heatmap supplements the pointwise contrasts in Results B/C. It does not replace their primary role in the analysis plan. No new simulations, parameter selection or confidence intervals were calculated.

For each of 256 retained parameter profiles and six directed scenarios, we integrated the difference between the scenario median equilibrium patch occupancy and the Only-Random median:

$$A_s(\theta)=\int_{0.005}^{0.925} [m_s(H,\theta)-m_0(H,\theta)]\,dH.$$

Here $m_s$ denotes the median across valid simulated landscapes, using 16 planned seeds per directed scenario and 8 for Only-Random. This is the difference of areas under median curves, not a median of paired seed-level areas. The trapezoidal rule used actual distances between the 85 sampled H values on the common contiguous domain. No equal-weight average over the irregular H grid was substituted. No endpoint extrapolation or internal missing-value bridge was used. H=0 and H=1 have undefined responses, and Only-Random at H=.975 has only 2/8 valid landscapes; these levels were excluded for every scenario/profile. At H=.925 the baseline has 5/8 valid landscapes, limiting high-H interpretation. Missing landscapes were never assigned zero occupancy.

Positive values indicate a larger area for the scenario, negative values a smaller area, in occupancy times habitat-proportion units. Dividing by the domain width .92 would give the mean occupancy difference across this H interval; the figure shows the unnormalised area difference requested. Both parameters and H are dimensionless, with dispersal distance expressed in grid cells. Positive and negative contributions can cancel, so a small net area does not imply similar curves. The saved table separates positive and negative areas, splitting linear segments at their zero crossings.

Surfaces use piecewise-linear Delaunay interpolation in unit-normalised log10 parameter coordinates, with no extrapolation beyond the sampled convex hull or the adopted conservative stepwise domain. All panels share the same diverging scale, -0.4 to +0.4, with neutral zero; red means positive and blue negative. Grey marks unsupported parameter locations. The diagnostic export also shows all 256 sampled locations. Interpolation patterns do not add parameter resolution or establish biological archetypes.

Independent scalar integration of saved-row CSV medians agreed with array-based calculations to 5.55e-16. Whole-seed omissions preserved the shared cohort 1-8 and directed-only cohort 9-16; the maximum absolute area-contrast change was 0.009901. Comparing 8 matched seeds with all 8/16 seeds changed a sampled area contrast by at most 0.025625. These descriptive checks do not establish confidence intervals or stability of every cell.

The sampled net areas range from -0.143309 (access_weighted_rho0999, domain012) to 0.369587 (access_weighted_rho0999, domain059). These extrema were selected after viewing the full design and are descriptive.

## Figure caption

Signed differences in areas under median equilibrium patch-occupancy curves across the species-parameter sample. Panels compare Edge-Random (left) and Access-based (right) with Only-Random, for rho=.9, .99 and .999 from top to bottom. Each sampled parameter profile uses 16 landscape seeds for the directed scenario and 8 for Only-Random, with at least three valid multi-patch landscapes required at every retained H. Areas cover the same 85 habitat proportions from .005 to .925 and use trapezoidal integration with the observed H spacing. Red indicates greater integrated occupancy than Only-Random, blue lower occupancy and neutral colours small net differences. Opposing pointwise differences may cancel. Grey areas lie outside the supported parameter domain; the diagnostic version marks sampled profiles with open circles. Piecewise-linear interpolation uses normalised log10 coordinates, restricted to the sampled convex hull and adopted inner domain. No uncertainty interval or significance threshold is encoded. This supplementary summary complements pointwise scenario contrasts at constant H.

## Reproduction

`.venv/bin/python -m src.tools.codex_run final-integrated-occupancy-contrast`

After rendering and inspecting both PDFs, confirm with `--confirm-visual-review`. Numerical producers and original checked outputs remain unchanged.
