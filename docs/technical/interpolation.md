> Technical calculation record from the thesis. Generated-table links refer to local outputs. Use the public workflow for current execution; earlier pilot replication does not replace the final 8/16 design.

# Interpolated habitat-by-parameter contrast maps

These figures display the unchanged conditional means from the checked16-class habitat-by-parameter maps. They use bilinear interpolation: linear in the actual habitat proportion H and linear in log10(r) or log10(dispersal half-distance). Each original parameter-class mean is anchored at its logarithmic class midpoint. The85 H values and16 class means are unchanged; interpolation recovers every original node and cannot exceed the local corner range. No higher-order spline, model fit, new simulation, seed resampling or new confidence interval was used.

Display interpolation is limited to H=.005-.925 and to the first through last logarithmic class midpoints. The outer half-class strips are grey because interpolating beyond the midpoint range would require extrapolation. Both axes retain their previous full limits for comparison. Grey also marks the unavailable H region, including the sparse baseline at H=.975. The denser display grid adds no independent information or sampling resolution. Original un-interpolated figures and their audits remain preserved.

## Figure captions

**Ratio r.** Interpolated pointwise occupancy contrast against Only-Random across habitat proportion and extinction-to-colonization ratio. Panels compare Edge-Random (left) and Access-based (right), with rho=.9, .99 and .999 from top to bottom. Underlying nodes are arithmetic means across profiles within16 logarithmic r classes of their scenario-minus-Only-Random median occupancy, based on16 and8 landscape seeds, respectively. Dispersal half-distance varies within each class under the adopted joint sampling design. Bilinear display interpolation is linear in H and log10(r) between the85 observed H levels and16 class midpoints. Red indicates positive and blue negative contrasts on the unchanged-1 to+1 occupancy-unit colour scale. Grey indicates locations outside interpolation support. No inferential interval is shown. Averaging remains conditional on the nonrectangular parameter domain and can conceal profile differences.

**Dispersal half-distance.** As above, using logarithmic half-distance classes, with r varying among profiles in each class. Interpolation is linear in H and log10(half-distance), restricted to the first and last class midpoints. All other statistical definitions, scenario panels and colour limits are unchanged.

## Reproduction

`.venv/bin/python -m src.tools.codex_run final-habitat-parameter-interpolated`

After actual PDF rendering and viewing, use `--confirm-visual-review`.
