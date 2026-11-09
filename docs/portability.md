# Portability changes

1. The scientific package is retained under Masterarbeit_final to preserve original path identities.

2. The public runner resolves the checkout root dynamically and filters aliases to included scripts.

3. The final Chapter-5 label driver selects its 21 source SVGs from configuration instead of requiring private manuscript files. Data-artist invariance checks are retained.

4. The regional-proposal guard no longer hashes an unrelated private manuscript file. Numerical and source checks remain.

5. The sampling-map renderer reads freshly generated selected-window NPZ files directly instead of requiring the separate viewer archives. It still validates each raster hash against the recomputed metrics.

6. External input default paths are relative. Original catalogue and selection settings are retained as metadata.

7. New-run provenance is explicitly rebound after generation; old audit receipts and generated data are not bundled.

8. Python package versions reflect the tested Linux verification environment; the originally recorded lock is retained separately, not asserted to have been installed here.
