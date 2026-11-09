# External inputs and their local preparation

No source dataset is included in Git. Download or otherwise lawfully obtain
the inputs below. Preserve their versions and original contents. Local
preparation writes derived files; it must not overwrite the source products.
Configuration paths in this document are relative to `Masterarbeit_final/`
unless a checkout-root path is written explicitly.

## Forest map and land mask

Use **JRC Global Forest Cover 2020, version 3**, not the similarly named Hansen
Global Forest Change product and not whichever version a `LATEST` URL happens
to serve. The [JRC version-3 page](https://forobs.jrc.ec.europa.eu/GFC/v3)
provides the product and access routes. Place the global COG at
`data/external/JRC_GFC2020_V3_COG.tif`. The original raster is approximately
39 GB, in EPSG:4326, with bounds −180, −60, 180, 80, and shape
1,680,000 × 4,320,000 cells. `config/real_source_metadata.json` preserves its
header description. A matching header alone does not prove identical pixels.

Use **GSHHG 2.3.7**, full resolution `f`, from the
[GSHHG distribution](https://www.soest.hawaii.edu/pwessel/gshhg/). Extract its
shapefiles under `data/external/gshhg-shp-2.3.7/GSHHS_shp/`. The mask includes
levels 1–4 with alternating land/water signs and excludes Antarctica levels.
The exact original settings are in `config/real_land_mask_metadata.json`.

Run the catalogue with EPSG:6933, 64 km spacing, macrogrid seed 20260831 and
all eligible centres (`--max-centers 0`). The original offset is
x = 10365.126793736528 m and y = 41185.34665668228 m. There are 115,116
centres within the source bounds and 32,288 land centres. Five window extents
give 161,440 candidate records. The catalogue retains the information needed
for both H ranges, even when a candidate is outside the initial range.

The original low-H selection uses **seed 20260833**, 100 windows per stratum,
H ≤ 0.6 and exclusion of zero forest cover. The high-H extension uses
**seed 20260902**, 0.6 < H ≤ 1, and the same sample size. H-bin assignment uses
the original 1e−12 tolerance at boundaries; omitting it changes the sample.
Candidates are sorted by identifier within each extent–H stratum before the
seeded draw. There are 59,509 eligible low-H and 43,383 high-H candidates.

```sh
.venv/bin/python reproduce.py run real-catalogue
.venv/bin/python reproduce.py run real-high-cover
.venv/bin/python reproduce.py run real-selection-audit
.venv/bin/python reproduce.py run real-metrics
```

The independent selection audit must reproduce 10,000 identifiers in the
original order. The public workflow enforces the published fingerprint of
that identifier order, so a different but internally consistent new sample
does not pass. Expected original catalogue and selection hashes are in
`external_inputs.json`. Hashes of whole CSV files can differ due to local
path or elapsed-time columns; identifier order and raster identities therefore
provide separate scientific checks. The published audit verifies selection
against the original catalogue; it is not a fresh global extraction audit.

## Published bird predictions

Use Table 1, Figure 2 and supplementary Table S2 of Morante-Filho, Benchimol
and Faria (2021), DOI
[10.1007/s10980-020-01121-6](https://doi.org/10.1007/s10980-020-01121-6).
The author-uploaded full text used for Table 1 is linked in
`config/bird_source_manifest.json`. These are published model predictions,
not the original field-detection histories. Do not replace them with another
bird dataset. All 19 rows are retained in source order; 15 Forest rows enter
the reconstruction.

`reproduce.py run bird-inputs` downloads the publisher PNG and supplementary
DOCX into `Masterarbeit_final/data/external/morante_filho_2021/`, checking
their expected SHA-256 values. Publisher access can require manual download.
Do not bypass access restrictions. The source manifests specify exact URLs,
file identities and limitations.

The article table itself requires a documented text extraction from the
external publication. Save it as `table1_web_extract.txt` in that same local
directory. The parser accepts source rows in this format, one row per line:

```text
L<source line number>: <published species name> <Forest or Non-forest> <guild> <psi at 20%> <psi at 40%> <H at 60% occupancy, percent>
```

Preserve the source's two-decimal occupancy values, one-decimal threshold,
row order and species spelling. Accepted guild tokens are `frugivore`,
`insectivore`, `omnivore` and `nectarivore`. Line-number prefixes are extraction
metadata, not data values. The reconstruction rejects a wrong species count,
duplicate identifiers, inconsistent classification or incompatible held-out
threshold. Its emitted `table1.csv` can be checked against the original hash
`11c231c4b068514dd6970ac4a0ff818d6c6f26056ff283194ac8fb4ca73c2a4a`.

Run `bird-reconstruction`, then `bird-digitisation`. Figure 2 source geometry,
masks and calibration are specified in `11_figure2_digitization.json`.
`bird-band-source-review` implements the source-visible correction used in
the final selected-species display. Neither grey-band extraction nor that
display correction affects the reconstructed central predictions or RMSEs.

## Geographical context in figures

The global sampling map uses the Natural Earth low-resolution shapefile
distributed as a pyogrio test fixture. `prepare-map` retrieves that fixture
from pyogrio v0.11.1 and verifies the original `.shp` hash before conversion
to the local `world_land.geojson`. It is geographical context, not an input
to landscape metrics.

The two real examples in Appendix B use the original selected candidate IDs
and forest-grid hashes in `config/real_examples.json`. Their forest rasters
are obtained from the locally generated sample. Overlaid infrastructure and
river lines come from OpenStreetMap; configuration preserves sample IDs,
way IDs, versions, source timestamps and feature types. `real-examples`
requests historical geometry for 20 October 2026 and rejects changed way
versions. Historical endpoint access and exact line geometry have not yet
been independently retested for this assembly. These map lines are visual
context only and do not enter forest metrics. Attribute OpenStreetMap
contributors and retain the applicable ODbL terms.

## Reuse and provenance

External content retains its source terms. An unavailable download is an
explicit prerequisite, not permission to substitute another dataset. New
downloads and outputs must record their actual identities. Never copy a
completed audit from the original project to certify different local data.
