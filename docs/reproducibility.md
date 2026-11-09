# Reproducing the thesis

## Environment

Python 3.12 on Linux is the supported production setup. Install the pinned
packages in `environment/requirements.txt`. The recorded original package
list is preserved as `requirements-original.txt`; it differs from the tested
assembly environment and is not claimed to have been installed here.
Some original dependency annotations were maintained manually.

Install the Latin Modern OpenType Roman 8 and Math fonts, fontconfig, Cairo,
librsvg and Poppler. Typical Debian/Ubuntu packages are `fonts-lmodern`,
`libcairo2`, `librsvg2-2` and `poppler-utils`. Final renderers also resolve
Latin Modern Math under `/usr/share/texmf/fonts/opentype/public/lm-math/`.
Check that the required file exists on your distribution. Pure numerical
tests do not establish that the graphic system is configured.

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r environment/requirements.txt
.venv/bin/python reproduce.py check
.venv/bin/python reproduce.py test
```

`check` validates the registry, source syntax, final reference PDF hashes,
documentation links for every repository promise and required packages.
`test` runs the original scientific unit and fixture tests in the standalone
layout. Neither command reruns the production simulations.

## Execution model

The [workflow registry](../workflow.json) names the computation and rendering
steps, their arguments and prerequisites. It has separate entries for
simulation, statistics, independent audits and presentation. Source dates
and numbering in internal filenames identify the original implementations;
the public step names describe their role.

```sh
.venv/bin/python reproduce.py list
.venv/bin/python reproduce.py plan chapter4-figures
.venv/bin/python reproduce.py plan complete-thesis
.venv/bin/python reproduce.py run landscapes --workers 4
.venv/bin/python reproduce.py run landscape-summary
```

`run` executes one step. Arguments after the step name are forwarded to its
scientific script. `plan` prints prerequisites and commands without computing.
`pipeline` executes prerequisites in order, but stops at an unconfirmed visual
review. It does not infer review from elapsed time, and it does not skip all
earlier commands just because a receipt exists. The scientific producers
perform their own cache validation and reuse. Follow the displayed sequence
or execute subsequent `run` steps after each checkpoint.

For a small computational check, `run landscapes --max-trajectories 1
--workers 1` generates one original-size trajectory and deliberately withholds
the complete-ensemble aggregate. Its successful exit is not a completed
landscape analysis. To generate only the constructed hybrid fixture use
`run hybrid-schematic`. The majority-aggregation illustration can be generated
with `run method-figures --figure majority-aggregation`.

## Full order and scope

1. Acquire the external sources following [data sources](data_sources.md).
   Recreate and audit the real-landscape sample. Source data stay read-only.
2. Run `distance`, `access`, `redundancy` and `redundancy-summary` for the
   reported preliminary landscape decisions. `landscapes` and
   `landscape-summary` generate the seven-scenario structural ensemble and its
   summaries; `coverage` then compares that ensemble with recomputed empirical
   metrics. The registry includes the supplementary geometric diagnostics.
3. Run the limit-based and transition parameter surveys, the initial 128-point
   response calculation and its independent vector/plausibility checks.
   Then reconstruct the response-domain bounds and the conditional 256-point
   sample. This recreates a reported selection procedure rather than merely
   hard-coding its selected numerical outcome.
4. Compute the first five conditional-design blocks and their independent
   summary. After the review checkpoint, `occupancy-bind-producer` binds the
   8/16 extension configuration to the newly calculated producer fingerprint.
   This updates a provenance value in configuration, not a scientific setting.
   Original audit receipts are never bundled as calculation inputs.
5. Extend to the eight shared and eight additional directed blocks. Audit and
   export the unequal design, then calculate the separately selected reference
   on 20 runs per scenario. Reference uncertainty is calculated separately.
6. Reconstruct published bird responses, digitise Figure 2 and perform the
   sequential original/conditional/final matching steps. Generate the response
   grouping and selected-example surfaces. Earlier fit stages remain separate
   prerequisites because the final source checks their input lineage.
7. Compute integrated contrasts, H-by-parameter maps and parameter correlations.
   Run weighted metric associations through prepare, bootstrap and export,
   followed by rank/replication sensitivities and the final exploratory A–D
   region definitions.
8. Compute the q = 1 paired pilot, independent recovery checks and eight-run
   extension, then its five-H response-domain screen. The final thesis excludes
   the separate area-weighted response analysis; its development scripts are
   not public workflow roots.
9. Generate conceptual examples, source figures and the final chapter layouts
   and label revisions. `final-method-labels` completes the last two method
   figure label changes. [The figure index](thesis_figures.csv) records the
   final PDF identity and entry step. Supplementary diagrams and intermediate
   vector sources are generated locally, rather than uploaded as caches.

`complete-thesis` collects the producer paths for all 61 figures, the two
tables and the promised supplementary analyses. Its final check reports
missing outputs; it does not replace the individual numerical audits.

## Review checkpoints and numerical failures

The original source binds some downstream calculations to completed numerical
and visual audits. Preserve this distinction. After generating and inspecting
the relevant figures, run the corresponding `*-review` step with `--reviewed`.
For example:

```sh
.venv/bin/python reproduce.py run response-domain
# Render and inspect the generated figures.
.venv/bin/python reproduce.py run response-domain-review --reviewed
```

The public driver does not mark every intermediate image reviewed on a new
checkout. Region and self-pilot review actions provide portable equivalents
of the original authoring-session closeout receipts, after checking the
numerical audit. The numerical source functions and their criteria remain.

If a solver does not converge, stop and inspect its diagnostic. The included
continuation/recovery helpers use the documented tighter solve and independent
equation checks. Do not count an unfinished trajectory, failed interval or
missing source as a successful complete run. `tools/job_aliases.json` preserves
the required original aliases for subprocesses; the public runner resolves
them from the current checkout.

## Source units, caches and reproducibility limits

The final manuscript expresses edge density per cell area. Some original
stored tables used unlike adjacencies divided by 79,600, and their plot stage
converted by 1.99. Current producers generate the canonical denominator
40,000 directly. After successful fresh structural, empirical and access
production, `figure-provenance` records the new table hashes and denominator
40,000 so that a second conversion is not applied. It refuses tables without
the required producer receipts. This step does not certify old copied caches.

Local data and results are excluded from Git. Source/configuration fingerprints
protect reuse; incompatible caches must be regenerated. Outputs include the
numerical tables and arrays behind graphics. Source code, config and input
identities, exclusions, solver status and resampling settings belong in the
generated provenance. A wrapper's successful process exit is only a command
receipt; completed scientific audits and sample sizes must also be checked.

Exact PDF bytes can change with rendering libraries and font containers.
Validate scientific arrays and geometry with declared tolerances, then check
the rendered result at the intended size. The reference PDFs under `figures/`
identify the figures included in the final manuscript.

## Resources and validation

The complete design includes 1,400 structural trajectories at 88 H values,
an initial 128-point response survey, a conditional 256-point design on 104
trajectories, a separate 20-run reference, independent patch-vector audits,
bootstrap calculations, empirical extraction and sensitivities. Dense distance
and kernel matrices can dominate memory, so begin with low worker counts.
The 39 GB forest source and generated geometry/vector caches require substantial
disk space. A universal full-run duration has not been measured for this release.

See [validation status](validation.md). The supplied checks do not claim that
the entire chain has already been rerun from an empty checkout.
