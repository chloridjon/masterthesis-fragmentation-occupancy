# Deforestation patterns, landscape structure and patch occupancy

Computational supplement to Jon Magnus Donner's master's thesis,
*How Deforestation Patterns Shape Landscape Structure and Species-Dependent
Responses: A Simulation Study with Tropical Forest Birds as an Ecological Reference*
(Universität Leipzig, 9 November 2026).

The reference is the final integrated manuscript, rather than earlier chapter
drafts. This repository contains source code, configurations, methodological
details and the 61 final scientific figures. Generated datasets, simulation
caches, external publications and the thesis manuscript are not distributed.

## Start here

- [Detailed methods](docs/methods.md): includes the details delegated to this
  supplement in the thesis.
- [Reproduction instructions](docs/reproducibility.md): installation, execution
  order, external inputs, local outputs and review checkpoints.
- [Thesis figure index](docs/thesis_figures.csv): all final figures and source
  identities, including Appendices A and B.
- [Repository promises](docs/repository_promises.csv): the 19 passages in the
  final manuscript that refer readers to the accompanying project.
- [Validation status](docs/validation.md): what has actually been tested and
  which complete computations remain unverified.

## Installation and commands

Use Python 3.12 and an environment of your choice. The original development
environment is not included. On Linux:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r environment/requirements.txt
.venv/bin/python reproduce.py check
.venv/bin/python reproduce.py test
.venv/bin/python reproduce.py list
.venv/bin/python reproduce.py plan
```

`run <step>` executes one registered step; `pipeline <step>` executes its
prerequisites first. The full production design can require substantial
resources. Inspect the plan before running it. External empirical sources must
be acquired separately as described in the instructions. Review flags must
only be supplied after inspecting the corresponding figures.

The internal `Masterarbeit_final/` directory preserves the scientific code's
relative paths and source identities. The outer checkout can have any name.
There is no dependency on the author's parent working repository.

The code generates data below `Masterarbeit_final/data/processed` and results
below `Masterarbeit_final/results`. These directories are excluded from Git.
Final PDFs under `figures/` are reference artwork, not calculation inputs.

## Interpretation

The study compares seven fixed removal scenarios. Species parameters describe
a numerical target distribution, not the prevalence of biological species.
Modelled mean equilibrium patch occupancy differs from empirical forest-site
occupancy. The bird curves reconstruct published predictions; they are not raw
detection histories. The confidence level of the digitised grey bands is an
explicit assumption, not verified statistical coverage.

The source assembly and tests must not be mistaken for a completed rerun of
every production simulation. See the validation record for the exact status.

## Citation and licensing

See [CITATION.cff](CITATION.cff). Original software is distributed under the
[MIT License](LICENSE). External inputs retain their own terms. Reference
figures may contain digitised third-party predictions; their inclusion does
not relicense those source materials.
