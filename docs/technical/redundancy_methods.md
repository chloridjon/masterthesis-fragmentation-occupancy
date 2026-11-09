> Technical calculation record from the thesis. Generated-table links refer to local outputs. Use the public workflow for current execution; earlier pilot replication does not replace the final 8/16 design.

# Analysis 1.2.2: methods for the 30-block sample

The seven fixed focal scenarios comprise Only-Random at $\rho=0$ and
Edge-Random and Access-based at $\rho=0.9$, $0.99$, and $0.999$. The updated
sample uses 30 matched seed blocks, each a complete trajectory with eight
habitat snapshots. The $200\times200$-cell generator uses periodic
boundaries; four-neighbour patches and structural metrics use open
boundaries. Access-based trajectories use the reference $\eta=1$, $k=3$,
and $\lambda=0.5$. Analysis 1.2.1 remains at ten blocks.

The first ten blocks are identical to the validated original 1.2.2
sample. Its 40 non-Access trajectories were copied unchanged after
producer-fingerprint verification, and the 30 Access reference trajectories
were read from the validated 1.2.1 source. The additional 20 blocks were
generated for all seven focal scenarios with the same seed matching,
generator, metric definitions, and habitat levels. Only the selected
Access reference was generated for these blocks; the full parameter
factorial was not extended. The two source fingerprints, source manifest
hashes, generator settings, and row-level fingerprints are recorded in
the data manifest and summary. The resulting sample has 210 trajectories
and 1,680 snapshots.

For every $H$ and each of the 15 unordered metric pairs, the signed
Spearman correlation pools finite paired observations across the seven
scenarios. Correlations never pool different $H$ levels. Pooling at fixed
$H$ does not control for scenario identity, so it describes association
in the selected scenario mixture. A coefficient requires at least four
finite pairs and two distinct values for each metric. An undefined
coefficient remains missing. Each matched seed block is omitted once to
assess influence; a maximum change above 0.1 is flagged as sensitive.

The 95% percentile bootstrap interval for a pooled correlation resamples
whole matched seed blocks. Its initial batch is 500 draws, increments are
500, and the maximum is 6,000. The endpoints and bootstrap median must each
change by at most 0.01 for three consecutive batches, with at least 95%
finite draws. Unconverged intervals would be withheld. All 100 estimable
pooled correlations converged; none crossed the seed-omission sensitivity
threshold. Complete-MST $D_{50}$ was constant at $H\geq0.7$, leaving its
20 pairwise correlations undefined at these four levels.

The across-$H$ summary gives median absolute correlation, quartiles, range,
share of defined levels with $|r_s|\geq0.8$, and sign counts. Each defined
level has equal weight. A pair is flagged for the diagnostic figures when
at least two defined levels reach $|r_s|=0.8$. The four pairs flagged in
the original ten-block analysis remain flagged in the 30-block analysis.
The broader exclusion screen requires both a median absolute correlation
of at least 0.8 and a strong correlation at at least half of the defined
levels. No pair met this screen. Definitions and interpretation determine
the final metric decision, rather than the threshold alone.

The four original diagnostic pairs were also recalculated separately
within each scenario and $H$ using the same finite-pair and variation
rules. Of 224 scenario–$H$–pair cells, 220 have defined correlations.
These estimates use at most 30 independent trajectories per scenario
and are presented as descriptive diagnostics. No formal test of
scenario-specific correlation differences was performed.

Mean Nearest-Neighbour Distance takes the minimum only over exact direct
Voronoi-neighbour edges. Mean Inter-Patch Distance and Complete-MST $D_{50}$
use the full depth-1 Voronoi-hybrid matrix. Metapopulation Capacity is not
part of structural metric selection.
