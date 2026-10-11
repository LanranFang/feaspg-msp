# Packaging changes

The original training, projection, simulator and certificate mathematics were
retained. The following changes make the experiments usable outside the
research workspace:

1. Replaced `regularized_a1_v2.prepare()`'s dependency on a missing
   manuscript-adjacent script with readable input recipes. All 24 bundles were
   checked against every original array field. Real generated pools were
   regenerated from the included fitted models and also checked exactly.
2. Kept the full-comparison shard out of the general A1 entry point, since it
   uses a separate policy builder. Avoided double-counting that shard in the
   combined selection/evaluation driver.
3. Removed the plateau driver's dependency on hashes of manuscript files.
   Its numerical protocol and source-code hashing remain intact.
4. Replaced local absolute paths in copied metadata and reporting instructions
   with relative paths or `python`. Redirected the single-reservoir report's
   figure export into the artifact's output directory.
5. Added portable replay, plotting, summarization, input verification, and
   smoke-test entry points. The linearized-SDDP launch driver was reconstructed
   around the retained solver implementation and published protocol; the
   original manuscript-adjacent driver was unavailable.
6. Updated the main controlled-budget plot to match the revised paper:
   two legend rows with black text and shaded mean ±1 sample-SD bands.
   Numerical results and printed font sizes are unchanged. The main
   certificate figure also has an 18%-narrower variant with the original
   printed font sizes; the original full-width appendix variant is retained.
7. Separated archived results and selected A1 weights from newly generated
   outputs. Excluded unrelated online/carryover development experiments,
   unrelated manuscript drafts, Git metadata, binaries, logs, credentials, and caches.

`artifact/source_provenance.json` records original and packaged source hashes.
The coupled-certificate engine's original source checks remain enabled.
`MANIFEST.sha256` records the final distributed files. Cached `.pkl` bundles
are created locally for compatibility with the original training APIs; the
distributed input specifications use JSON and numerical NPZ files.

## Alignment with the supplied preprint (2026-10-11)

- Reused the portable submission code without changing its training, projection,
  simulator, or certificate mathematics.
- Identified the exact 104-page PDF by SHA-256 and added named citation metadata.
- Added Figure 2's three-panel drawing, Figure 3's immutable validation histories,
  compact preprint Figures 4/5, and Figure 7's regularized cost/latency drawing.
  Drawing computations are retained from their research scripts; input/output
  paths now resolve inside this repository.
- Retained the 5,000-path hydrology extension in the current published CSV.
- Added a numerical rebuild of Table 1 and the full figure/table-to-code map.
- Replaced anonymous front-page text and old figure numbering with English
  preprint documentation. Historical validation remains dated and separate.

`artifact/preprint_provenance.json` identifies the new exports and their source checksums.
`artifact/source_provenance.json` records the inherited portability changes.

## Code-only GitHub distribution

The paper, diagram/table source snapshots, and exported images are excluded
from Git and release archives. README and citation metadata link to
https://optimization-online.org/?p=37002. Plotting reads numerical records
and writes figures locally; it does not require a bundled manuscript.
