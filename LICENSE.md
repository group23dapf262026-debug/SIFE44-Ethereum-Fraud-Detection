# License scope

## Original contributions: MIT

The authors license their original contributions listed in
[CODE_LICENSE_SCOPE.json](CODE_LICENSE_SCOPE.json) under the [MIT License](LICENSE).
The grant applies only to copyright owned or controlled by the listed authors.
It permits use, modification and redistribution, including research reproduction
and commercial use, subject to the MIT notice and disclaimer conditions.

The full-file list includes the SIFE44/Pure36 feature extractor, feature tests,
analysis/replay tools, the portable runner, the verifier, statistical redraw code,
and author-written repository documentation. The machine-readable scope records
the current SHA256 of each listed file.

For mixed main/ablation notebooks, only the specifically listed SIFE feature
definition cells are covered by this grant. Their code is the inline version of
the listed feature extractor. The notebook as a whole is not relicensed MIT.
Original code-cell bytes and completed-run source identities remain unchanged.

## Components outside this grant

The following are not licensed by the authors under the root MIT grant:

- Meta-IFD-derived or mixed model/attention/view-generation components, and
  other experiment files not explicitly listed as author-owned.
- Public pretrained decoder tensor values and cells containing those values.
- Upstream benchmark graphs, raw/processed data, address/label reference records,
  root NPZ evidence, prediction CSVs and original result archives/manifests.
- Supplied paper figure PDFs and their raster preview. The figure-generation
  Python code can be MIT-covered while the artwork/export remains outside it.
- Installed libraries, which retain their own licenses and are not vendored here.

Exclusion means that this MIT grant supplies no additional permission for those
components; it does not assert that every excluded file has the same owner or
terms. Existing permissions, if any, must come from the relevant rights holders.

## Upstream status

No Meta-IFD license file was identified in the recorded local reference snapshot
or the public repository file list checked on 4 November 2026. This is an
observation, not a claim that the upstream authors refused permission. This
artifact does not grant permission on their behalf. Attribution alone does not
replace an upstream permission grant.

The mixed Meta-IFD components remain present for traceability. This license update
does not resolve permission to redistribute them; confirm upstream terms before
publicly distributing the entire mixed repository. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Citation

Please cite the SIFE44 manuscript and the reference works when academically
appropriate; see [CITATION.cff](CITATION.cff). This citation request does not add a
new restriction to the MIT license. Do not label the entire mixed repository as
unconditionally MIT-licensed.
