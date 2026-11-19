# Third-party notices

## Meta-IFD

Reference article: C. Jin et al., *Enhancing Ethereum Fraud Detection via Generative and Contrastive Self-Supervision*, IEEE Transactions on Information Forensics and Security 20, 839–853 (2025). DOI: https://doi.org/10.1109/TIFS.2024.3521611

Reference code/data access: https://github.com/GISec-Team/Meta-IFD

Recorded upstream commit: `439c6daa936f67f19486be76642e690a2153d0c5`.

HMSL/attention components and the public conditional decoder follow that implementation, with study-specific integration recorded by the sources. Numeric decoder reference tensors are data, not SIFE-trained classifier checkpoints. Benchmark graphs remain upstream inputs rather than part of this repository.

The root MIT License applies only to author-owned contributions and feature cells listed in [CODE_LICENSE_SCOPE.json](CODE_LICENSE_SCOPE.json); it does not cover upstream or mixed components. No upstream license file was present in the local reference snapshot, and none was identified in the public repository file list checked on 4 November 2026. Upstream permission to redistribute the derived/mixed components remains unconfirmed. The authors do not grant that permission on behalf of Meta-IFD. Keep attribution and confirm the upstream terms before distributing the entire mixed repository. See [License scope](LICENSE.md).

## Libraries

The code uses NumPy, Pandas, SciPy, scikit-learn, PyTorch, PyTorch Geometric, LightGBM, joblib, threadpoolctl and Matplotlib. These packages remain under their own licenses; their code is not vendored here. Requirements and observed runtime versions identify dependencies without relicensing them.

## Manuscript figures

The five supplied paper figure PDFs were provided by the authors. No third-party paper figures were copied into the figure folder. Venue-specific figure-origin requirements are separate from software packaging and are not automatically satisfied by this repository.
