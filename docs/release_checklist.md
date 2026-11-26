# Release checklist

Before tagging a release of EthoLive:

- [ ] `pytest -q` passes in a fresh environment (`pip install -e ".[video,figures,dev]"`).
- [ ] `python scripts/verify_mouse_validation.py` gives macro F1 0.771 with the release's `mouse.pt`.
- [ ] `python scripts/paper_figures.py` regenerates `paper/figures/` without changes.
- [ ] Model files attached to the GitHub release; `shasum -a 256 -c models/CHECKSUMS.sha256` passes on the downloads.
- [ ] `models/README.md` lists every attached file and its license constraints.
- [ ] `CITATION.cff` version and repository URL match the tag; publication details added only once published.
- [ ] README results table matches `results/` (no number without a file behind it).
- [ ] No data, outputs, credentials or personal paths tracked (`git ls-files` review).
- [ ] A license file chosen and added.
