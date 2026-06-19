# Validation Summary

Built on 2026-06-19 from the local three-repository workspace.

## Deck Checks

- `node build_deck.cjs --template /path/to/Deakin_TEAL_16x9.potx`: passed.
- `pptx_static_audit.py three_repo_student_onboarding_deakin.pptx`: passed with 61 slides and 0 issues.
- Rendered outputs: 61 slide PNGs, 61 layout JSON files, `rendered/contact_sheet.png`, and image-based PDF.
- Visual spot checks: contact sheet plus slides 06, 26, 36, 39, 49, 54, and 61.

The artifact-tool build emits `Unsupported image: image/x-emf` while reading one template resource. The committed deck does not include the private POTX; rendered slides use editable text for the Deakin label and footer.

## Repository Checks

- `python3 -m compileall .`: passed in all three repositories.
- `pytest -q` with the foundation `.venv`: foundation 11 passed, Note 1 9 passed, Note 2 7 passed.
- Foundation smoke: `.venv/bin/python run_all.py --skip-reference --skip-scalability` passed.
- Note 1 smoke: `PATH="../network-control-differential-games/.venv/bin:$PATH" bash scripts/run_smoke_tests.sh` passed.
- Note 2 smoke: `PATH="../network-control-differential-games/.venv/bin:$PATH" bash scripts/run_smoke_tests.sh` passed.
- LaTeX builds passed for the foundation note, foundation code walkthrough, Note 1 main note, Note 1 implementation/code guides, Note 2 main note, and Note 2 implementation/code guides.
- `cross_repo_audit.py`: no owned `lecture` hits. Remaining findings are known large teaching scripts and common helper/function names.

## Licensing Boundary

The Deakin POTX was used only as a local build input. Do not add it to public Git history unless redistribution rights are confirmed.
