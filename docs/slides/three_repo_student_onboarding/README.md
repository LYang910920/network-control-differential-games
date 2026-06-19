# Three-Repo Student Onboarding Deck

This folder contains a Deakin-style onboarding deck for the three-repository tutorial family:

- network-control-differential-games
- note1-cyber-control-games
- note2-pinn-pidl-cyber-control

The deck is built from the current repository text and generated figures. The private Deakin POTX template is used as a local input and is not committed.

## Outputs

- `three_repo_student_onboarding_deakin.pptx`: editable PowerPoint deck.
- `three_repo_student_onboarding_deakin.pdf`: image-based PDF generated from rendered slides.
- `deck_content.json`: slide IDs, titles, takeaways, notes, commands, and visual metadata.
- `speaker_notes.md`: presenter notes for every slide.
- `asset_manifest.csv`: figure source, ownership/license note, slide usage, and alt text.
- `rendered/contact_sheet.png`: visual QA contact sheet.
- `rendered/slides/`: one PNG render and layout JSON per slide.
- `qa/deck_static_audit.json`: lightweight content audit.

## Rebuild

From this folder:

```bash
NODE_PATH=/Users/ylx910920/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules \
/Users/ylx910920/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node build_deck.cjs \
  --template /path/to/Deakin_TEAL_16x9.potx
```

The Deakin template is intentionally passed as a local file path. Do not commit the private POTX unless redistribution rights are confirmed. If the sibling Note 1 or Note 2 repositories are absent, the deck still builds, but slides that depend on their generated figures show a missing-figure placeholder.

## Teaching Use

Use the core slides for a 60-90 minute onboarding session. The appendix gives command, file, hyperparameter, debugging, glossary, and reference slides for live support.
