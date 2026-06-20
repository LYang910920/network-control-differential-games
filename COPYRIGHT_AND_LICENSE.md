# Copyright and License Notes

This repository is organized as a personal educational and research workspace.

## No Blanket License

No project-wide open-source license is granted by default.

The root [`LICENSE`](LICENSE) file is a repository-level notice, not a permissive open-source license. Unless a file or upstream project explicitly says otherwise, all rights are reserved by the relevant rights holder. This is intentionally conservative because the repository combines:

- user-supplied tutorial documents;
- generated or adapted teaching code;
- generated figures and CSV summaries;
- scripts that interact with third-party reference repositories;
- results derived from third-party research code.

## Tutorial Documents

The files under `docs/` are included as user-supplied educational materials:

```text
docs/network_control_foundations.pdf
docs/network_control_foundations.tex
docs/code_walkthrough_and_model_adaptation_guide.pdf
docs/code_walkthrough_and_model_adaptation_guide.tex
```

Because this repository is public, keep these PDF and LaTeX files here only if you have the right to redistribute them. If in doubt, remove the PDF/LaTeX documents from the public repository or move the repository back to private.

## Tutorial Example Code

The teaching example code under `examples/foundations/code/` comes from the tutorial package supplied by the repository owner. If you want this repository to be public and reusable by others, choose an explicit license for your own tutorial code and document it here.

At present, the repository-level notice keeps this first-party teaching code and runner code under an all-rights-reserved default unless a future file or license notice explicitly grants broader rights.

First-party runner and teaching scripts include a short copyright notice pointing back to this file. Third-party source snapshots should keep their original upstream license headers and should not be overwritten with this repository's notice.

Recommended options:

- Keep all rights reserved for a private teaching repository.
- Add a permissive license such as MIT or Apache-2.0 only if you are comfortable granting reuse rights.
- Use a Creative Commons license for documents only if the document authorship and redistribution rights are clear.

## Third-party Reference Repositories

The three reference repositories are third-party works by their original authors. They are not relicensed by this repository.

This repo provides:

- source snapshots under `examples/reference/reference_repositories/`;
- upstream links;
- a download script;
- a lightweight smoke-run script;
- a compatibility patch;
- generated smoke-run figures and CSV summaries.

The included third-party source snapshots retain their original `LICENSE`, `README`, and citation files. Do not remove upstream attribution.

Direct local license files:

- [`OpinionMalware_TIFS_2025_Code/LICENSE`](examples/reference/reference_repositories/OpinionMalware_TIFS_2025_Code/LICENSE)
- [`PropagandaWar_TIFS_2024_Code/LICENSE`](examples/reference/reference_repositories/PropagandaWar_TIFS_2024_Code/LICENSE)
- [`Propaganda_TCSS_2025_Code/LICENSE`](examples/reference/reference_repositories/Propaganda_TCSS_2025_Code/LICENSE)

The default `.gitignore` excludes full-paper data folders under:

```text
examples/reference/reference_repositories/**/data/
```

This avoids accidentally vendoring external datasets into your own repository history. If you intentionally add datasets later, confirm their redistribution rights first.

## Generated Results

Generated reruns are written to ignored folders such as `artifacts/`, `simple_outputs/`, `example_outputs/`, and the output directory selected for `examples/reference/run_reference_smoke.py`. Curated README figures and current PDFs are kept under `docs/assets/` and `docs/` for convenience and reproducibility checks. If you publish these figures, cite the relevant tutorial code and upstream reference repositories as appropriate.

## Practical Publishing Recommendation

For a public GitHub repository, the safest default is:

1. Keep the tutorial documents only if you own or have redistribution rights.
2. Keep the tutorial example code if you choose a license for it.
3. Keep the reference smoke runner and patch.
4. Keep third-party source snapshots only with their upstream licenses and attribution intact.
5. Cite the upstream papers and repositories in `THIRD_PARTY_NOTICES.md`.
