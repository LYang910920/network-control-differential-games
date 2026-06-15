# Copyright and License Notes

This repository is organized as a personal educational and research workspace.

## No Blanket License

No project-wide open-source license is granted by default.

Unless a file or upstream project explicitly says otherwise, all rights are reserved by the relevant rights holder. This is intentionally conservative because the repository combines:

- user-supplied tutorial documents;
- generated or adapted teaching code;
- generated figures and CSV summaries;
- scripts that interact with third-party reference repositories;
- results derived from third-party research code.

## Tutorial Documents

The files under `docs/` are included as user-supplied educational materials:

```text
docs/lecture_note.pdf
docs/lecture_note.tex
docs/code_walkthrough_and_model_adaptation_guide.pdf
docs/code_walkthrough_and_model_adaptation_guide.tex
```

Before publishing this repository publicly, confirm that you have the right to redistribute these PDF and LaTeX files. If in doubt, keep the repository private or remove the PDF/LaTeX documents before pushing.

## Lecture Example Code

The teaching example code under `examples/lecture/code/` comes from the tutorial package supplied by the repository owner. If you want this repository to be public and reusable by others, choose an explicit license for your own tutorial code and document it here.

Recommended options:

- Keep all rights reserved for a private teaching repository.
- Add a permissive license such as MIT or Apache-2.0 only if you are comfortable granting reuse rights.
- Use a Creative Commons license for documents only if the document authorship and redistribution rights are clear.

## Third-party Reference Repositories

The three reference repositories are third-party works by their original authors. They are not relicensed by this repository.

This repo provides:

- upstream links;
- a download script;
- a lightweight smoke-run script;
- a compatibility patch;
- generated smoke-run figures and CSV summaries.

Downloaded third-party repositories should retain their original `LICENSE`, `README`, and citation files. Do not remove upstream attribution.

The default `.gitignore` excludes:

```text
examples/reference/reference_repositories/
```

This avoids accidentally vendoring third-party code and datasets into your own repository history. If you intentionally vendor third-party repositories, include their original license files and keep their attribution intact.

## Generated Results

The files in `examples/*/results/` are generated outputs from the tutorial and smoke-run scripts. They are included for convenience and reproducibility checks. If you publish these figures, cite the relevant tutorial code and upstream reference repositories as appropriate.

## Practical Publishing Recommendation

For a public GitHub repository, the safest default is:

1. Keep the tutorial documents only if you own or have redistribution rights.
2. Keep the lecture example code if you choose a license for it.
3. Keep the reference smoke runner and patch.
4. Do not commit downloaded third-party repositories unless you intentionally vendor them with their licenses.
5. Cite the upstream papers and repositories in `THIRD_PARTY_NOTICES.md`.
