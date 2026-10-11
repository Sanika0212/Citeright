# Benchmarks

Every number in the README comes from these two suites. Datasets and HTTP responses are
cached in `~/.cache/citeright` (set `CITERIGHT_CACHE` to move it); runs write
`results.json` and `results.md` under `runs/`.

```bash
uv sync --extra models --extra bench --extra viz
```

## 1. Are the references real, correct and not retracted? (`bench match`)

```bash
uv run citeright bench match --seeds 60 --per-seed 12 --retracted 60
```

1. **Data.** A random sample of 60 articles published 2021 to 2024 with more than five
   citations is drawn from OpenAlex, across all fields, and up to 12 works from each
   reference list are kept. Each is written the way an author would (title, the first
   3, 5 or 10 authors, full names or initials, year, venue, and the DOI half the time)
   and given at most one fault:
   1. `correct`: unchanged; expected `verified`;
   2. `typo`: two letters swapped or a word dropped in the title; expected `verified`;
   3. `wrong_year`: the year moved by 2 to 4; expected `mismatch`;
   4. `wrong_author`: the first author replaced by the first author of another
      reference; expected `mismatch`;
   5. `doi_swap`: the DOI of another reference in the sample; expected `mismatch`;
   6. `fake_doi`: the right publisher prefix with an invented suffix; expected
      `mismatch` (the work exists, the DOI does not);
   7. `fabricated`: a title spliced from two real titles, the authors and venue of a
      third reference, a random year, and an invented DOI half the time; expected
      `not_found`;
   8. `retracted`: 60 retracted articles sampled from OpenAlex, written correctly;
      expected `retracted`.

   The references are shuffled and split 60/20/20 into training, development and test.
2. **Match model.** For every training reference, all candidates are gathered (the DOI's
   record, Crossref bibliographic search, and OpenAlex title search when Crossref finds
   no close title) and labelled by whether they are the true record. A logistic
   regression over the field comparisons is fitted on these pairs; its decision
   threshold is the one with the best status accuracy on the development references.
   The fitted model is written to `src/citeright/data/match_model.json`.
3. **Measures**, on the test references: accuracy of the status; precision and recall
   of "this reference has a problem" (any status other than verified); accuracy per
   fault type.
4. **Baselines.**
   1. `doi only`: flags a reference only when its DOI does not exist, as link checkers
      do;
   2. `title search`: verified when a search hit has a near-identical title (0.9 or
      more), otherwise not found;
   3. `rules`: the first candidate with a near-identical title, then hand-set
      tolerances on first author and year, plus the retraction flag.

## 2. Does the cited work back the sentence? (`bench support`)

```bash
uv run citeright bench support --checkers nli minicheck embedding lexical
```

1. **Data.**
   1. SciFact (Wadden et al., 2020): claims written from citing sentences, against the
      abstract of the cited paper; supports, contradicts, or not enough information.
      Train (919 pairs) for fitting, validation (340) for testing.
   2. Citation-Integrity (Sarol et al., 2024): real citing sentences from biomedical
      papers against the full text of the cited paper. Accurate citations count as
      supported, contradicting ones as contradicted, and every other label
      (not substantiated, oversimplified, misquoted, indirect, etiquette, irrelevant,
      no evidence) as not backed. Train (2,141) for fitting, test (606) for testing.
2. **Checkers.** `nli`: DeBERTa-v3 large trained on MNLI, FEVER, ANLI, LingNLI and WANLI;
   `minicheck`: MiniCheck RoBERTa large; `embedding`: cosine similarity of
   all-MiniLM-L6-v2 embeddings; `lexical`: share of the claim's words in the passage.
   Each scores the top 6 BM25 windows of the cited text.
3. **Calibration.** A multinomial logistic regression over the window scores and
   coverage features is fitted on both training sets together, one per checker, and
   written to `src/citeright/data/support_<checker>.json`.
4. **Measures**, on SciFact validation and Citation-Integrity test: accuracy and macro
   F1 of the three-way verdict; AUROC and expected calibration error of P(supports)
   for telling backed from not backed; the same AUROC and ECE for the raw checker score
   before calibration.

## Figures

```bash
uv run python scripts/make_figures.py
```

Reads both `results.json` files, checks `examples/draft.md` against
`examples/refs.bib` for the animation and the report screenshot, and writes everything
to `docs/assets/` in light and dark versions.

## Limits

1. **Planted faults are clean.** Real errors combine: a wrong year and a shortened
   title, a preprint cited with the journal's DOI. Each test reference carries one.
2. **The reference sample follows OpenAlex's coverage**, which is broadest for journal
   articles with DOIs and thinner for books, theses and conference papers without
   DOIs.
3. **Citation-Integrity is biomedical** and SciFact mostly is. Support checking in
   other fields may need refitting on labelled pairs from that field.
