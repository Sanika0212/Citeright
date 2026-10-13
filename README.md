<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
  <img alt="Citeright: check that every citation is real, correct, not retracted, and says what you cite it for" src="docs/assets/banner-light.svg" width="100%">
</picture>

**Citeright checks every citation in a draft: that the reference exists, that it is the
paper you meant, that it has not been retracted, and that the cited work actually says
what the sentence citing it claims.**

Writing assistants produce references that look right and are not: a plausible title, real
author names, a DOI with a real publisher's prefix. People make the quieter mistakes:
a wrong year, a DOI copied from the neighbouring entry, a paper cited for a result it
does not contain, a study cited years after its retraction. Link checkers catch only a
DOI that does not resolve. Asking an LLM "is this reference real?" asks the same kind of
model that invented it.

Citeright looks each reference up in the scholarly record and checks each citing
sentence against the cited text:

1. **Is it real?** DOIs are checked against the DOI handle system, then the reference
   is matched to records in Crossref, OpenAlex and arXiv by title, authors, year and
   venue. A model fitted on real references with planted faults turns the comparison
   into a calibrated match probability; nothing is a hand-set threshold.
2. **Is it right?** When the work exists but a field is wrong, Citeright says which
   field and what the record says: "DOI 10.1038/nature14539 belongs to another work
   ('Deep learning'); this work's DOI is 10.1109/cvpr.2016.90".
3. **Is it retracted?** Retractions, corrections and expressions of concern come from
   Crossref's Retraction Watch data and OpenAlex, with the notice's date and DOI.
4. **Does it say that?** Each citing sentence is checked against the passages of the
   cited abstract (or full text, if you have it) that match it best, with a calibrated
   verdict: supports, contradicts, or not found in the cited work, and the passage
   that decides it.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/check-dark.gif">
  <img alt="Animation: the references of a short draft are looked up one by one and marked verified, mismatch, not found or retracted; then each citing sentence is linked to its reference with a verdict" src="docs/assets/check-light.gif" width="100%">
</picture>

## Quickstart

```bash
pip install "citeright[models] @ git+https://github.com/Sanika0212/Citeright"
citeright check paper.tex --bib refs.bib --html report.html
```

It reads LaTeX (`\cite`, `\citep`, `\citet`, ...), Pandoc Markdown (`[@key]`),
numbered (`[1]`, `[2-4]`) and author-year citations, with a BibTeX file, a
`thebibliography` block or a plain reference list.

On [examples/draft.md](examples/draft.md), a short text with real papers and five
planted problems:

```text
ok lecun2015                verified
!! vaswani2017              mismatch
     year is 2017, not 2015
!! he2016                   mismatch
     DOI 10.1038/nature14539 belongs to another work ('Deep learning'); this work's DOI is 10.1109/cvpr.2016.90
?? smith2021                not_found
     DOI 10.5555/jmlr.2021.0193 does not exist
XX wakefield1998            retracted
     retracted on 2010-02-06 (10.1016/s0140-6736(10)60175-4)
     correction published 2004-03-06 (10.1016/s0140-6736(04)15715-2)

line 8: [lecun2015] not_found (P(supports) 0.17)
     Deep networks are provably robust to adversarial examples.
```

The fabricated reference, the wrong year, the borrowed DOI and the retracted study are
caught, and so is the sentence that cites a real, correct paper for something it does
not say.


The HTML report is a single file with no external requests: a summary, a graph of
which sentences cite which references, every reference with what is wrong with it, and
every citing sentence with the passage that decides it.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/report-dark.png">
  <img alt="HTML report: status counts, a graph of which sentences cite which references, and each citing sentence with its verdict and the passage of the cited abstract that decides it" src="docs/assets/report-light.png" width="100%">
</picture>

## In CI

```bash
citeright check paper.tex --bib refs.bib --no-support --fail-on retracted,not_found
```

exits with 1 when the draft cites a retracted work or a reference that cannot be found,
so a paper repository can refuse to merge one. `--json findings.json` writes everything
for other tools.

## In Python

```python
from citeright.check import check
from citeright.report import write_report

report = check(open("paper.tex").read(), open("refs.bib").read())
for r in report.resolutions:
    print(r.reference.key, r.status.value, r.issues)
for c in report.checks:
    print(c.verdict.value, round(c.probability, 2), c.citation.claim, c.evidence)
write_report(report, "report.html")
```

Every HTTP response is cached in `~/.cache/citeright`, so checking the draft again
after an edit makes no new requests for unchanged references. OpenAlex limits anonymous
searches per day; Citeright searches Crossref first and uses OpenAlex as a fallback.
Set `OPENALEX_API_KEY` for a larger budget, and `CITERIGHT_MAILTO` if you want your
address sent to the APIs' polite pools (it is never sent otherwise).

## Results

Every number comes from [docs/benchmarks.md](docs/benchmarks.md), with the commands to
reproduce it.

### Are the references real, correct and not retracted?

649 works drawn from the reference lists of 60 random 2021 to 2024 articles in
OpenAlex (all fields) and 56 retracted articles, each written the way an author would
and given at most one planted fault: a typo in the title, a year off by 2 to 4, another
reference's first author, another reference's DOI, an invented DOI, or a fabricated
reference spliced from real ones. The match model is fitted on 60% and its threshold
chosen on 20%; every number below is on the other 141 references.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/match-dark.png">
  <img alt="Share of references given the right status, per planted fault: Citeright 97% overall, hand-set rules 96%, title search 63%, DOI check 55%; title search and the DOI check miss every wrong year, wrong author and borrowed DOI" src="docs/assets/match-light.png" width="100%">
</picture>

| Method | Right status | Problem precision | Problem recall |
|---|---|---|---|
| **Citeright** | **97.2%** | **97.6%** | **98.8%** |
| Hand-set rules on the same lookups | 96.5% | 97.6% | 97.6% |
| Title search only | 63.1% | 97.1% | 39.3% |
| DOI check only | 55.3% | 100.0% | 45.2% |

1. **Checking that the DOI resolves catches less than half the problems.** It misses
   every wrong year, wrong author and borrowed DOI, and the fabricated references written
   without a DOI.
2. **A title search finds fabrications but not errors**: a reference whose title exists
   passes whatever its year, authors or DOI say.
3. **Comparing every field catches nearly everything**: 98.8% of the references with a
   problem are flagged, and 97.6% of the flags are right. Hand-set rules over the same
   lookups come within one reference of the fitted model; 141 references cannot separate
   them. Where they differ, it is on retractions: an expression of concern or a
   withdrawal notice is often published under the paper's own title, and the rules take
   the first record with that title. The model scores every candidate and finds the
   article itself (8 of 9 retracted papers against 6 of 9).
4. **The four misses**: a title typo that no search returned; a metadata record that
   gives the first author's names in reverse order; a reference to a retraction notice,
   which OpenAlex flags as retracted and Crossref does not; and a reference with a wrong
   year that Citeright reports as retracted instead, which it is: the article was
   retracted in 2022.

### Does the cited work back the sentence?

Two expert-labelled datasets: SciFact (claims written from citing sentences, checked
against the cited abstract; 340 held-out claims) and Citation-Integrity (real citing
sentences from biomedical papers, checked against the full text of the cited paper;
606 held-out citations). The calibration is fitted on both training sets. A reporting
frame ("Smith et al. showed that ...") is stripped before checking, since the cited work
supports the claim inside it; this changes neither benchmark measurably but removes a
clear class of false alarms.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/support-dark.png">
  <img alt="AUROC per checker: NLI 92% on SciFact and 67% on Citation-Integrity, MiniCheck 88% and 67%, embedding similarity 79% and 63%, word overlap 78% and 64%" src="docs/assets/support-light.png" width="100%">
</picture>

| Checker | SciFact: accuracy (3-way) | SciFact: AUROC | Citation-Integrity: AUROC | Calibration error, raw score → calibrated (SciFact / Citation-Integrity) |
|---|---|---|---|---|
| **NLI (DeBERTa-v3 large), default** | **76.5%** | **0.915** | **0.668** | 0.140 → 0.084 / 0.358 → 0.079 |
| MiniCheck (RoBERTa large) | 67.9% | 0.878 | 0.666 | 0.172 → 0.077 / 0.464 → 0.057 |
| Embedding similarity | 62.6% | 0.789 | 0.632 | 0.248 → 0.054 / 0.107 → 0.051 |
| Word overlap | 61.8% | 0.780 | 0.641 | 0.061 → 0.054 / 0.156 → 0.041 |

1. **Against an abstract, the check is strong**: 76.5% three-way accuracy on SciFact
   (always guessing the commonest label gives 41%) and 0.915 AUROC for supported against
   not supported.
2. **Real citations against full papers are hard.** On Citation-Integrity the best
   checker reaches 0.668 AUROC; its three-way accuracy (63%) barely beats always saying
   "supports" (61%). Many inaccurate citations there are subtle: oversimplified, cited
   for a result the paper only mentions, or citing a paper that cites the source.
   Citeright treats these verdicts as leads to read, and shows the passage it judged by.
3. **Calibration matters most for the raw scores that look most confident**: the NLI
   model's probabilities are off by 0.36 on Citation-Integrity before calibration and by
   0.08 after.


## How it works

[docs/how-it-works.md](docs/how-it-works.md) covers parsing, resolution and the match
model, retraction data, and the support check with its calibration.
[docs/benchmarks.md](docs/benchmarks.md) describes both benchmarks and how to reproduce
them.

## Limits

1. **An abstract is not the paper.** Without full text, "not found" means "not in the
   abstract". The report always says which text was checked; pass full texts with
   `--texts` for a stronger check.
2. **Metadata has gaps.** Books, theses, very recent preprints and works without a DOI
   are harder to resolve. Treat a `not_found` for those as a lead to check by hand.
3. **The support check reads English** and is calibrated on biomedical and general
   scientific writing. `citeright bench support` refits it on labelled pairs from
   your field.
4. **Retraction data is as complete as Crossref's and OpenAlex's.**

## Development

```bash
uv sync --extra models --extra bench --extra viz
uv run ruff check src tests scripts && uv run mypy
uv run coverage run -m pytest && uv run coverage report
```

Tests never touch the network or download weights. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## References

1. Wadden et al., *Fact or Fiction: Verifying Scientific Claims*, EMNLP 2020 (SciFact).
2. Sarol, Ming, Kilicoglu et al., *Assessing Citation Integrity in Biomedical
   Publications: Corpus Annotation and NLP Models*, Bioinformatics 2024.
3. Walters and Wilder, *Fabrication and errors in the bibliographic citations generated
   by ChatGPT*, Scientific Reports 2023.
4. Priem, Piwowar and Orr, *OpenAlex: A fully-open index of scholarly works, authors,
   venues, institutions, and concepts*, 2022.
5. Laurer, van Atteveldt, Casas and Welbers, *Less Annotating, More Classifying:
   Addressing the Data Scarcity Issue of Supervised Machine Learning with Deep Transfer
   Learning and BERT-NLI*, Political Analysis 2024.
6. Tang, Laban and Durrett, *MiniCheck: Efficient Fact-Checking of LLMs on Grounding
   Documents*, EMNLP 2024.

## License

Apache License 2.0. Copyright 2026 Sanika Najan. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
