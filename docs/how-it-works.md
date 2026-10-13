# How Citeright works

A citation can fail in four ways, and each needs a different check:

1. **The reference does not exist.** Language models invent plausible references:
   real-sounding titles, real author names, DOIs with a real publisher prefix.
2. **The reference exists but is written wrongly.** A wrong year, a wrong first author,
   or a DOI that belongs to another paper. Readers who follow it reach the wrong work.
3. **The work has been retracted.** A retraction notice is published, but the paper
   keeps being cited.
4. **The work does not say what the sentence claims.** The reference is real and
   correct, but the cited paper does not back the sentence: it says something else,
   says less, or says the opposite.

Citeright checks all four for every reference and every citing sentence of a draft.

```text
draft ──▶ citing sentences + reference keys ──────────────────────────────┐
                                                                           ▼
bibliography ──▶ references ──▶ resolve (doi.org, Crossref, OpenAlex, arXiv) ──▶ status
                                         │ abstract or your full text
                                         ▼
                        support check: best passages ──▶ checker ──▶ calibrated verdict
```

## 1. Reading the draft

`citeright.citations` finds every sentence that cites something and the keys it cites,
in LaTeX (`\cite`, `\citep`, `\citet`, `\parencite`, ...), Pandoc Markdown (`[@key]`,
`@key`), numbered (`[1]`, `[2-4]`) and author-year (`(Smith et al., 2020)`) styles.
Citation markers are removed to get the claim each sentence makes.

References come from a BibTeX file (`citeright.bibtex`, a small dependency-free reader
that handles nested braces, `@string` macros, `#` concatenation, LaTeX accents and
cut-off entries), a `thebibliography` block, or a plain-text reference list in IEEE,
APA, ACL or Vancouver style.

## 2. Resolving references

For each reference (`citeright.resolve`):

1. **Identifiers.** A DOI is first checked against the DOI handle system
   (`doi.org/api/handles`), which says definitively whether the DOI is registered at
   all, whichever agency registered it. A registered DOI is looked up in OpenAlex and
   Crossref; an arXiv id on arXiv.
2. **Search.** If the identifiers do not lead to a confident match, Crossref's
   bibliographic search is tried, then OpenAlex's title search (OpenAlex gives
   anonymous users a daily search budget, so it is the fallback), then arXiv's.
3. **Match.** Each candidate is compared with the reference field by field: title
   similarity (robust to case, accents, LaTeX, typos and dropped subtitles), whether the
   first author matches (surnames compared without particles such as "de" or "van" and
   without hyphens, which sources write differently), author overlap, year difference, DOI and arXiv equality, venue
   similarity. A logistic model turns the comparison into P(this record is the work the
   reference means). The model and its decision threshold are fitted on real references
   with injected faults (`citeright bench match`), not set by hand.
4. **Decide.**
   1. `verified`: the best match agrees on every field that was given;
   2. `mismatch`: the work exists, but a field is wrong; the issue says which, and
      what the record says instead;
   3. `not_found`: nothing matches; with a DOI that does not exist, a likely
      fabrication;
   4. `retracted`: the matched work carries a retraction notice (Crossref's
      Retraction Watch data, or OpenAlex's retraction flag); corrections and
      expressions of concern are listed as issues.

When a match's year is far from the reference's, the search continues: the same work
often exists as a preprint and a later reprint, and the reference may mean either.

The text a citing sentence is checked against is the record's abstract. OpenAlex and
Crossref often lack abstracts for some publishers, so Citeright also asks Europe PMC
(by DOI) and arXiv (for preprints).

Every HTTP response is cached on disk (`~/.cache/citeright`, 30 days), so a second
check of the same draft makes no requests.

## 3. Checking that the cited work backs the sentence

`citeright.support` checks each (sentence, cited work) pair against the cited work's
abstract, or against its full text if you provide it (`--texts`):

1. **Find the passage.** The text is split into sentences and windows of up to three
   sentences; BM25 ranks them against the claim and the top six go on.
2. **Score it.** A checker gives P(the passage entails the claim) and P(it contradicts
   the claim) for each window. The default is an NLI cross-encoder (DeBERTa-v3 large,
   trained on MNLI, FEVER, ANLI, LingNLI and WANLI). MiniCheck, embedding similarity and
   word overlap are available for comparison.
3. **Calibrate.** A multinomial logistic model maps the strongest window's scores and
   simple coverage features (how much of the claim's vocabulary and numbers appear in
   the cited text) to calibrated probabilities of *supports*, *contradicts* and
   *not found*. It is fitted on two expert-labelled datasets: SciFact (claims against
   abstracts) and Citation-Integrity (real biomedical citing sentences against the full
   text of the cited paper).

The verdict is the most probable label; the report also gives P(supports) and the
passage that decided it, so a reader can see why.

## Limitations

1. An abstract often does not contain the specific result a sentence cites. Without the
   full text, *not found* means "not in the abstract", which is why the report says
   which text was checked.
2. Metadata sources disagree and have gaps. Works without a DOI, books, and very recent
   preprints are harder to resolve, and a `not_found` for those deserves a manual look.
3. The support checker reads English. Its calibration is fitted on biomedical and
   general scientific text; other fields may need refitting (`citeright bench support`
   shows how).
4. Retraction data is as complete as Crossref's and OpenAlex's.
