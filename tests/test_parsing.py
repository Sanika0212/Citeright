import pytest

from citeright.bibtex import parse_bibtex, split_authors
from citeright.citations import (
    extract_citations,
    match_author_year,
    parse_reference_list,
    parse_reference_string,
    strip_markers,
)
from citeright.model import Reference
from citeright.text import (
    delatex,
    normalize,
    normalize_arxiv,
    normalize_doi,
    sentences,
    surname,
    title_similarity,
)

BIB = r"""
@string{neurips = "Advances in Neural Information Processing Systems"}
@comment{ignored {entirely}}
@inproceedings{vaswani2017,
  title     = {Attention Is All You Need},
  author    = {Vaswani, Ashish and Shazeer, Noam and others},
  booktitle = neurips,
  year      = 2017,
  eprint    = {1706.03762},
  archivePrefix = {arXiv},
}
@article{lecun2015,
  title = "Deep {L}earning",
  author = "Yann LeCun and Yoshua Bengio and Geoffrey Hinton",
  journal = {Nature}, year = {2015}, doi = {https://doi.org/10.1038/NATURE14539},
}
@article{godel, title = {{\"U}ber formal unentscheidbare S{\"a}tze}, author = {G{\"o}del, Kurt},
  year = {1931}, journal = "Monatshefte f" # "{\"u}r Mathematik"}
@misc{cutoff, title = {Truncated entry
"""


def test_bibtex_entries_fields_and_macros() -> None:
    refs = {r.key: r for r in parse_bibtex(BIB)}
    assert set(refs) == {"vaswani2017", "lecun2015", "godel", "cutoff"}
    v = refs["vaswani2017"]
    assert v.title == "Attention Is All You Need"
    assert v.authors == ["Vaswani, Ashish", "Shazeer, Noam"]
    assert v.year == 2017
    assert v.arxiv == "1706.03762"
    assert v.venue.startswith("Advances in Neural")
    assert refs["lecun2015"].doi == "10.1038/nature14539"
    assert refs["lecun2015"].surnames == ["lecun", "bengio", "hinton"]
    assert refs["godel"].title == "Über formal unentscheidbare Sätze"
    assert refs["godel"].venue == "Monatshefte für Mathematik"
    assert refs["godel"].surnames == ["godel"]


def test_text_helpers() -> None:
    assert delatex(r"Schr\"{o}dinger's \emph{cat} -- \textbf{alive}") == "Schrödinger's cat - alive"
    assert delatex(r"Erd\H{o}s and \L{}ukasiewicz") == "Erdős and Łukasiewicz"
    assert normalize("Deep  Learning: A Review!") == "deep learning a review"
    assert surname("Ludwig van Beethoven") == "beethoven"
    assert surname("Andrielli Morais de Oliveira") == surname("Oliveira, A. M. d.") == "oliveira"
    assert surname("Asmae El Hassouni") == surname("El-Hassouni, A.") == "hassouni"
    assert surname("Quoc V. Le") == surname("Le, Quoc") == "le"
    assert surname("He K") == "he"
    assert surname("A. Vaswani") == "vaswani"
    assert surname("") == ""
    assert normalize_doi("doi:10.1000/XYZ.") == "10.1000/xyz"
    assert normalize_doi("not a doi") is None
    assert normalize_doi(None) is None
    assert normalize_arxiv("arXiv:1706.03762v5") == "1706.03762"
    assert normalize_arxiv("https://arxiv.org/abs/hep-th/9901001") == "hep-th/9901001"
    assert normalize_arxiv("2101.00001") == "2101.00001"
    assert normalize_arxiv(None) is None
    assert split_authors("A and B and others") == ["A", "B"]


def test_title_similarity() -> None:
    assert title_similarity("Attention is all you need", "Attention Is All You Need.") == 1.0
    assert title_similarity("Attention is all you need", "Attention is all you nede") > 0.9
    assert title_similarity("BERT", "") == 0.0
    long = "Deep residual learning for image recognition"
    assert title_similarity(long, long + ": a study of depth") >= 0.95
    assert title_similarity(long, "Generative adversarial networks") < 0.5


def test_sentences_keep_abbreviations() -> None:
    text = "Smith et al. showed this in Fig. 2. The effect was 2.5 times larger. Next? Yes!"
    assert sentences(text) == [
        "Smith et al. showed this in Fig. 2.",
        "The effect was 2.5 times larger.",
        "Next?",
        "Yes!",
    ]
    assert sentences("  ") == []


def test_latex_citations() -> None:
    tex = r"""
\section{Intro}
Transformers replaced recurrence \cite{vaswani2017}. % a comment \cite{hidden}
Deep nets work well~\citep[p.~3]{lecun2015, he2016} and scale \citet{kaplan2020}.

\begin{equation} x = \cite{inmath} \end{equation}
No citation here.
"""
    cites = extract_citations(tex)
    assert [c.keys for c in cites] == [["vaswani2017"], ["lecun2015", "he2016", "kaplan2020"]]
    assert cites[0].claim == "Transformers replaced recurrence."
    assert cites[1].line == 4


def test_pandoc_and_numeric_citations() -> None:
    md = "LLMs hallucinate references [@walters2023; @alkaissi2023, p. 4]. See @ji2023 too.\n"
    assert extract_citations(md)[0].keys == ["walters2023", "alkaissi2023"]
    assert extract_citations(md)[1].keys == ["ji2023"]
    assert strip_markers("Mail me at a@b.org [@x].") == "Mail me at a@b.org."
    numeric = (
        "Results improved [1, 3-5]. Unrelated [2].\n\n## References\n[1] A. B. Title one. 2020."
    )
    cites = extract_citations(numeric)
    assert cites[0].keys == ["1", "3", "4", "5"]
    assert cites[1].keys == ["2"]


def test_author_year_citations() -> None:
    refs = [
        Reference("a", "Deep learning", ["LeCun, Yann"], 2015),
        Reference("b", "Other", ["Smith, J"], 2020),
        Reference("c", "Another", ["Smith, K"], 2020),
    ]
    cites = extract_citations("Deep nets learn features (LeCun et al., 2015).", style="author-year")
    assert cites[0].keys == ["lecun:2015"]
    assert match_author_year("lecun:2015", refs) is refs[0]
    assert match_author_year("smith:2020", refs) is None  # ambiguous


@pytest.mark.parametrize(
    ("raw", "title", "first", "year"),
    [
        (
            'A. Vaswani, N. Shazeer, and N. Parmar, "Attention is all you need," in Proc. '
            "NeurIPS, 2017, pp. 5998-6008.",
            "Attention is all you need",
            "vaswani",
            2017,
        ),
        (
            "LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. Nature, 521, 436-444. "
            "https://doi.org/10.1038/nature14539",
            "Deep learning",
            "lecun",
            2015,
        ),
        (
            "Ashish Vaswani and Noam Shazeer. 2017. Attention is all you need. In NeurIPS.",
            "Attention is all you need",
            "vaswani",
            2017,
        ),
        (
            "He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. "
            "Proc CVPR. 2016:770-8.",
            "Deep residual learning for image recognition",
            "he",
            2016,
        ),
    ],
)
def test_reference_strings(raw: str, title: str, first: str, year: int) -> None:
    ref = parse_reference_string("k", raw)
    assert ref.title == title
    assert ref.surnames[0] == first
    assert ref.year == year


def test_reference_list_numbered_and_ids() -> None:
    text = (
        "Body text [1].\n\nReferences\n"
        '[1] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," Nature, 2015. '
        "doi:10.1038/nature14539\n"
        '[2] A. Vaswani et al., "Attention is all you need," arXiv:1706.03762, 2017.\n'
    )
    refs = parse_reference_list(text)
    assert [r.key for r in refs] == ["1", "2"]
    assert refs[0].doi == "10.1038/nature14539"
    assert refs[1].arxiv == "1706.03762"
    assert refs[1].surnames == ["vaswani"]
