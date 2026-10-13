"""Build every README figure.

    uv run --extra models --extra viz python scripts/make_figures.py

Inputs: ``runs/match/results.json`` and ``runs/support/results.json`` (from
``citeright bench``) for the charts; the animation and the report screenshot come from
checking ``examples/draft.md`` (network and the NLI model needed; ``--skip-demo``
skips them). Outputs go to ``docs/assets/`` as ``<name>-light.*`` and ``<name>-dark.*``.

Statuses use fixed colours with an icon and a label, never colour alone: verified ✓,
mismatch !, not found ?, retracted ⊘.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

ASSETS = Path("docs/assets")
DPI = 200
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"
CHROME = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")


@dataclass(frozen=True)
class Theme:
    name: str
    surface: str
    raised: str
    ink: str
    ink2: str
    muted: str
    grid: str
    axis: str
    neutral: str
    series: tuple[str, ...]
    good: str
    warn: str
    bad: str


THEMES = {
    "light": Theme(
        "light",
        "#fcfcfb",
        "#ffffff",
        "#0b0b0b",
        "#52514e",
        "#898781",
        "#e1e0d9",
        "#c3c2b7",
        "#a8a79f",
        ("#2a78d6", "#eb6834", "#1baf7a", "#eda100"),
        "#0ca30c",
        "#c98500",
        "#d03b3b",
    ),
    "dark": Theme(
        "dark",
        "#1a1a19",
        "#222220",
        "#ffffff",
        "#c3c2b7",
        "#898781",
        "#2c2c2a",
        "#383835",
        "#64635e",
        ("#3987e5", "#d95926", "#199e70", "#c98500"),
        "#0ca30c",
        "#eda100",
        "#e05252",
    ),
}
STATUS = {
    "verified": ("✓", "Verified", "good"),
    "mismatch": ("!", "Mismatch", "warn"),
    "not_found": ("?", "Not found", "bad"),
    "retracted": ("⊘", "Retracted", "bad"),
    "unchecked": ("…", "Unchecked", "muted"),
}
VERDICT = {
    "supports": ("✓", "supports", "good"),
    "contradicts": ("✗", "contradicts", "bad"),
    "not_found": ("?", "not in the cited work", "warn"),
    "no_text": ("–", "no text", "muted"),
}
FAULTS = {
    "correct": "Correct reference",
    "typo": "Typo in the title",
    "wrong_year": "Year off by 2 to 4",
    "wrong_author": "Wrong first author",
    "doi_swap": "DOI of another paper",
    "fake_doi": "Invented DOI",
    "fabricated": "Fabricated reference",
    "retracted": "Retracted paper",
}
METHODS = {
    "Citeright": ("Citeright", 0),
    "rules": ("Hand-set rules", 1),
    "title search": ("Title search only", 2),
    "doi only": ("DOI check only", 3),
}


def use_theme(mode: str) -> Theme:
    theme = THEMES[mode]
    plt.rcParams.update(
        {
            "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9.5,
            "figure.facecolor": theme.surface,
            "axes.facecolor": theme.surface,
            "savefig.facecolor": theme.surface,
            "text.color": theme.ink,
            "axes.labelcolor": theme.ink2,
            "axes.edgecolor": theme.axis,
            "xtick.color": theme.muted,
            "ytick.color": theme.muted,
            "xtick.labelcolor": theme.ink2,
            "ytick.labelcolor": theme.ink2,
            "grid.color": theme.grid,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )
    return theme


def _tone(theme: Theme, tone: str) -> str:
    return {"good": theme.good, "warn": theme.warn, "bad": theme.bad, "muted": theme.muted}[tone]


def _title(fig: Figure, title: str, subtitle: str, theme: Theme) -> None:
    fig.text(0.012, 0.975, title, ha="left", va="top", fontsize=12.5, weight="bold")
    fig.text(
        0.012,
        0.975 - 0.32 / fig.get_figheight(),
        subtitle,
        ha="left",
        va="top",
        fontsize=9,
        color=theme.ink2,
    )


def _save(fig: Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(path)
    return path


# ------------------------------------------------------------------------ banner


def banner(mode: str) -> str:
    t = THEMES[mode]
    width, height = 1200, 260
    cards = [
        ("✓", "verified", t.good),
        ("!", "wrong DOI", t.warn),
        ("?", "not found", t.bad),
        ("⊘", "retracted", t.bad),
    ]
    shapes = []
    x0, y0 = 640, 54
    shapes.append(
        f'<rect x="{x0}" y="{y0}" width="250" height="150" rx="10" fill="{t.raised}" '
        f'stroke="{t.axis}" stroke-width="1.5"/>'
    )
    for i in range(6):
        w = 210 - (i % 3) * 40
        shapes.append(
            f'<rect x="{x0 + 18}" y="{y0 + 20 + i * 20}" width="{w}" height="7" rx="3.5" '
            f'fill="{t.ink2}" fill-opacity="0.28"/>'
        )
    for i, (icon, label, color) in enumerate(cards):
        y = y0 + i * 38
        shapes.append(
            f'<path d="M {x0 + 250} {y0 + 30 + i * 28} C {x0 + 300} {y0 + 30 + i * 28}, '
            f'{x0 + 300} {y + 16}, {x0 + 340} {y + 16}" fill="none" stroke="{color}" '
            f'stroke-width="2"/>'
        )
        shapes.append(
            f'<rect x="{x0 + 340}" y="{y}" width="190" height="32" rx="8" fill="{color}" '
            f'fill-opacity="0.14" stroke="{color}" stroke-width="1.5"/>'
        )
        shapes.append(
            f'<text x="{x0 + 354}" y="{y + 22}" font-family="{FONT}" font-size="16" '
            f'font-weight="700" fill="{color}">{icon}</text>'
        )
        shapes.append(
            f'<text x="{x0 + 380}" y="{y + 21}" font-family="{FONT}" font-size="14" '
            f'fill="{t.ink}">{label}</text>'
        )
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Citeright: check that every citation is real, correct, not retracted, and says what you cite it for">
  <rect width="{width}" height="{height}" rx="18" fill="{t.surface}"/>
  <text x="56" y="118" font-family="{FONT}" font-size="64" font-weight="700" fill="{t.ink}">Citeright</text>
  <text x="58" y="162" font-family="{FONT}" font-size="22" fill="{t.ink2}">Is every reference real, correct and not retracted,</text>
  <text x="58" y="192" font-family="{FONT}" font-size="22" fill="{t.ink2}">and does it say what you cite it for?</text>
  {"".join(shapes)}
</svg>
"""


# ----------------------------------------------------------------------- charts


def plot_match(summary: dict[str, Any], out: Path, mode: str) -> Path:
    theme = use_theme(mode)
    results = summary["results"]
    faults = [f for f in FAULTS if any(f in r["by_fault"] for r in results.values())]
    fig = plt.figure(figsize=(7.6, 0.6 * len(faults) + 1.9))
    _title(
        fig,
        "Which reference problems does each check catch?",
        f"{summary['test']} held-out references from OpenAlex reference lists, each with at "
        "most one planted fault (count in brackets).\nShare given the right status (verified, "
        "mismatch, not found or retracted).",
        theme,
    )
    top = 1 - 1.0 / fig.get_figheight()
    bottom = 1.0 / fig.get_figheight()
    ax = fig.add_axes((0.27, bottom, 0.7, top - bottom))
    ax.set_xlim(-0.02, 1.04)
    ax.set_ylim(-0.6, len(faults) - 0.4)
    counts = summary.get("faults", {})
    labels = [f"{FAULTS[f]} ({counts[f]})" if f in counts else FAULTS[f] for f in faults]
    ax.set_yticks(range(len(faults)), labels[::-1])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.grid(True, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(length=0, pad=6)
    for name, (label, slot) in reversed(list(METHODS.items())):
        r = results.get(name)
        if r is None:
            continue
        xs = [r["by_fault"].get(f, float("nan")) for f in faults][::-1]
        ours = name == "Citeright"
        dodge = 0.27 - 0.18 * slot  # each method on its own line within the row
        ax.scatter(
            xs,
            [y + dodge for y in range(len(faults))],
            s=64 if ours else 40,
            color=theme.series[slot],
            zorder=4 if ours else 3,
            edgecolors=theme.surface,
            linewidths=1.5,
            label=f"{label} ({r['accuracy']:.0%} overall)",
        )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[::-1], labels[::-1], loc="lower center", ncol=2, bbox_to_anchor=(0.6, 0.0))
    return _save(fig, out)


CHECKERS = {
    "nli": "NLI (DeBERTa-v3 large)",
    "minicheck": "MiniCheck (RoBERTa large)",
    "embedding": "Embedding similarity",
    "lexical": "Word overlap",
}
DATASETS = {
    "scifact-validation": "SciFact (abstracts)",
    "citint-test": "Citation-Integrity (full text)",
}


def plot_support(results: dict[str, Any], out: Path, mode: str) -> Path:
    theme = use_theme(mode)
    checkers = [c for c in CHECKERS if c in results]
    fig, axes = plt.subplots(
        1, len(DATASETS), figsize=(7.6, 0.5 * len(checkers) + 2.0), sharey=True
    )
    _title(
        fig,
        "Does the cited work back the sentence?",
        "Area under the ROC curve for telling citing sentences the cited work supports from "
        "those it does not, on\nheld-out expert labels (50% = chance). Rings: the raw "
        "checker score without calibration.",
        theme,
    )
    for ax, (key, label) in zip(axes, DATASETS.items(), strict=True):
        for y, c in enumerate(reversed(checkers)):
            r = results[c][key]
            ours = c == "nli"
            color = theme.series[0] if ours else theme.neutral
            ax.plot([0.5, r["auroc"]], [y, y], color=color, lw=2, solid_capstyle="butt")
            ax.scatter(
                [r["auroc"]],
                [y],
                s=70,
                color=color,
                zorder=3,
                edgecolors=theme.surface,
                linewidths=1.5,
            )
            ax.scatter(
                [r["raw_auroc"]],
                [y],
                s=46,
                facecolor=theme.surface,
                edgecolor=color,
                linewidths=1.5,
                zorder=4,
            )
            ax.text(
                max(r["auroc"], r["raw_auroc"]) + 0.02,
                y,
                f"{r['auroc']:.0%}",
                va="center",
                fontsize=9,
                color=theme.ink if ours else theme.ink2,
                weight="bold" if ours else "normal",
            )
        ax.set_yticks(range(len(checkers)), [CHECKERS[c] for c in reversed(checkers)])
        ax.set_xlim(0.45, 1.05)
        ax.set_xticks([0.5, 0.75, 1.0], ["50%", "75%", "100%"])
        ax.set_title(
            f"{label}\n{results[checkers[0]][key]['n']} claims",
            fontsize=9.5,
            loc="left",
            color=theme.ink,
        )
        ax.grid(True, axis="x")
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=5)
    fig.subplots_adjust(
        left=0.24, right=0.98, top=1 - 1.45 / fig.get_figheight(), bottom=0.1, wspace=0.12
    )
    return _save(fig, out)


# -------------------------------------------------------------------- animation


def animate_check(report: dict[str, Any], draft: str, out: Path, mode: str) -> Path:
    """Sentences on the left, references on the right; each check lands one by one."""
    theme = use_theme(mode)
    checks = report["checks"]
    resolutions = report["resolutions"]
    claims: list[str] = []
    for c in checks:
        if c["citation"]["claim"] not in claims:
            claims.append(c["citation"]["claim"])
    keys = [r["reference"]["key"] for r in resolutions]
    fig = plt.figure(figsize=(7.6, 4.6))
    _title(
        fig,
        f"Checking {draft}",
        "Each reference is looked up, then each citing sentence is checked against the cited work.",
        theme,
    )
    ax = fig.add_axes((0.0, 0.0, 1.0, 0.86))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    rows = max(len(claims), len(keys))
    step = 88 / max(rows, 1)
    cy = {c: 92 - i * step for i, c in enumerate(claims)}
    ky = {k: 92 - i * step for i, k in enumerate(keys)}
    for c, y in cy.items():
        text = textwrap.shorten(c, 62, placeholder="…")
        ax.add_patch(
            FancyBboxPatch(
                (1.5, y - 4.2),
                50,
                8.4,
                boxstyle="round,pad=0,rounding_size=1.2",
                facecolor=theme.raised,
                edgecolor=theme.axis,
                lw=0.8,
            )
        )
        ax.text(3, y, text, fontsize=7.6, va="center", color=theme.ink)
    status_art: dict[str, Any] = {}
    for r in resolutions:
        k, y = r["reference"]["key"], ky[r["reference"]["key"]]
        ax.add_patch(
            FancyBboxPatch(
                (70, y - 4.2),
                28.5,
                8.4,
                boxstyle="round,pad=0,rounding_size=1.2",
                facecolor=theme.raised,
                edgecolor=theme.axis,
                lw=0.8,
            )
        )
        ax.text(72, y + 1.6, k, fontsize=7.6, va="center", color=theme.ink2, family="monospace")
        status_art[k] = ax.text(
            72, y - 1.9, "looking up…", fontsize=7.6, va="center", color=theme.muted
        )
    frames: list[tuple[str, int]] = [("ref", i) for i in range(len(resolutions))]
    frames += [("claim", i) for i in range(len(checks))]
    frames += [("hold", 0)] * 14
    edges: list[Line2D] = []

    def draw(n: int) -> list[Any]:
        kind, i = frames[n]
        if kind == "ref":
            r = resolutions[i]
            icon, label, tone = STATUS[r["status"]]
            art = status_art[r["reference"]["key"]]
            art.set_text(f"{icon} {label}")
            art.set_color(_tone(theme, tone))
            art.set_fontweight("bold")
        elif kind == "claim":
            c = checks[i]
            if c["key"] in ky:
                icon, label, tone = VERDICT[c["verdict"]]
                y0, y1 = cy[c["citation"]["claim"]], ky[c["key"]]
                color = _tone(theme, tone)
                (line,) = ax.plot(
                    [52, 69.5],
                    [y0, y1],
                    color=color,
                    lw=1.8,
                    ls="--" if c["verdict"] == "no_text" else "-",
                )
                edges.append(line)
                ax.text(
                    60.5,
                    (y0 + y1) / 2 + 1.6,
                    icon,
                    fontsize=9,
                    ha="center",
                    color=color,
                    weight="bold",
                )
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=500, blit=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=2), dpi=110)
    plt.close(fig)
    print(out)
    return out


def screenshot(page: str, name: str) -> None:
    if not CHROME.exists():
        print(f"skipping screenshots: {CHROME} not found")
        return
    with tempfile.TemporaryDirectory() as tmp:
        for mode in ("light", "dark"):
            themed = Path(tmp) / f"{name}-{mode}.html"
            themed.write_text(page.replace("<html", f'<html data-theme="{mode}"', 1))
            target = ASSETS / f"{name}-{mode}.png"
            subprocess.run(
                [
                    str(CHROME),
                    "--headless=new",
                    "--hide-scrollbars",
                    "--force-device-scale-factor=2",
                    "--window-size=1100,1300",
                    f"--screenshot={target.resolve()}",
                    themed.as_uri(),
                ],
                check=True,
                capture_output=True,
            )
            print(target)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", default="runs/match/results.json")
    parser.add_argument("--support", default="runs/support/results.json")
    parser.add_argument("--skip-demo", action="store_true")
    args = parser.parse_args()
    ASSETS.mkdir(parents=True, exist_ok=True)
    for mode in THEMES:
        (ASSETS / f"banner-{mode}.svg").write_text(banner(mode))
        print(ASSETS / f"banner-{mode}.svg")
    if Path(args.match).exists():
        summary = json.loads(Path(args.match).read_text())
        for mode in THEMES:
            plot_match(summary, ASSETS / f"match-{mode}.png", mode)
    if Path(args.support).exists():
        results = json.loads(Path(args.support).read_text())
        for mode in THEMES:
            plot_support(results, ASSETS / f"support-{mode}.png", mode)
    if args.skip_demo:
        return
    from citeright.check import check
    from citeright.report import render

    draft = Path("examples/draft.md")
    report = check(draft.read_text(), Path("examples/refs.bib").read_text())
    Path("examples/report.json").write_text(json.dumps(report.to_dict(), indent=1))
    for mode in THEMES:
        animate_check(report.to_dict(), draft.name, ASSETS / f"check-{mode}.gif", mode)
    screenshot(render(report, title=f"Citation check: {draft.name}"), "report")


if __name__ == "__main__":
    main()
