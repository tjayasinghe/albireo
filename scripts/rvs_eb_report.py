"""Build the report of the eclipsing-binary survey experiment from its collected runs.

Reads the runs that ``scripts/rvs_eb_run.py`` made, computes every quantity of the report
from their tables, and writes into ``docs/reports/gaia-rvs-eclipsing-binaries/``:

- ``figures/``: sixteen figures, each in a light and a dark rendering;
- ``tables/``: the binned table behind every figure, as CSV, and the population as a
  compressed CSV;
- ``numbers.json``: every number that the page quotes;
- ``manifests/``: the manifest of every run;

and the page itself, ``docs/reports/gaia-rvs-eclipsing-binaries.md``, rendered from
``PAGE`` with those numbers and nothing else. ``tests/test_survey_report.py`` renders the
page again from ``numbers.json`` and compares.

Usage::

    python scripts/rvs_eb_report.py                 # from the design run
    python scripts/rvs_eb_report.py --design pilot  # from the pilot, while developing

The definitions are constants at the top of the file. They were fixed on the pilot of 300
systems, before the design run.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import itertools
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ROOT = REPO / "internal" / "research" / "2026-10-07-rvs-eclipsing-todcor" / "runs"
QUERIES = ROOT.parent / "A_queries"
OUT = REPO / "docs" / "reports" / "gaia-rvs-eclipsing-binaries"
PAGE_PATH = REPO / "docs" / "reports" / "gaia-rvs-eclipsing-binaries.md"

# --- definitions, fixed on the pilot ----------------------------------------------------
CLASSES = ("K", "G", "F", "A")
CLASS_RANGES = {"K": "4000-5300 K", "G": "5300-6000 K", "F": "6000-7250 K", "A": "7250-10,000 K"}
EDGES = np.arange(6.0, 13.5 + 1e-9, 0.5)
WRONG_SIGMAS, WRONG_KMS = 5.0, 3.0  # an epoch velocity is wrong beyond both
RECOVERED = (0.05, 0.10)  # both semi-amplitudes within these fractions
MASS_LEVELS = ((0.01, 0.0045), (0.03, 0.013), (0.10, 0.045))  # mass precision: K precision
FALSE_ALARM = 0.01  # share of the null systems above the detection threshold
BRIGHT_RATIO = 0.2  # flux ratio above which a system is called double-lined by design
FWHM_KMS = 2.3548 * 11.67  # of the delivered line-spread function
HEADLINE = "classified"
# Scatter of one transit of single dwarfs of 5500 to 6500 K with `vbroad` below 20 km/s or
# absent, from the errors that Gaia DR3 publishes (A_gaia_archive.md, section 3) [km/s]:
# at integer magnitudes for the figure, the last an extrapolation, and as medians over
# ranges of G_RVS for the comparison with the simulated single stars.
GAIA_SCATTER = ((6.0, 0.19), (8.0, 0.36), (10.0, 1.07), (12.0, 5.4))
GAIA_TRANSITS = (
    (6, 6.0, 6.5, 0.20),
    (8, 7.5, 8.5, 0.36),
    (10, 9.5, 10.5, 1.07),
    (12, 11.5, 12.0, 4.31),
)
ZERO_POINT_KMS = 0.17  # the offset of a transit in the simulation

THEMES = {
    "light": {
        "surface": "#ffffff",
        "ink": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e6e5e1",
        "neutral": "#9a9892",
        "series": ("#2a78d6", "#eb6834", "#1baf7a"),
        "ramp": ("#eef4fc", "#b3d0f5", "#5598e7", "#256abf", "#0e3a73"),
        "diverging": ("#256abf", "#dddcd8", "#c93a39"),
        "soft": "#a9cbf3",
        "faint": "#d3d2cd",
    },
    "dark": {
        "surface": "#1e2129",
        "ink": "#f2f1ee",
        "muted": "#b5b3ad",
        "grid": "#3a3d47",
        "neutral": "#80828c",
        "series": ("#3987e5", "#d95926", "#199e70"),
        "ramp": ("#232836", "#1d3f72", "#2a69bd", "#5d9fee", "#c2dcfb"),
        "diverging": ("#5d9fee", "#4a4d57", "#e66767"),
        "soft": "#1f4f8f",
        "faint": "#4d505b",
    },
}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def wilson(k, n, z: float = 1.0):
    """Wilson score interval of a fraction ``k / n``: ``(p, low, high)`` at ``z`` sigma."""
    k, n = np.asarray(k, dtype=float), np.asarray(n, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(n > 0, k / n, np.nan)
        centre = (p + z * z / (2 * n)) / (1 + z * z / n)
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return p, centre - half, centre + half


def robust_sigma(x) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")


def weighted_quantile(x, q, w=None):
    """Quantiles of weighted values: the smallest value at which the cumulative weight
    reaches the share ``q`` of the total. A weight of two is the value twice."""
    x = np.asarray(x, dtype=float)
    w = np.ones_like(x) if w is None else np.asarray(w, dtype=float)
    keep = np.isfinite(x) & (w > 0)
    if not keep.any():
        return np.full(np.shape(q), np.nan)
    order = np.argsort(x[keep], kind="stable")
    xs, ws = x[keep][order], w[keep][order]
    cumulative = np.cumsum(ws)
    target = np.asarray(q, dtype=float) * cumulative[-1]
    index = np.searchsorted(cumulative, target * (1.0 - 1e-12), side="left")
    return xs[np.clip(index, 0, xs.size - 1)]


def fraction_by_bin(x, flag, edges, keep=None, weights=None):
    """Counts and fractions of ``flag`` in bins of ``x``, with Wilson intervals.

    Returns ``(n, k, p, low, high)``: the numbers of rows and of flagged rows in each bin,
    and the fraction with its interval at one sigma. With ``weights`` the fraction is the
    weighted one and its interval is that of the effective number of rows,
    ``(sum w)^2 / sum w^2``.
    """
    x, flag = np.asarray(x), np.asarray(flag, dtype=bool)
    keep = np.ones(x.shape, dtype=bool) if keep is None else keep
    index = np.digitize(x, edges) - 1
    bins = range(len(edges) - 1)
    n = np.array([np.sum(keep & (index == i)) for i in bins])
    k = np.array([np.sum(keep & flag & (index == i)) for i in bins])
    if weights is None:
        p, lo, hi = wilson(k, n)
        return n, k, p, lo, hi
    weights = np.asarray(weights, dtype=float)
    total = np.array([np.sum(weights[keep & (index == i)]) for i in bins])
    part = np.array([np.sum(weights[keep & flag & (index == i)]) for i in bins])
    square = np.array([np.sum(weights[keep & (index == i)] ** 2) for i in bins])
    with np.errstate(invalid="ignore", divide="ignore"):
        effective = np.where(square > 0, total**2 / square, 0.0)
        share = np.where(total > 0, part / total, np.nan)
    p, lo, hi = wilson(share * effective, effective)
    return n, k, p, lo, hi


# ---------------------------------------------------------------------------
# Loading and derived quantities
# ---------------------------------------------------------------------------


class Run:
    """A collected run: its tables, its population and what follows from them."""

    def __init__(self, name: str):
        from albireo.eclipsing import population_columns
        from albireo.population import read_population
        from albireo.survey import read_tables

        self.name = name
        self.dir = ROOT / name
        self.manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        self.declarations = list(self.manifest["declarations"])
        self.systems, self.epochs = read_tables(self.dir)
        self.population = read_population(self.dir / "population.json")
        self.cols = population_columns(self.population)
        self.names = np.array([s.name for s in self.population])
        self._derive()

    def decl(self, declaration: str) -> int:
        return self.declarations.index(declaration)

    def _derive(self) -> None:
        s, e, c = self.systems, self.epochs, self.cols
        i = s["system"]
        with np.errstate(invalid="ignore", divide="ignore"):
            for prefix in ("e_", "p_"):
                r1 = s[prefix + "k1"] / s["k1_true"] - 1.0
                r2 = s[prefix + "k2"] / s["k2_true"] - 1.0
                s[prefix + "r1"], s[prefix + "r2"] = r1, r2
                worst = np.fmax(np.abs(r1), np.abs(r2))
                worst = np.where(np.isfinite(r1) & np.isfinite(r2), worst, np.inf)
                s[prefix + "worst"] = worst
            s["z2"] = s["e_k2"] / s["e_k2_err"]
            s["z1"] = s["e_k1"] / s["e_k1_err"]
            s["s_r1"] = s["s_k1"] / s["k1_true"] - 1.0
            # Minimum masses: M1 sin^3 i ~ (K1 + K2)^2 K2 (1 - e^2)^1.5, M2 with K1.
            ecc = c["ecc"][i]
            shape = ((1.0 - s["e_ecc"] ** 2) / (1.0 - ecc**2)) ** 1.5
            total = ((s["e_k1"] + s["e_k2"]) / (s["k1_true"] + s["k2_true"])) ** 2
            m1 = total * s["e_k2"] / s["k2_true"] * shape - 1.0
            m2 = total * s["e_k1"] / s["k1_true"] * shape - 1.0
            s["m_worst"] = np.where(
                np.isfinite(m1) & np.isfinite(m2), np.fmax(np.abs(m1), np.abs(m2)), np.inf
            )
        for key in ("grvs", "light_ratio", "q", "period", "ecc", "vsini1", "vsini2", "teff1"):
            s[key] = c[key][i]
        s["klass"] = c["klass"][i]
        s["weight"], s["weight_cell"] = c["weight"][i], c["weight_cell"][i]
        # Epochs: errors of the velocities assigned by the ephemeris, and what became of them.
        j = e["system"]
        e["grvs"], e["klass"], e["light_ratio"] = c["grvs"][j], c["klass"][j], c["light_ratio"][j]
        e["vsini_max"] = np.maximum(c["vsini1"][j], c["vsini2"][j])
        e["vsini1"] = c["vsini1"][j]
        e["weight_cell"] = c["weight_cell"][j]
        out = e["in_eclipse"] == 0
        with np.errstate(invalid="ignore", divide="ignore"):
            e["err1"], e["err2"] = e["ve1"] - e["v1_true"], e["ve2"] - e["v2_true"]
            e["pull1"], e["pull2"] = e["err1"] / e["se1"], e["err2"] / e["se2"]
            # Against the velocity that was put into the spectrum: the orbit's plus the
            # zero point of the transit, which no quoted error contains.
            e["dev1"], e["dev2"] = e["err1"] - e["zero_point"], e["err2"] - e["zero_point"]
            e["pulls1"], e["pulls2"] = e["dev1"] / e["se1"], e["dev2"] / e["se2"]
            right = [
                ~(
                    (np.abs(e[f"err{k}"]) > WRONG_SIGMAS * e[f"se{k}"])
                    & (np.abs(e[f"err{k}"]) > WRONG_KMS)
                )
                & np.isfinite(e[f"err{k}"])
                for k in (1, 2)
            ]
        usable = out & (e["ge"] == 1)
        measured = out & np.isfinite(e["v1"]) & np.isfinite(e["v2"])
        e["usable"] = usable
        e["right1"], e["right2"] = usable & right[0], usable & right[1]
        e["wrong"] = usable & ~(right[0] & right[1])
        # 0 both right, 1 primary right only, 2 primary wrong (with the secondary right
        # or not), 3 flagged, 4 not measured, 5 in eclipse.
        fate = np.full(j.shape, 4)
        fate[measured & ~usable] = 3
        fate[usable] = 2
        fate[usable & right[0] & ~right[1]] = 1
        fate[usable & right[0] & right[1]] = 0
        fate[~out] = 5
        e["fate"] = fate
        e["separation_kms"] = np.abs(e["v1_true"] - e["v2_true"])

    def rows(self, declaration: str) -> dict[str, np.ndarray]:
        d = self.systems["decl"] == self.decl(declaration)
        return {key: value[d] for key, value in self.systems.items()}

    def epoch_rows(self, declaration: str) -> dict[str, np.ndarray]:
        d = self.epochs["decl"] == self.decl(declaration)
        return {key: value[d] for key, value in self.epochs.items()}


def detection_thresholds(null: Run | None) -> dict[str, float]:
    """The value of ``K2 / sigma(K2)`` that one null system in a hundred exceeds."""
    out = {}
    for declaration in ("injected", "classified", "catalogue"):
        if null is None or declaration not in null.declarations:
            out[declaration] = 5.0
            continue
        z = null.rows(declaration)["z2"]
        z = np.where(np.isfinite(z), z, 0.0)
        out[declaration] = float(np.quantile(z, 1.0 - FALSE_ALARM))
    return out


# ---------------------------------------------------------------------------
# Figure style
# ---------------------------------------------------------------------------


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def style(theme: dict):
    return {
        "figure.facecolor": theme["surface"],
        "axes.facecolor": theme["surface"],
        "savefig.facecolor": theme["surface"],
        "axes.edgecolor": theme["grid"],
        "axes.labelcolor": theme["muted"],
        "axes.titlecolor": theme["ink"],
        "text.color": theme["ink"],
        "xtick.color": theme["muted"],
        "ytick.color": theme["muted"],
        "grid.color": theme["grid"],
        "axes.grid": True,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 9.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "font.size": 9,
        "font.family": "DejaVu Sans",
        "lines.linewidth": 2.0,
        "lines.markersize": 5.5,
        "axes.axisbelow": True,
    }


def ramp(theme: dict):
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list("ramp", theme["ramp"])
    cmap.set_bad(theme["grid"])
    return cmap


def plain_log(axis, ticks=None, percent: bool = False) -> None:
    """Tick labels of a logarithmic axis as plain numbers, at the given ticks."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    if ticks is not None:
        axis.set_major_locator(FixedLocator(list(ticks)))
    scale = 100.0 if percent else 1.0
    axis.set_major_formatter(FuncFormatter(lambda value, _: f"{scale * value:g}"))
    axis.set_minor_formatter(NullFormatter())


def note(ax, text: str, theme: dict, **kwargs) -> None:
    ax.text(
        kwargs.pop("x", 0.02),
        kwargs.pop("y", 0.04),
        text,
        transform=ax.transAxes,
        color=theme["muted"],
        fontsize=7.5,
        **kwargs,
    )


def write_csv(name: str, header: list[str], rows: list[list]) -> None:
    path = OUT / "tables" / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        for row in rows:
            writer.writerow([f"{v:.8g}" if isinstance(v, (float, np.floating)) else v for v in row])


CENTRES = 0.5 * (EDGES[:-1] + EDGES[1:])
WIDE_EDGES = np.array([6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 13.5])


# ---------------------------------------------------------------------------
# Figures. Each returns the figure; its table and numbers are written once, by `tables`.
# ---------------------------------------------------------------------------


def fig01(data, theme):
    plt = _mpl()
    cols, cat = data["design"].cols, data["catalogue"]
    fig, axes = plt.subplots(2, 4, figsize=(11.0, 5.4), constrained_layout=True)
    blue, orange, _ = theme["series"]
    centres = 10 ** (0.2 * (np.arange(-4, 13) + 0.5))
    for ax, name in zip(axes[0], CLASSES, strict=True):
        sel = cols["klass"] == name
        pbin = np.floor(np.log10(cols["period"][sel]) / 0.2).astype(int)
        target = np.array([cat["period"].get((name, p), 0.0) for p in range(-4, 13)])
        ax.bar(
            centres,
            target / target.sum(),
            width=np.diff(10 ** (0.2 * np.arange(-4, 14))),
            color=theme["grid"],
            edgecolor=theme["neutral"],
            linewidth=0.6,
            label="DR3 catalogue",
        )
        for key, colour, label in (
            ("weight_cell", blue, "sample"),
            ("weight", orange, "sample, raked"),
        ):
            w = cols[key][sel]
            share = np.array([w[pbin == p].sum() for p in range(-4, 13)]) / w.sum()
            ax.step(
                np.r_[centres, centres[-1]],
                np.r_[share, share[-1]],
                where="mid",
                color=colour,
                linewidth=1.8,
                label=label,
            )
        ax.set_xscale("log")
        plain_log(ax.xaxis, [0.3, 1, 3, 10, 30, 100])
        ax.set_xlim(0.2, 200)
        ax.set_title(f"{name} primaries: period")
        ax.set_xlabel("period [d]")
    axes[0, 0].set_ylabel("share per 0.2 dex")
    axes[0, 0].legend(loc="upper right")
    ax = axes[1, 0]
    ax.bar(
        CENTRES,
        cat["by_bin"],
        width=0.42,
        color=theme["grid"],
        edgecolor=theme["neutral"],
        linewidth=0.6,
        label="DR3 catalogue",
    )
    ax.step(
        np.r_[EDGES[:-1], EDGES[-1]],
        np.r_[data["n_per_bin"], data["n_per_bin"][-1]],
        where="post",
        color=blue,
        label="design sample",
    )
    ax.set_yscale("log")
    ax.set_title("Systems per 0.5 mag")
    ax.set_xlabel(r"$G_{\rm RVS}$ [mag]")
    ax.set_ylabel("systems")
    ax.legend(loc="upper left")
    ax = axes[1, 1]
    bright = cols["grvs"] <= 12.0
    depth_edges = np.arange(0.0, 1.21, 0.1)
    target = cat["depth"]
    ax.bar(
        depth_edges[:-1] + 0.05,
        target / target.sum(),
        width=0.1,
        color=theme["grid"],
        edgecolor=theme["neutral"],
        linewidth=0.6,
    )
    hist, _ = np.histogram(
        np.clip(cols["depth_mag"][bright], 0, 1.19),
        depth_edges,
        weights=cols["weight_cell"][bright],
    )
    ax.step(
        np.r_[depth_edges[:-1], depth_edges[-1]],
        np.r_[hist, hist[-1]] / hist.sum(),
        where="post",
        color=blue,
    )
    ax.set_title("Depth of the deeper eclipse")
    ax.set_xlabel("depth in G [mag]")
    ax.set_ylabel("share per 0.1 mag")
    ax = axes[1, 2]
    q_edges = np.linspace(0.1, 1.0, 19)
    for sel, colour, label in (
        (np.ones(bright.shape, bool), blue, "all systems"),
        (cols["light_ratio"] >= 0.1, orange, "flux ratio above 0.1"),
    ):
        hist, _ = np.histogram(cols["q"][sel], q_edges, weights=cols["weight_cell"][sel])
        ax.step(
            np.r_[q_edges[:-1], q_edges[-1]],
            np.r_[hist, hist[-1]] / hist.sum(),
            where="post",
            color=colour,
            label=label,
        )
    ax.set_title("Mass ratio")
    ax.set_xlabel(r"$q = M_2 / M_1$")
    ax.set_ylabel("share per 0.05")
    ax.legend(loc="upper left")
    ax = axes[1, 3]
    grid = np.linspace(0.0, 0.7, 141)
    order = np.argsort(cols["rho"])
    cdf = np.cumsum(cols["weight_cell"][order]) / cols["weight_cell"].sum()
    ax.plot(cols["rho"][order], cdf, color=blue, label="sample")
    ax.plot(
        [0.2, 0.3, 0.4, 0.5, 0.6],
        [0.12, 0.34, 0.54, 0.78, 0.95],
        "o",
        color=theme["neutral"],
        label="ASAS-SN detached",
    )
    ax.set_xlim(grid[0], grid[-1])
    ax.set_title("Sum of the fractional radii")
    ax.set_xlabel(r"$(R_1 + R_2) / a$")
    ax.set_ylabel("cumulative share")
    ax.legend(loc="lower right")
    return fig


def fig02(data, theme):
    plt = _mpl()
    spectra = data["spectra"]
    fig, axes = plt.subplots(
        4, 3, figsize=(11.0, 7.4), sharex=True, sharey=True, constrained_layout=True
    )
    blue, orange, _ = theme["series"]
    for r, name in enumerate(CLASSES):
        for c, grvs in enumerate((8.0, 10.0, 12.0)):
            ax, item = axes[r, c], spectra[(name, grvs)]
            ax.plot(item["wave"] / 10.0, item["flux"], color=theme["neutral"], linewidth=0.7)
            ax.plot(
                item["model_wave"] / 10.0, 1.0 + item["components"][0], color=blue, linewidth=1.1
            )
            ax.plot(
                item["model_wave"] / 10.0, 1.0 + item["components"][1], color=orange, linewidth=1.1
            )
            ax.set_ylim(0.25, 1.35)
            ax.set_xlim(846.0, 870.0)
            if r == 0:
                ax.set_title(rf"$G_{{\rm RVS}}$ = {grvs:.0f}, S/N {item['snr']:.0f} per pixel")
            if c == 0:
                ax.set_ylabel(f"{name}: {item['label']}", fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("wavelength [nm]")
    axes[0, 0].plot([], [], color=blue, label="primary")
    axes[0, 0].plot([], [], color=orange, label="secondary")
    axes[0, 0].plot([], [], color=theme["neutral"], label="one transit")
    axes[0, 0].legend(loc="lower left", ncols=3, fontsize=7)
    return fig


def _heat(ax, grid, theme, xlabels, ylabels, fmt="{:.0f}"):
    image = ax.imshow(
        np.ma.masked_invalid(grid),
        aspect="auto",
        cmap=ramp(theme),
        vmin=0.0,
        vmax=1.0,
        origin="lower",
    )
    ax.grid(False)
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    for (r, c), value in np.ndenumerate(grid):
        if np.isfinite(value):
            dark_text = (value < 0.55) == (theme["surface"] == "#ffffff")
            ax.text(
                c,
                r,
                fmt.format(100 * value),
                ha="center",
                va="center",
                fontsize=6.5,
                color=THEMES["light"]["ink"] if dark_text else THEMES["dark"]["ink"],
            )
    return image


def fig03(data, theme):
    plt = _mpl()
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 5.2), constrained_layout=True)
    labels = [f"{x:.1f}" for x in EDGES[:-1]]
    for ax, key, title in (
        (axes[0], "recovery_all", "All systems: both semi-amplitudes within 10 percent [%]"),
        (
            axes[1],
            "recovery_bright",
            f"Flux ratio above {BRIGHT_RATIO}: both semi-amplitudes within 10 percent [%]",
        ),
    ):
        image = _heat(ax, data[key], theme, labels, list(CLASSES))
        ax.set_title(title)
        ax.set_ylabel("class of the primary")
    axes[1].set_xlabel(r"$G_{\rm RVS}$, lower edge of the 0.5 mag bin")
    fig.colorbar(image, ax=axes, shrink=0.8, pad=0.01, label="fraction recovered")
    return fig


def fig04(data, theme):
    plt = _mpl()
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.3), sharey=True, constrained_layout=True)
    table = data["epoch_error"]
    for ax, name in zip(axes, CLASSES, strict=True):
        for k, (colour, label) in enumerate(
            zip(theme["series"][:2], ("primary", "secondary"), strict=True)
        ):
            rows = table[(name, k + 1)]
            ax.fill_between(CENTRES, rows["lo"], rows["hi"], color=colour, alpha=0.18, linewidth=0)
            ax.plot(CENTRES, rows["sigma"], color=colour, marker="o", label=label)
            ax.plot(CENTRES, rows["pred"], color=colour, linestyle=(0, (2, 2)), linewidth=1.2)
        gx, gy = zip(*GAIA_SCATTER, strict=True)
        ax.plot(gx, gy, "s", color=theme["neutral"], markersize=5, label="Gaia DR3, single dwarfs")
        ax.set_yscale("log")
        plain_log(ax.yaxis, [0.1, 1, 10, 100])
        ax.set_ylim(0.1, 300)
        ax.set_title(f"{name} primaries")
        ax.set_xlabel(r"$G_{\rm RVS}$ [mag]")
    axes[0].set_ylabel("error of one transit [km/s]")
    axes[0].legend(loc="upper left")
    note(axes[1], "dashed: photon-limited prediction", theme)
    return fig


def fig05(data, theme):
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.5), constrained_layout=True)
    ax = axes[0]
    grid = np.linspace(-6, 6, 61)
    centre = 0.5 * (grid[1:] + grid[:-1])
    for k, (colour, label) in enumerate(
        zip(theme["series"][:2], ("primary", "secondary"), strict=True)
    ):
        ax.step(centre, data["pull_hist"][k], where="mid", color=colour, label=label)
    ax.plot(
        centre,
        np.exp(-0.5 * centre**2) / math.sqrt(2 * math.pi),
        color=theme["neutral"],
        linestyle=(0, (2, 2)),
        label="unit normal",
    )
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1.0)
    ax.set_title("Pulls of the epoch velocities, injected templates")
    ax.set_xlabel("(measured - injected) / quoted error")
    ax.set_ylabel("density")
    ax.legend(loc="upper right")
    ax = axes[1]
    for colour, declaration in zip(theme["series"], data["declarations"], strict=False):
        rows = data["pull_width"][declaration]
        ax.plot(rows["snr"], rows["width"], marker="o", color=colour, label=declaration)
    ax.axhline(1.0, color=theme["neutral"], linewidth=1.0)
    ax.set_xscale("log")
    plain_log(ax.xaxis, [2, 5, 10, 20, 50, 100])
    ax.set_title("Robust width of the pulls of the primary")
    ax.set_xlabel("S/N per pixel of the transit")
    ax.set_ylabel("width (1 = calibrated)")
    ax.legend(loc="upper left")
    return fig


def fig06(data, theme):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(11.0, 3.6), constrained_layout=True)
    shares = data["fate"]
    blue, orange, _ = theme["series"]
    colours = (blue, theme["soft"], orange, theme["neutral"], theme["faint"])
    labels = ("both right", "primary right only", "primary wrong", "flagged", "not measured")
    bottom = np.zeros(CENTRES.size)
    for k in range(5):
        ax.bar(
            CENTRES,
            shares[:, k],
            bottom=bottom,
            width=0.42,
            color=colours[k],
            edgecolor=theme["surface"],
            linewidth=1.0,
            label=labels[k],
        )
        bottom += shares[:, k]
    ax.set_ylim(0, 1)
    ax.set_xlim(EDGES[0], EDGES[-1])
    ax.set_title(f"What becomes of a transit out of eclipse ({HEADLINE} templates)")
    ax.set_xlabel(r"$G_{\rm RVS}$ [mag]")
    ax.set_ylabel("share of transits")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=5)
    return fig


def fig07(data, theme):
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.4), sharex=True, constrained_layout=True)
    for colour, (label, rows) in zip(theme["series"], data["separation"].items(), strict=True):
        axes[0].plot(rows["x"], rows["usable"], marker="o", color=colour, label=label)
        axes[1].plot(rows["x"], rows["wrong"], marker="o", color=colour, label=label)
    for ax, title, ylabel in (
        (axes[0], "Transits with two usable velocities", "share of transits out of eclipse"),
        (axes[1], "Usable transits with a wrong velocity", "share of usable transits"),
    ):
        ax.set_xscale("log")
        plain_log(ax.xaxis, [0.2, 0.5, 1, 2, 5])
        ax.set_title(title)
        ax.set_xlabel("line separation / line width")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.axvline(1.0, color=theme["neutral"], linewidth=1.0)
    axes[0].legend(loc="lower right", title="S/N per pixel")
    return fig


def fig08(data, theme):
    plt = _mpl()
    fig, axes = plt.subplots(
        2, 2, figsize=(11.0, 5.8), sharex=True, sharey=True, constrained_layout=True
    )
    ratios, mags = data["ratio_edges"], WIDE_EDGES
    for ax, name in zip(axes.ravel(), CLASSES, strict=True):
        grid = data["detection"][name]
        image = ax.pcolormesh(
            mags,
            ratios,
            np.ma.masked_invalid(grid),
            cmap=ramp(theme),
            vmin=0,
            vmax=1,
            edgecolors=theme["surface"],
            linewidth=0.8,
        )
        ax.grid(False)
        ax.set_yscale("log")
        plain_log(ax.yaxis, [0.001, 0.01, 0.1, 1])
        ax.axhline(BRIGHT_RATIO, color=theme["ink"], linewidth=1.0, linestyle=(0, (4, 2)))
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                if np.isfinite(grid[r, c]):
                    dark_text = (grid[r, c] < 0.55) == (theme["surface"] == "#ffffff")
                    ax.text(
                        0.5 * (mags[c] + mags[c + 1]),
                        math.sqrt(ratios[r] * ratios[r + 1]),
                        f"{100 * grid[r, c]:.0f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color=THEMES["light"]["ink"] if dark_text else THEMES["dark"]["ink"],
                    )
        ax.set_title(f"{name} primaries: secondary detected [%]")
    for ax in axes[-1]:
        ax.set_xlabel(r"$G_{\rm RVS}$ [mag]")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"flux ratio $F_2 / F_1$")
    fig.colorbar(image, ax=axes, shrink=0.8, pad=0.01, label="fraction detected")
    return fig


def fig09(data, theme):
    plt = _mpl()
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.2), sharey=True, constrained_layout=True)
    for ax, name in zip(axes, CLASSES, strict=True):
        for colour, declaration in zip(theme["series"], ("injected", "classified"), strict=False):
            rows = data["rotation"][(name, declaration)]
            ax.plot(rows["x"], rows["ratio"], marker="o", color=colour, label=declaration)
        ax.axhline(1.0, color=theme["neutral"], linewidth=1.0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        plain_log(ax.xaxis, [5, 10, 20, 50, 100, 200])
        plain_log(ax.yaxis, [0.5, 1, 2, 5, 10, 20])
        ax.set_ylim(0.5, 30)
        ax.set_title(f"{name} primaries")
        ax.set_xlabel(r"$v \sin i$ of the primary [km/s]")
    axes[0].set_ylabel("scatter / predicted error")
    axes[0].legend(loc="upper left")
    return fig


def fig10(data, theme):
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.5), constrained_layout=True)
    for ax, key, title, xlabel, lines in (
        (
            axes[0],
            "k_cdf",
            "Larger error of the two semi-amplitudes",
            "relative error [%]",
            (0.013, 0.045),
        ),
        (
            axes[1],
            "m_cdf",
            "Larger error of the two minimum masses",
            "relative error [%]",
            (0.03, 0.10),
        ),
    ):
        for colour, (label, (x, y)) in zip(theme["series"], data[key].items(), strict=True):
            ax.plot(x, y, color=colour, label=label)
        for value in lines:
            ax.axvline(value, color=theme["neutral"], linewidth=1.0, linestyle=(0, (2, 2)))
        ax.set_xscale("log")
        plain_log(ax.xaxis, [0.001, 0.01, 0.1, 1.0], percent=True)
        ax.set_xlim(1e-3, 1.0)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("share of all systems")
    axes[0].legend(loc="upper left", title=r"$G_{\rm RVS}$")
    note(axes[1], "dashed: 3 and 10 percent", theme, x=0.55)
    note(axes[0], "dashed: the precision that masses\nto 3 and 10 percent need", theme, x=0.5)
    return fig


def _paired(ax, x, series, theme, labels):
    # The series of one bin stand closer to each other than to those of the next bin.
    step = float(np.min(np.diff(x))) if len(x) > 1 else 1.0
    width = 0.3 * step / max(len(series) - 1, 1)
    for k, (colour, (label, rows)) in enumerate(zip(theme["series"], series.items(), strict=False)):
        offset = (k - 0.5 * (len(series) - 1)) * width
        yerr = np.clip(np.vstack([rows["p"] - rows["lo"], rows["hi"] - rows["p"]]), 0.0, None)
        ax.errorbar(
            np.asarray(x) + offset,
            rows["p"],
            yerr=yerr,
            fmt="o",
            color=colour,
            label=label,
            elinewidth=1.2,
            capsize=0,
        )
    ax.set_ylim(0, 1)
    if labels is not None:
        ax.set_xticks(x, labels)


def fig11(data, theme):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(11.0, 3.4), constrained_layout=True)
    _paired(ax, CENTRES, data["ephemeris"], theme, None)
    ax.set_title(
        f"Both semi-amplitudes within 10 percent: what the fit holds ({HEADLINE} templates)"
    )
    ax.set_xlabel(r"$G_{\rm RVS}$ [mag]")
    ax.set_ylabel("share of systems")
    ax.legend(loc="upper right")
    return fig


def fig12(data, theme):
    plt = _mpl()
    fig, axes = plt.subplots(1, 4, figsize=(11.0, 3.3), sharey=True, constrained_layout=True)
    centres = 0.5 * (WIDE_EDGES[:-1] + WIDE_EDGES[1:])
    for ax, name in zip(axes, CLASSES, strict=True):
        for colour, declaration in zip(theme["series"], data["declarations"], strict=False):
            rows = data["templates"][(name, declaration)]
            ax.fill_between(centres, rows["lo"], rows["hi"], color=colour, alpha=0.18, linewidth=0)
            ax.plot(centres, rows["p"], marker="o", color=colour, label=declaration)
        ax.set_ylim(0, 1)
        ax.set_xlim(WIDE_EDGES[0], WIDE_EDGES[-1])
        ax.set_title(f"{name} primaries")
        ax.set_xlabel(r"$G_{\rm RVS}$ [mag]")
    axes[0].set_ylabel(f"recovered, flux ratio above {BRIGHT_RATIO}")
    axes[0].legend(loc="lower left")
    return fig


def fig13(data, theme):
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.4), sharey=True, constrained_layout=True)
    rows = data["perturbations"]
    labels = [row["label"] for row in rows]
    y = np.arange(len(rows))[::-1]
    low, mid, high = theme["diverging"]
    for ax, key, title, xlabel in (
        (axes[0], "delta", "Change of the share recovered to 10 percent", "percentage points"),
        (axes[1], "scatter", "Change of the scatter of the primary's velocities", "percent"),
    ):
        values = np.array([row[key][0] for row in rows])
        errors = np.array(
            [[row[key][0] - row[key][1], row[key][2] - row[key][0]] for row in rows]
        ).T
        ax.barh(
            y,
            values,
            color=[
                high if (v < 0 and key == "delta") or (v > 0 and key == "scatter") else low
                for v in values
            ],
            height=0.55,
        )
        ax.errorbar(values, y, xerr=errors, fmt="none", ecolor=theme["ink"], elinewidth=1.2)
        ax.axvline(0.0, color=theme["neutral"], linewidth=1.0)
        ax.set_yticks(y, labels)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
    del mid
    return fig


def fig14(data, theme):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(11.0, 3.6), constrained_layout=True)
    for colour, declaration in zip(theme["series"], data["declarations"], strict=False):
        rows = data["function"][declaration]
        yerr = np.clip(np.vstack([rows["p"] - rows["lo"], rows["hi"] - rows["p"]]), 0.0, None)
        ax.errorbar(
            rows["x"],
            rows["p"],
            yerr=yerr,
            fmt="o",
            color=colour,
            elinewidth=1.0,
            label=declaration,
        )
        ax.plot(rows["grid"], rows["fit"], color=colour, linewidth=1.6)
    ax.set_xscale("log")
    plain_log(ax.xaxis, [1, 3, 10, 30, 100, 300, 1000, 3000])
    ax.set_ylim(0, 1)
    ax.set_xlim(1, 3000)
    ax.set_title(
        "Recovery to 10 percent against the predicted S/N of the secondary's semi-amplitude"
    )
    ax.set_xlabel(r"predicted $K_2 / \sigma(K_2)$")
    ax.set_ylabel("share of systems")
    ax.legend(loc="upper left")
    return fig


def fig15(data, theme):
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(11.0, 3.8), constrained_layout=True)
    rows = data["yield"]
    ax.plot(
        EDGES[1:],
        rows["candidates"],
        color=theme["neutral"],
        label="DR3 candidates, detached classes",
    )
    for colour, key, label in zip(
        theme["series"],
        ("orbit", "mass10", "mass3"),
        ("double-lined orbit", "masses to 10 percent", "masses to 3 percent"),
        strict=True,
    ):
        ax.fill_between(
            EDGES[1:], rows[key + "_lo"], rows[key + "_hi"], color=colour, alpha=0.2, linewidth=0
        )
        ax.plot(EDGES[1:], rows[key], color=colour, marker="o", label=label)
    ax.set_yscale("log")
    plain_log(ax.yaxis)
    ax.set_title(
        f"Systems brighter than a limit, weighted to the DR3 catalogue ({HEADLINE} templates)"
    )
    ax.set_xlabel(r"limiting $G_{\rm RVS}$ [mag]")
    ax.set_ylabel("cumulative number of systems")
    ax.legend(loc="upper left")
    return fig


def fig16(data, theme):
    plt = _mpl()
    cases = data["cases"]
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.2), constrained_layout=True)
    blue, orange, _ = theme["series"]
    for ax in axes.ravel()[len(cases) :]:
        ax.set_visible(False)
    for ax, case in zip(axes.ravel(), cases, strict=False):
        phase = np.linspace(0, 1, 400)
        ax.plot(phase, case["curve1"], color=blue, linewidth=1.2)
        ax.plot(phase, case["curve2"], color=orange, linewidth=1.2)
        good = case["usable"]
        for k, colour in ((1, blue), (2, orange)):
            ax.errorbar(
                case["phase"][good],
                case[f"v{k}"][good],
                yerr=case[f"s{k}"][good],
                fmt="o",
                color=colour,
                markersize=3.5,
                elinewidth=0.8,
            )
            ax.plot(
                case["phase"][~good],
                case[f"v{k}"][~good],
                "o",
                markerfacecolor="none",
                markeredgecolor=colour,
                markersize=3.5,
            )
        centre = case["gamma"]
        swing = max(np.abs(case["curve1"] - centre).max(), np.abs(case["curve2"] - centre).max())
        # The note stands below the lowest point drawn, and the legend above the highest.
        drawn = np.concatenate([case["v1"], case["v2"]])
        drawn = drawn[np.isfinite(drawn) & (np.abs(drawn - centre) < 3.0 * swing)]
        low = min(centre - 1.15 * swing, drawn.min() if drawn.size else centre)
        high = max(centre + 1.15 * swing, drawn.max() if drawn.size else centre)
        ax.set_ylim(low - 0.38 * (high - low), high + 0.2 * (high - low))
        ax.set_xlim(0, 1)
        ax.set_title(case["title"])
        note(ax, case["note"], theme, y=0.03, linespacing=1.35)
    for ax in axes[-1]:
        ax.set_xlabel("phase from the primary's eclipse")
    for ax in axes[:, 0]:
        ax.set_ylabel("velocity [km/s]")
    axes[0, 0].plot([], [], "o", color=blue, markersize=3.5, label="primary")
    axes[0, 0].plot([], [], "o", color=orange, markersize=3.5, label="secondary")
    axes[0, 0].legend(loc="upper right", ncols=2, fontsize=7)
    return fig


FIGURES = (
    ("fig01-sample", fig01),
    ("fig02-spectra", fig02),
    ("fig03-recovery", fig03),
    ("fig04-epoch-error", fig04),
    ("fig05-pulls", fig05),
    ("fig06-fate", fig06),
    ("fig07-separation", fig07),
    ("fig08-secondary", fig08),
    ("fig09-rotation", fig09),
    ("fig10-orbits", fig10),
    ("fig11-ephemeris", fig11),
    ("fig12-templates", fig12),
    ("fig13-perturbations", fig13),
    ("fig14-function", fig14),
    ("fig15-yield", fig15),
    ("fig16-cases", fig16),
)


# ---------------------------------------------------------------------------
# What the figures show: computed once, written as tables, quoted as numbers
# ---------------------------------------------------------------------------
#
# Two conventions hold throughout. A share that pools cells of the design (several
# magnitudes, or the four classes) is weighted by `weight_cell`, the number of catalogue
# candidates a system stands for, so that it describes the catalogue's mixture and not the
# flat design. A quantity of the estimator against the S/N, the separation of the lines or
# the rotation is not weighted. And an error is taken against the orbit's velocity
# (`err`), which contains the zero point of the transit, except where the calibration of
# the estimator itself is tested: there it is taken against the velocity that was put
# into the spectrum (`dev`).


def catalogue_tables() -> dict:
    """The DR3 catalogue's detached classes in the bins of the design (aggregate queries)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from rvs_eb_run import catalogue_counts

    counts, totals = catalogue_counts()
    period: dict[tuple[str, int], float] = {}
    by_bin = np.zeros(EDGES.size - 1)
    for (m, name, p), value in counts.items():
        period[(name, p)] = period.get((name, p), 0.0) + value
        by_bin[m] += value
    depth = np.zeros(12)
    with open(QUERIES / "q05c_eb_depth_joint_grvs_le12.csv", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(float(row["cls"])) in (1, 4, 7, 10) and row["dpbin"] not in ("", "NaN", "nan"):
                depth[min(int(float(row["dpbin"])), 11)] += float(row["n"])
    return {"period": period, "by_bin": by_bin, "depth": depth, "counts": counts, "totals": totals}


def representative_spectra(design: Run) -> dict:
    """One transit near quadrature of a typical pair of each class, at three magnitudes."""
    from albireo.gaia import RVS_DR4_EPOCH, rvs_snr_per_pixel, simulate_rvs_dataset
    from albireo.grids import C_KMS
    from albireo.operators import _gaussian_kernel_numpy
    from albireo.survey import SurveyConfig, _component, _context

    config = SurveyConfig()
    ctx = _context(tuple(config.libraries), float(config.v_max_kms), float(config.vsini_max_kms))
    cols = design.cols
    out, rows = {}, []
    for name in CLASSES:
        sel = (
            (cols["klass"] == name)
            & (cols["light_ratio"] > 0.35)
            & (cols["light_ratio"] < 0.7)
            & (cols["ecc"] == 0.0)
            & (cols["period"] > 1.5)
            & (cols["period"] < 4.0)
        )
        index = np.flatnonzero(sel)
        score = np.abs(cols["vsini1"][index] - np.median(cols["vsini1"][cols["klass"] == name]))
        system = design.population[int(index[np.argmin(score)])]
        epsilon = system.meta["limb_darkening"]
        components = [
            _component(ctx, system.teff1, system.logg1, system.mh, system.vsini1, epsilon[0]),
            _component(ctx, system.teff2, system.logg2, system.mh, system.vsini2, epsilon[1]),
        ]
        velocity = np.array([[system.gamma - system.k1], [system.gamma + system.k2]])
        fractions = np.asarray(system.light_fractions)
        wave = np.asarray(ctx.grid.wave)
        kernel = _gaussian_kernel_numpy(11.07 / ctx.grid.dv_kms)
        shown = []
        for component, v, fraction in zip(components, velocity[:, 0], fractions, strict=True):
            shifted = np.interp(wave / (1.0 + v / C_KMS), wave, component)
            shown.append(fraction * np.convolve(shifted, kernel, mode="same"))
        for grvs in (8.0, 10.0, 12.0):
            snr = float(rvs_snr_per_pixel(grvs))
            dataset, _ = simulate_rvs_dataset(
                components,
                ctx.grid,
                bjd=np.array([2457000.0]),
                light_fractions=fractions,
                velocities=velocity,
                snr=snr,
                product=RVS_DR4_EPOCH,
                library_resolving_power=ctx.libraries.libraries[0].resolving_power,
                seed=int(system.meta["seed"]),
            )
            epoch = dataset.epochs[0]
            out[(name, grvs)] = {
                "wave": np.asarray(epoch.wave),
                "flux": np.asarray(epoch.flux),
                "model_wave": wave,
                "components": shown,
                "snr": snr,
                "label": f"{system.teff1:.0f} + {system.teff2:.0f} K",
            }
            rows.append(
                [
                    name,
                    system.name,
                    grvs,
                    snr,
                    system.teff1,
                    system.teff2,
                    system.light_ratio,
                    system.vsini1,
                    system.vsini2,
                    system.period,
                    system.k1,
                    system.k2,
                ]
            )
    write_csv(
        "fig02-spectra",
        [
            "class",
            "system",
            "grvs",
            "snr_per_pixel",
            "teff1_K",
            "teff2_K",
            "flux_ratio",
            "vsini1_kms",
            "vsini2_kms",
            "period_d",
            "k1_kms",
            "k2_kms",
        ],
        rows,
    )
    return out


def _logistic(x, top, mid, width):
    return top / (1.0 + np.exp(-(np.log10(x) - mid) / width))


def fit_recovery(x, flag):
    """Maximum-likelihood ``top / (1 + exp(-(log10 x - mid) / width))`` to binary outcomes."""
    from scipy.optimize import minimize

    keep = np.isfinite(x) & (x > 0)
    x, flag = x[keep], flag[keep].astype(float)

    def cost(p):
        top, mid, width = 1.0 / (1.0 + math.exp(-p[0])), p[1], math.exp(p[2])
        prob = np.clip(_logistic(x, top, mid, width), 1e-9, 1 - 1e-9)
        return -np.sum(flag * np.log(prob) + (1 - flag) * np.log(1 - prob))

    best = minimize(
        cost,
        [2.0, 1.5, -1.0],
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": 4000},
    )
    return 1.0 / (1.0 + math.exp(-best.x[0])), float(best.x[1]), math.exp(best.x[2])


def pct(value, digits: int = 0) -> str:
    return "n/a" if not np.isfinite(value) else f"{100 * value:.{digits}f}"


def wshare(flag, keep, weights) -> float:
    """Weighted share of ``flag`` among the rows ``keep``."""
    total = float(np.sum(weights[keep]))
    return float(np.sum(weights[keep & flag]) / total) if total > 0.0 else float("nan")


def compute(design: Run, null: Run | None, variants: dict[str, Run], recorded: Run | None):
    data: dict = {"design": design, "declarations": design.declarations}
    numbers: dict[str, str] = {}
    cols = design.cols
    head = design.rows(HEADLINE)
    head_e = design.epoch_rows(HEADLINE)
    thresholds = detection_thresholds(null)
    rng = np.random.default_rng(1)

    # --- the sample --------------------------------------------------------------------
    cat = catalogue_tables()
    data["catalogue"] = cat
    data["n_per_bin"] = np.array(
        [np.sum((cols["grvs"] >= lo) & (cols["grvs"] < hi)) for lo, hi in itertools.pairwise(EDGES)]
    )
    w = cols["weight_cell"]
    summary = json.loads((ROOT / "population_summary.json").read_text(encoding="utf-8"))
    sb2 = cols["light_ratio"] >= 0.1
    depth = np.array([s.meta["depth_mag"] for s in design.population])
    deep = depth.max(axis=1) >= 0.2
    same = np.floor(depth.min(axis=1) / 0.1) == np.floor(depth.max(axis=1) / 0.1)
    everything = np.ones(w.shape, dtype=bool)
    quantiles = weighted_quantile(cols["period"], [0.16, 0.5, 0.84], w)
    bright_share = wshare(cols["light_ratio"] >= BRIGHT_RATIO, everything, w)
    numbers.update(
        n_design=f"{len(design.population):,}",
        per_cell=str(int(len(design.population) / (len(CLASSES) * (EDGES.size - 1)))),
        catalogue_all=f"{cat['totals']['all']:,.0f}",
        catalogue_classes=f"{sum(cat['counts'].values()):,.0f}",
        catalogue_outside_classes=pct(cat["totals"]["outside_classes"] / cat["totals"]["all"]),
        catalogue_outside_periods=pct(summary["share_outside_period_support"]),
        ess_cell=f"{summary['effective_size_cell']:,.0f}",
        ess_raked=f"{summary['effective_size_raked']:,.0f}",
        period_p16=f"{quantiles[0]:.2f}",
        period_p50=f"{quantiles[1]:.2f}",
        period_p84=f"{quantiles[2]:.2f}",
        q95=pct(wshare(cols["q"] > 0.95, sb2, w)),
        q90=pct(wshare(cols["q"] > 0.90, sb2, w)),
        q70=pct(wshare(cols["q"] > 0.70, sb2, w)),
        q95_all=pct(wshare(cols["q"] > 0.95, everything, w)),
        sb2_share=pct(bright_share),
        faint_share=pct(1.0 - bright_share),
        tenth_share=pct(wshare(cols["light_ratio"] < 0.1, everything, w)),
        equal_depth=pct(wshare(same, deep, w)),
        post_ms=pct(wshare(cols["eep1"] >= 454, everything, w)),
        ecc_small=pct(wshare(cols["ecc"] < 0.05, everything, w)),
        rho_median=f"{weighted_quantile(cols['rho'], [0.5], w)[0]:.2f}",
        depth_median=f"{weighted_quantile(cols['depth_mag'], [0.5], w)[0]:.2f}",
        transits_median=f"{weighted_quantile(cols['n_transits'], [0.5], w)[0]:.0f}",
        vsini_median=f"{weighted_quantile(cols['vsini1'], [0.5], w)[0]:.0f}",
    )
    for name in CLASSES:
        sel = cols["klass"] == name
        numbers[f"period_{name}"] = (
            f"{weighted_quantile(cols['period'][sel], [0.5], w[sel])[0]:.1f}"
        )
        numbers[f"post_ms_{name}"] = pct(wshare(cols["eep1"] >= 454, sel, w))
        numbers[f"catalogue_{name}"] = (
            f"{sum(v for (m, c, p), v in cat['counts'].items() if c == name):,.0f}"
        )
    rows = []
    for name in CLASSES:
        sel = cols["klass"] == name
        pbin = np.floor(np.log10(cols["period"][sel]) / 0.2).astype(int)
        total = sum(v for (c, p), v in cat["period"].items() if c == name)
        for p in range(-4, 13):
            rows.append(
                [
                    name,
                    10 ** (0.2 * p),
                    cat["period"].get((name, p), 0.0) / total,
                    float(
                        cols["weight_cell"][sel][pbin == p].sum() / cols["weight_cell"][sel].sum()
                    ),
                    float(
                        cols["weight"][sel][pbin == p].sum()
                        / max(cols["weight"][sel].sum(), 1e-300)
                    ),
                    int(np.sum(pbin == p)),
                ]
            )
    write_csv(
        "fig01-sample-periods",
        [
            "class",
            "period_low_d",
            "catalogue_share",
            "sample_share",
            "sample_share_raked",
            "n_sample",
        ],
        rows,
    )

    # --- spectra -----------------------------------------------------------------------
    data["spectra"] = representative_spectra(design)

    # --- recovery ----------------------------------------------------------------------
    # The maps are per cell and need no weight. The shares of a range of magnitudes pool
    # cells and are weighted to the catalogue.
    rows = []
    for key, base in (
        ("recovery_all", np.ones(head["grvs"].shape, bool)),
        ("recovery_bright", head["light_ratio"] >= BRIGHT_RATIO),
    ):
        grid = np.full((len(CLASSES), EDGES.size - 1), np.nan)
        for r, name in enumerate(CLASSES):
            n, _, p, _, _ = fraction_by_bin(
                head["grvs"], head["e_worst"] < RECOVERED[1], EDGES, base & (head["klass"] == name)
            )
            grid[r] = np.where(n >= 5, p, np.nan)
        data[key] = grid
    for declaration in design.declarations:
        d = design.rows(declaration)
        for name in CLASSES:
            for subset, base in (
                ("all", np.ones(d["grvs"].shape, bool)),
                ("flux ratio above 0.2", d["light_ratio"] >= BRIGHT_RATIO),
            ):
                keep = base & (d["klass"] == name)
                n, k5, _, _, _ = fraction_by_bin(
                    d["grvs"], d["e_worst"] < RECOVERED[0], EDGES, keep
                )
                _, k10, _, _, _ = fraction_by_bin(
                    d["grvs"], d["e_worst"] < RECOVERED[1], EDGES, keep
                )
                weight = [
                    float(np.mean(d["weight_cell"][keep & (d["grvs"] >= lo) & (d["grvs"] < hi)]))
                    if np.any(keep & (d["grvs"] >= lo) & (d["grvs"] < hi))
                    else 0.0
                    for lo, hi in itertools.pairwise(EDGES)
                ]
                for b in range(EDGES.size - 1):
                    rows.append(
                        [
                            declaration,
                            name,
                            subset,
                            EDGES[b],
                            int(n[b]),
                            int(k5[b]),
                            int(k10[b]),
                            weight[b],
                        ]
                    )
    write_csv(
        "fig03-recovery",
        [
            "declaration",
            "class",
            "systems",
            "grvs_low",
            "n",
            "both_within_5pct",
            "both_within_10pct",
            "catalogue_weight_per_system",
        ],
        rows,
    )
    ranges = ((6.0, 9.0), (9.0, 10.0), (10.0, 11.0), (11.0, 12.0), (12.0, 13.0), (13.0, 13.5))
    for declaration in design.declarations:
        d = design.rows(declaration)
        wd = d["weight_cell"]
        bright = d["light_ratio"] >= BRIGHT_RATIO
        for lo, hi in ranges:
            m = (d["grvs"] >= lo) & (d["grvs"] < hi)
            tag = f"{declaration}_{lo:g}_{hi:g}".replace(".", "p")
            numbers[f"rec10_{tag}"] = pct(wshare(d["e_worst"] < RECOVERED[1], m, wd))
            numbers[f"rec5_{tag}"] = pct(wshare(d["e_worst"] < RECOVERED[0], m, wd))
            numbers[f"rec10b_{tag}"] = pct(wshare(d["e_worst"] < RECOVERED[1], m & bright, wd))
            with np.errstate(invalid="ignore"):
                single = np.abs(d["s_r1"]) < RECOVERED[1]
                numbers[f"sb1_{tag}"] = pct(wshare(single, m, wd))
                numbers[f"sb1f_{tag}"] = pct(wshare(single, m & (d["light_ratio"] < 0.1), wd))
        # The magnitude beyond which fewer than half of the systems with a bright companion
        # are recovered: between the last bin at or above one half and the next.
        _, _, p, _, _ = fraction_by_bin(d["grvs"], d["e_worst"] < RECOVERED[1], EDGES, bright, wd)
        above = np.flatnonzero(np.nan_to_num(p, nan=0.0) >= 0.5)
        half = np.nan
        if above.size and above[-1] + 1 < CENTRES.size:
            i = above[-1]
            half = CENTRES[i] + (p[i] - 0.5) / (p[i] - p[i + 1]) * (CENTRES[i + 1] - CENTRES[i])
        elif above.size:
            half = EDGES[-1]
        numbers[f"half_{declaration}"] = f"{half:.1f}"

    # --- epoch errors ------------------------------------------------------------------
    inj = design.epoch_rows("injected")
    table, rows = {}, []
    for name in CLASSES:
        for k in (1, 2):
            base = inj["usable"] & (inj["klass"] == name)
            if k == 2:
                base &= inj["light_ratio"] >= BRIGHT_RATIO
            out = {key: np.full(CENTRES.size, np.nan) for key in ("sigma", "lo", "hi", "pred")}
            for b in range(CENTRES.size):
                m = base & (inj["grvs"] >= EDGES[b]) & (inj["grvs"] < EDGES[b + 1])
                if m.sum() < 30:
                    continue
                err = inj[f"err{k}"][m]
                out["sigma"][b] = robust_sigma(err)
                out["lo"][b], out["hi"][b] = np.quantile(np.abs(err), [0.16, 0.84])
                out["pred"][b] = np.median(inj[f"pred{k}"][m])
                rows.append(
                    [
                        name,
                        k,
                        EDGES[b],
                        int(m.sum()),
                        out["sigma"][b],
                        out["lo"][b],
                        out["hi"][b],
                        out["pred"][b],
                    ]
                )
            table[(name, k)] = out
    data["epoch_error"] = table
    write_csv(
        "fig04-epoch-error",
        [
            "class",
            "component",
            "grvs_low",
            "n_epochs",
            "robust_sigma_kms",
            "abs_error_p16",
            "abs_error_p84",
            "predicted_kms",
        ],
        rows,
    )
    for grvs in (8, 10, 12):
        b = int(np.searchsorted(EDGES, grvs + 1e-9) - 1)
        for name in CLASSES:
            numbers[f"sigma1_{name}_{grvs}"] = f"{table[(name, 1)]['sigma'][b]:.2f}"
            numbers[f"sigma2_{name}_{grvs}"] = f"{table[(name, 2)]['sigma'][b]:.1f}"
    clean = inj["usable"] & (inj["snr"] > 10)
    high = inj["usable"] & (inj["snr"] > 100)
    numbers["scatter_over_predicted"] = (
        f"{robust_sigma(inj['dev1'][clean] / inj['pred1'][clean]):.2f}"
    )
    numbers["scatter_over_predicted_orbit"] = (
        f"{robust_sigma(inj['err1'][clean] / inj['pred1'][clean]):.2f}"
    )
    numbers["scatter_high"] = f"{robust_sigma(inj['dev1'][high] / inj['pred1'][high]):.2f}"
    numbers["scatter_high_orbit"] = f"{robust_sigma(inj['err1'][high] / inj['pred1'][high]):.2f}"

    # --- pulls -------------------------------------------------------------------------
    # Against the velocity put into the spectrum. The secondary is that of the systems
    # with a flux ratio above BRIGHT_RATIO.
    grid = np.linspace(-6, 6, 61)
    hist = []
    for k in (1, 2):
        base = inj["usable"] & ((inj["light_ratio"] >= BRIGHT_RATIO) if k == 2 else True)
        values = inj[f"pulls{k}"][base]
        density, _ = np.histogram(values, grid, density=False)
        hist.append(density / (values.size * (grid[1] - grid[0])))
        numbers[f"pull_width_{k}"] = f"{robust_sigma(values):.2f}"
        numbers[f"pull_beyond3_{k}"] = pct(np.mean(np.abs(values) > 3), 1)
        numbers[f"pull_beyond5_{k}"] = pct(np.mean(np.abs(values) > 5), 1)
    numbers["pull_high"] = f"{robust_sigma(inj['pulls1'][high]):.2f}"
    numbers["pull_high_orbit"] = f"{robust_sigma(inj['pull1'][high]):.2f}"
    data["pull_hist"] = hist
    snr_edges = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 150.0])
    widths, rows = {}, []
    for declaration in design.declarations:
        e = design.epoch_rows(declaration)
        width = np.full(snr_edges.size - 1, np.nan)
        for b in range(snr_edges.size - 1):
            m = e["usable"] & (e["snr"] >= snr_edges[b]) & (e["snr"] < snr_edges[b + 1])
            if m.sum() >= 50:
                width[b] = robust_sigma(e["pulls1"][m])
            rows.append([declaration, snr_edges[b], int(m.sum()), width[b]])
        widths[declaration] = {"snr": np.sqrt(snr_edges[:-1] * snr_edges[1:]), "width": width}
        numbers[f"pull1_{declaration}"] = f"{robust_sigma(e['pulls1'][e['usable']]):.2f}"
        sel2 = e["usable"] & (e["light_ratio"] >= BRIGHT_RATIO)
        numbers[f"pull2_{declaration}"] = f"{robust_sigma(e['pulls2'][sel2]):.2f}"
        numbers[f"wrong_{declaration}"] = pct(np.mean(e["wrong"][sel2]), 1)
    data["pull_width"] = widths
    write_csv("fig05-pulls", ["declaration", "snr_low", "n_epochs", "pull_width_primary"], rows)

    # --- fate --------------------------------------------------------------------------
    fate = np.zeros((CENTRES.size, 5))
    rows = []
    we = head_e["weight_cell"]
    for b in range(CENTRES.size):
        m = (head_e["grvs"] >= EDGES[b]) & (head_e["grvs"] < EDGES[b + 1]) & (head_e["fate"] < 5)
        for k in range(5):
            fate[b, k] = wshare(head_e["fate"] == k, m, we)
        rows.append([EDGES[b], int(m.sum()), *fate[b]])
    data["fate"] = fate
    write_csv(
        "fig06-fate",
        [
            "grvs_low",
            "n_transits",
            "both_right",
            "primary_right_only",
            "primary_wrong",
            "flagged",
            "not_measured",
        ],
        rows,
    )
    numbers["in_eclipse"] = pct(wshare(head_e["fate"] == 5, np.ones(we.shape, bool), we))
    for grvs in (8, 10, 12, 13):
        b = int(np.searchsorted(EDGES, grvs + 1e-9) - 1)
        numbers[f"both_right_{grvs}"] = pct(fate[b, 0])
        numbers[f"primary_right_{grvs}"] = pct(fate[b, 0] + fate[b, 1])

    # --- separation --------------------------------------------------------------------
    out_e = (head_e["fate"] < 5) & (head_e["light_ratio"] >= BRIGHT_RATIO)
    width_kms = np.hypot(FWHM_KMS, 1.6 * head_e["vsini_max"])
    x = head_e["separation_kms"] / width_kms
    sep_edges = np.array([0.1, 0.2, 0.35, 0.6, 1.0, 1.5, 2.5, 4.0, 8.0])
    separation, rows = {}, []
    for label, lo, hi in (("below 10", 0, 10), ("10 to 30", 10, 30), ("above 30", 30, 1e9)):
        m = out_e & (head_e["snr"] >= lo) & (head_e["snr"] < hi)
        n, _, usable, _, _ = fraction_by_bin(x, head_e["usable"], sep_edges, m)
        _, _, wrong, _, _ = fraction_by_bin(x, head_e["wrong"], sep_edges, m & head_e["usable"])
        separation[label] = {
            "x": np.sqrt(sep_edges[:-1] * sep_edges[1:]),
            "usable": usable,
            "wrong": wrong,
        }
        for b in range(sep_edges.size - 1):
            rows.append([label, sep_edges[b], int(n[b]), usable[b], wrong[b]])
    data["separation"] = separation
    write_csv(
        "fig07-separation",
        ["snr_per_pixel", "separation_low", "n_transits", "usable_share", "wrong_share_of_usable"],
        rows,
    )
    wide = out_e & (x >= 1.5) & (head_e["snr"] >= 30)
    close = out_e & (x < 0.6) & (head_e["snr"] >= 30)
    numbers["usable_wide"] = pct(np.mean(head_e["usable"][wide]))
    numbers["usable_close"] = pct(np.mean(head_e["usable"][close]))
    numbers["wrong_wide"] = pct(np.mean(head_e["wrong"][wide & head_e["usable"]]), 1)
    numbers["wrong_close"] = pct(np.mean(head_e["wrong"][close & head_e["usable"]]), 1)

    # --- detection of the secondary -----------------------------------------------------
    ratio_edges = np.array([0.001, 0.03, 0.1, 0.2, 0.4, 0.7, 1.6])
    detection, rows = {}, []
    detected = np.nan_to_num(head["z2"], nan=0.0) > thresholds[HEADLINE]
    head["detected"] = detected
    for name in CLASSES:
        grid = np.full((ratio_edges.size - 1, WIDE_EDGES.size - 1), np.nan)
        for r in range(ratio_edges.size - 1):
            for c in range(WIDE_EDGES.size - 1):
                m = (
                    (head["klass"] == name)
                    & (head["light_ratio"] >= ratio_edges[r])
                    & (head["light_ratio"] < ratio_edges[r + 1])
                    & (head["grvs"] >= WIDE_EDGES[c])
                    & (head["grvs"] < WIDE_EDGES[c + 1])
                )
                if m.sum() >= 5:
                    grid[r, c] = np.mean(detected[m])
                rows.append([name, ratio_edges[r], WIDE_EDGES[c], int(m.sum()), grid[r, c]])
        detection[name] = grid
    data["detection"], data["ratio_edges"] = detection, ratio_edges
    write_csv(
        "fig08-secondary", ["class", "flux_ratio_low", "grvs_low", "n", "detected_share"], rows
    )
    for declaration in design.declarations:
        d = design.rows(declaration)
        wd = d["weight_cell"]
        found = np.nan_to_num(d["z2"], nan=0.0) > thresholds[declaration]
        recovered = d["e_worst"] < RECOVERED[1]
        numbers[f"threshold_{declaration}"] = f"{thresholds[declaration]:.1f}"
        numbers[f"detected_{declaration}"] = pct(wshare(found, np.ones(found.shape, bool), wd))
        numbers[f"purity_{declaration}"] = pct(wshare(recovered, found, wd))
        numbers[f"complete_{declaration}"] = pct(wshare(found, recovered, wd))
        if null is not None and declaration in null.declarations:
            z = np.nan_to_num(null.rows(declaration)["z2"], nan=0.0)
            numbers[f"null_above5_{declaration}"] = pct(np.mean(z > 5.0), 1)
    numbers["n_null"] = "0" if null is None else f"{len(null.population):,}"
    if null is not None:
        # The primary alone, measured with its own template: a single star. The errors
        # are against the velocity put into the spectrum for the calibration, and against
        # the orbit for the comparison with the scatter Gaia publishes.
        single = null.epoch_rows("injected")
        with np.errstate(invalid="ignore", divide="ignore"):
            error = single["vs"] - single["v1_true"]
            deviation = error - single["zero_point"]
            pull = deviation / single["ss"]
            ratio = deviation / single["pred1"]
        fine = (single["gs"] == 1) & np.isfinite(error)
        bright = fine & (single["snr"] > 10)
        numbers["single_scatter"] = f"{robust_sigma(ratio[bright]):.2f}"
        numbers["single_pull"] = f"{robust_sigma(pull[bright]):.2f}"
        teff = null.cols["teff1"][single["system"]]
        slow = fine & (teff >= 5500) & (teff < 6500) & (single["vsini1"] < 20)
        import dataclasses

        from albireo.gaia import RVS_CONSTANTS, rvs_snr_per_pixel
        from albireo.survey import SurveyConfig

        # What a background that varies between transits, as in the instrument run, adds
        # to the standard deviation of the noise of a transit in the S/N model.
        limit = math.log(SurveyConfig().background_factor)
        factors = np.exp(np.random.default_rng(2).uniform(-limit, limit, size=2000))
        for grvs in (10, 12):
            varied = np.array(
                [
                    float(
                        rvs_snr_per_pixel(
                            float(grvs),
                            constants=dataclasses.replace(
                                RVS_CONSTANTS, background_e=RVS_CONSTANTS.background_e * factor
                            ),
                        )
                    )
                    for factor in factors
                ]
            )
            excess = math.sqrt(
                float(np.mean((float(rvs_snr_per_pixel(float(grvs))) / varied) ** 2))
            )
            numbers[f"noise_rms_{grvs}"] = f"{100 * (excess - 1):.0f}"

        rows = []
        steps = np.linspace(0.0, 3.0, 601)
        for grvs, lo, hi, published in GAIA_TRANSITS:
            m = slow & (single["grvs"] >= lo) & (single["grvs"] < hi)
            value = robust_sigma(error[m]) if m.sum() >= 30 else float("nan")
            # The magnitudes by which a simulated transit must be fainter to have the
            # published scatter, with the zero point taken from both.
            shift = float("nan")
            if min(value, published) > 1.2 * ZERO_POINT_KMS:
                factor = math.sqrt(
                    (published**2 - ZERO_POINT_KMS**2) / (value**2 - ZERO_POINT_KMS**2)
                )
                centre = 0.5 * (lo + hi)
                gain = np.asarray(rvs_snr_per_pixel(centre)) / np.asarray(
                    rvs_snr_per_pixel(centre + steps)
                )
                shift = float(np.interp(factor, gain, steps))
            numbers[f"single_sigma_{grvs}"] = f"{value:.2f}"
            numbers[f"single_gaia_{grvs}"] = f"{published:.2f}"
            numbers[f"single_ratio_{grvs}"] = f"{published / value:.1f}"
            if math.isfinite(shift):
                numbers[f"single_shift_{grvs}"] = f"{shift:.1f}"
            rows.append([lo, hi, int(m.sum()), value, published, published / value, shift])
        write_csv(
            "check-single-stars",
            [
                "grvs_low",
                "grvs_high",
                "n_transits",
                "robust_sigma_kms",
                "gaia_dr3_kms",
                "gaia_over_simulated",
                "equivalent_magnitudes",
            ],
            rows,
        )

    # --- rotation ----------------------------------------------------------------------
    v_edges = np.array([3.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0])
    rotation, rows = {}, []
    for declaration in ("injected", "classified"):
        e = design.epoch_rows(declaration)
        for name in CLASSES:
            ratio = np.full(v_edges.size - 1, np.nan)
            for b in range(v_edges.size - 1):
                m = (
                    e["usable"]
                    & (e["klass"] == name)
                    & (e["snr"] > 10)
                    & (e["vsini1"] >= v_edges[b])
                    & (e["vsini1"] < v_edges[b + 1])
                )
                if m.sum() >= 50:
                    ratio[b] = robust_sigma(e["dev1"][m] / e["pred1"][m])
                rows.append([declaration, name, v_edges[b], int(m.sum()), ratio[b]])
            rotation[(name, declaration)] = {
                "x": np.sqrt(v_edges[:-1] * v_edges[1:]),
                "ratio": ratio,
            }
    data["rotation"] = rotation
    write_csv(
        "fig09-rotation",
        ["declaration", "class", "vsini_low_kms", "n_epochs", "scatter_over_predicted"],
        rows,
    )

    # --- orbits and masses ---------------------------------------------------------------
    grid_x = np.logspace(-3, 0, 121)
    groups = (("below 9", 6.0, 9.0), ("9 to 11", 9.0, 11.0), ("11 to 12.5", 11.0, 12.5))
    wh = head["weight_cell"]
    rows = []
    for key, column in (("k_cdf", "e_worst"), ("m_cdf", "m_worst")):
        curves = {}
        for label, lo, hi in groups:
            m = (head["grvs"] >= lo) & (head["grvs"] < hi)
            y = np.array([wshare(head[column] < value, m, wh) for value in grid_x])
            curves[label] = (grid_x, y)
            for value in (0.0045, 0.013, 0.045, 0.01, 0.03, 0.05, 0.10):
                rows.append(
                    [column, label, value, wshare(head[column] < value, m, wh), int(m.sum())]
                )
        data[key] = curves
    write_csv("fig10-orbits", ["quantity", "grvs", "threshold", "weighted_share_below", "n"], rows)
    for label, lo, hi in groups:
        m = (head["grvs"] >= lo) & (head["grvs"] < hi)
        tag = label.replace(" ", "_").replace(".", "p")
        numbers[f"mass3_{tag}"] = pct(wshare(head["m_worst"] < 0.03, m, wh))
        numbers[f"mass10_{tag}"] = pct(wshare(head["m_worst"] < 0.10, m, wh))
        numbers[f"mass1_{tag}"] = pct(wshare(head["m_worst"] < 0.01, m, wh))
    for declaration in design.declarations:
        d = design.rows(declaration)
        with np.errstate(invalid="ignore"):
            ok = (d["e_ok"] == 1) & (d["e_worst"] < RECOVERED[1])
            for k in (1, 2):
                pull = (d[f"e_k{k}"] - d[f"k{k}_true"])[ok] / d[f"e_k{k}_err"][ok]
                numbers[f"k{k}_pull_{declaration}"] = f"{robust_sigma(pull):.2f}"
            offset = (d["e_gamma"] - design.cols["gamma"][d["system"]])[ok]
            numbers[f"gamma_sigma_{declaration}"] = f"{robust_sigma(offset):.2f}"

    # --- the ephemeris -----------------------------------------------------------------
    ephemeris, rows = {}, []
    for label, column in (
        ("period and conjunction held", "e_worst"),
        ("period known, conjunction fitted", "p_worst"),
    ):
        n, k, p, lo, hi = fraction_by_bin(
            head["grvs"], head[column] < RECOVERED[1], EDGES, None, wh
        )
        ephemeris[label] = {"p": p, "lo": lo, "hi": hi}
        for b in range(CENTRES.size):
            rows.append([label, EDGES[b], int(n[b]), int(k[b]), p[b]])
    data["ephemeris"] = ephemeris
    write_csv(
        "fig11-ephemeris", ["fit", "grvs_low", "n", "both_within_10pct", "weighted_share"], rows
    )
    for lo, hi in ((6.0, 11.0), (11.0, 12.5), (12.5, 13.5)):
        m = (head["grvs"] >= lo) & (head["grvs"] < hi)
        tag = f"{lo:g}_{hi:g}".replace(".", "p")
        numbers[f"held_{tag}"] = pct(wshare(head["e_worst"] < RECOVERED[1], m, wh))
        numbers[f"free_{tag}"] = pct(wshare(head["p_worst"] < RECOVERED[1], m, wh))
    fitted = head["p_ok"] == 1
    numbers["free_exchanged"] = pct(np.mean(head["p_swapped"][fitted] == 1))
    held, free = (ephemeris[key]["p"] for key in ephemeris)
    numbers["held_minus_free"] = f"{100 * np.nanmax(np.abs(held - free)):.0f}"

    # --- the templates -----------------------------------------------------------------
    rows, templates = [], {}
    for declaration in design.declarations:
        d = design.rows(declaration)
        for name in CLASSES:
            keep = (d["klass"] == name) & (d["light_ratio"] >= BRIGHT_RATIO)
            n, k, p, lo, hi = fraction_by_bin(
                d["grvs"], d["e_worst"] < RECOVERED[1], WIDE_EDGES, keep
            )
            templates[(name, declaration)] = {"p": p, "lo": lo, "hi": hi}
            for b in range(WIDE_EDGES.size - 1):
                rows.append([declaration, name, WIDE_EDGES[b], int(n[b]), int(k[b])])
            for label, lo_mag, hi_mag in (("bright", 6.0, 10.0), ("faint", 10.0, 12.0)):
                m = keep & (d["grvs"] >= lo_mag) & (d["grvs"] < hi_mag)
                numbers[f"templates_{label}_{declaration}_{name}"] = pct(
                    wshare(d["e_worst"] < RECOVERED[1], m, d["weight_cell"])
                )
    data["templates"] = templates
    write_csv(
        "fig12-templates", ["declaration", "class", "grvs_low", "n", "both_within_10pct"], rows
    )

    # --- perturbations -----------------------------------------------------------------
    # Paired with the design run, which has the same systems, transit times and seeds.
    # The shares are those of the design sample, which has equal numbers at every
    # magnitude and class, and are not weighted.
    index_of = {name: i for i, name in enumerate(design.names)}
    base_rows = {int(s): r for r, s in enumerate(head["system"])}
    perturbations, rows = [], []

    def epochs_by_system(run_epochs) -> dict[int, np.ndarray]:
        order = np.argsort(run_epochs["system"], kind="stable")
        bounds = np.flatnonzero(np.diff(run_epochs["system"][order])) + 1
        return {int(run_epochs["system"][chunk[0]]): chunk for chunk in np.split(order, bounds)}

    design_epochs = epochs_by_system(head_e)

    def paired(variant: Run, keep_names, label):
        v = variant.rows(HEADLINE)
        ve = variant.epoch_rows(HEADLINE)
        names = variant.names[v["system"]]
        keep = np.array([n in keep_names for n in names])
        a = np.array(
            [head["e_worst"][base_rows[index_of[n]]] < RECOVERED[1] for n in names[keep]],
            dtype=float,
        )
        b = (v["e_worst"][keep] < RECOVERED[1]).astype(float)
        # Median error of each semi-amplitude where the pair is double-lined and bright.
        clear = keep & (v["light_ratio"] >= BRIGHT_RATIO) & (v["grvs"] < 11.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            bias = [
                100 * float(np.nanmedian(v[f"e_k{k}"][clear] / v[f"k{k}_true"][clear] - 1.0))
                if clear.any()
                else float("nan")
                for k in (1, 2)
            ]
        # Scatter of the primary's velocity in km/s over the transits usable in both runs.
        ratios = []
        for i, chunk in epochs_by_system(ve).items():
            name = variant.names[i]
            if name not in keep_names or index_of[name] not in design_epochs:
                continue
            other = design_epochs[index_of[name]]
            if other.size != chunk.size:
                continue
            both = ve["usable"][chunk] & head_e["usable"][other]
            if both.sum() < 8:
                continue
            reference = robust_sigma(head_e["err1"][other][both])
            if reference > 0.0:
                ratios.append(robust_sigma(ve["err1"][chunk][both]) / reference)
        ratios = np.asarray(ratios)
        draws_d, draws_s = [], []
        for _ in range(400):
            pick = rng.integers(0, a.size, a.size)
            draws_d.append(100 * (b[pick].mean() - a[pick].mean()))
            draws_s.append(100 * (np.median(ratios[rng.integers(0, ratios.size, ratios.size)]) - 1))
        delta = (100 * (b.mean() - a.mean()), *np.quantile(draws_d, [0.16, 0.84]))
        scatter = (100 * (np.median(ratios) - 1), *np.quantile(draws_s, [0.16, 0.84]))
        perturbations.append({"label": label, "delta": delta, "scatter": scatter})
        rows.append(
            [
                label,
                int(keep.sum()),
                100 * a.mean(),
                100 * b.mean(),
                *delta[1:],
                *scatter,
                ratios.size,
            ]
        )
        return int(keep.sum()), delta, scatter, 100 * a.mean(), 100 * b.mean(), bias

    for key, tag, label, chooser in (
        ("third-light", "third", "third light", lambda s: s.meta.get("tertiary") is not None),
        (
            "third-light",
            "third10",
            "third light above 10 percent",
            lambda s: (s.meta.get("tertiary") or {}).get("light", 0) > 0.1,
        ),
        (
            "activity",
            "ca",
            "Ca II emission",
            lambda s: max(s.meta.get("activity_ew", [0, 0])) > 0,
        ),
        ("instrument", "background", "background and normalisation", lambda s: True),
        (
            "instrument",
            "background11",
            "the same, fainter than 11",
            lambda s: s.grvs >= 11.0,
        ),
    ):
        if key not in variants:
            continue
        chosen = {s.name for s in variants[key].population if chooser(s)}
        n, delta, scatter, before, after, bias = paired(variants[key], chosen, label)
        for k, value in enumerate(bias, start=1):
            if math.isfinite(value):
                numbers[f"pert_k{k}_{tag}"] = f"{value:+.1f}"
        numbers[f"pert_n_{tag}"] = str(n)
        numbers[f"pert_share_{tag}"] = pct(n / len(variants[key].population))
        numbers[f"pert_delta_{tag}"] = f"{delta[0]:+.1f}"
        numbers[f"pert_scatter_{tag}"] = f"{scatter[0]:+.0f}"
        numbers[f"pert_before_{tag}"] = f"{before:.0f}"
        numbers[f"pert_after_{tag}"] = f"{after:.0f}"
    data["perturbations"] = perturbations
    write_csv(
        "fig13-perturbations",
        [
            "effect",
            "n_systems",
            "recovered_baseline_pct",
            "recovered_variant_pct",
            "delta_p16",
            "delta_p84",
            "scatter_change_pct",
            "scatter_p16",
            "scatter_p84",
            "n_systems_in_scatter",
        ],
        rows,
    )
    # Transits in eclipse, as measured, against those out of eclipse.
    with np.errstate(invalid="ignore", divide="ignore"):
        raw = (inj["v1"] - inj["v1_true"] - inj["zero_point"]) / inj["pred1"]
    inside, outside = inj["in_eclipse"] == 1, inj["in_eclipse"] == 0
    numbers["eclipse_within3"] = pct(np.mean(np.abs(raw[inside]) < 3))
    numbers["outside_within3"] = pct(np.mean(np.abs(raw[outside]) < 3))

    # --- the recovery function -----------------------------------------------------------
    s_edges = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0, 512.0, 1024.0, 3000.0])
    function, rows = {}, []
    for declaration in design.declarations:
        d = design.rows(declaration)
        flag = d["e_worst"] < RECOVERED[1]
        n, k, p, lo, hi = fraction_by_bin(d["snr_k2"], flag, s_edges)
        top, mid, width = fit_recovery(d["snr_k2"], flag)
        grid_s = np.logspace(0, np.log10(3000), 200)
        function[declaration] = {
            "x": np.sqrt(s_edges[:-1] * s_edges[1:]),
            "p": p,
            "lo": lo,
            "hi": hi,
            "grid": grid_s,
            "fit": _logistic(grid_s, top, mid, width),
        }
        for b in range(s_edges.size - 1):
            rows.append([declaration, s_edges[b], int(n[b]), int(k[b])])
        numbers[f"fn_top_{declaration}"] = f"{top:.2f}"
        numbers[f"fn_mid_{declaration}"] = f"{10**mid:.0f}"
        numbers[f"fn_width_{declaration}"] = f"{width:.2f}"
    data["function"] = function
    write_csv(
        "fig14-function", ["declaration", "predicted_snr_low", "n", "both_within_10pct"], rows
    )

    # --- the catalogue -----------------------------------------------------------------
    flags = {
        "orbit": detected & (head["e_worst"] < RECOVERED[1]),
        "mass10": detected & (head["m_worst"] < 0.10),
        "mass3": detected & (head["m_worst"] < 0.03),
    }

    def cumulative(values, grvs):
        return np.array([np.sum(values[grvs < hi]) for hi in EDGES[1:]])

    result = {"candidates": cumulative(wh, head["grvs"])}
    # The design fixes the number of systems in a cell, so the bootstrap resamples the
    # systems within each cell.
    cells: dict[tuple[str, int], list[int]] = {}
    for r, (name, grvs) in enumerate(zip(head["klass"], head["grvs"], strict=True)):
        cells.setdefault((name, int(np.digitize(grvs, EDGES) - 1)), []).append(r)
    members = [np.asarray(v) for v in cells.values()]
    boot = {key: [] for key in flags}
    for _ in range(300):
        pick = np.concatenate([m[rng.integers(0, m.size, m.size)] for m in members])
        for key, flag in flags.items():
            boot[key].append(cumulative((wh * flag)[pick], head["grvs"][pick]))
    rows = []
    for key, flag in flags.items():
        result[key] = cumulative(wh * flag, head["grvs"])
        result[key + "_lo"], result[key + "_hi"] = np.quantile(
            np.array(boot[key]), [0.16, 0.84], axis=0
        )
    for b, hi in enumerate(EDGES[1:]):
        rows.append(
            [
                hi,
                result["candidates"][b],
                *(
                    result[k][b]
                    for k in (
                        "orbit",
                        "orbit_lo",
                        "orbit_hi",
                        "mass10",
                        "mass10_lo",
                        "mass10_hi",
                        "mass3",
                        "mass3_lo",
                        "mass3_hi",
                    )
                ),
            ]
        )
    data["yield"] = result
    write_csv(
        "fig15-yield",
        [
            "grvs_limit",
            "candidates",
            "orbit",
            "orbit_p16",
            "orbit_p84",
            "mass10",
            "mass10_p16",
            "mass10_p84",
            "mass3",
            "mass3_p16",
            "mass3_p84",
        ],
        rows,
    )
    for limit in (10.0, 11.0, 12.0, 13.0, 13.5):
        b = int(np.argmin(np.abs(EDGES[1:] - limit)))
        tag = f"{limit:g}".replace(".", "p")
        numbers[f"cand_{tag}"] = f"{result['candidates'][b]:,.0f}"
        below = head["grvs"] < EDGES[1:][b]
        for key, flag in flags.items():
            numbers[f"{key}_{tag}"] = f"{result[key][b]:,.0f}"
            numbers[f"{key}_{tag}_lo"] = f"{result[key + '_lo'][b]:,.0f}"
            numbers[f"{key}_{tag}_hi"] = f"{result[key + '_hi'][b]:,.0f}"
            numbers[f"{key}_{tag}_pct"] = pct(wshare(flag, below, wh), 1)
            # The same share with the weights raked to the catalogue's periods, which
            # cover a part of the candidates only.
            numbers[f"{key}_{tag}_raked_pct"] = pct(wshare(flag, below, head["weight"]), 1)
    numbers["raked_coverage"] = pct(head["weight"].sum() / wh.sum())

    # --- cases ---------------------------------------------------------------------------
    data["cases"] = cases(design, head, head_e)

    # --- the recorded systems and the cost ------------------------------------------------
    if recorded is not None:
        r = recorded.rows("injected")
        sep = (r["k1_true"] + r["k2_true"]) * (1.0 + recorded.cols["ecc"][r["system"]])
        ok5 = r["p_worst"] < RECOVERED[0]
        numbers.update(
            rec_n=str(len(recorded.population)),
            rec_usable=pct(np.sum(r["n_good"]) / np.sum(r["n_out"]), 1),
            rec_wide=f"{int(np.sum(ok5 & (sep >= 100)))} of {int(np.sum(sep >= 100))}",
            rec_close=f"{int(np.sum(ok5 & (sep < 100)))} of {int(np.sum(sep < 100))}",
        )
    for run in (design, *variants.values(), *((null,) if null is not None else ())):
        tag = run.name.replace("-", "_")
        numbers[f"wall_{tag}"] = f"{run.manifest.get('wall_clock_seconds', float('nan')) / 60:.0f}"
        numbers[f"ran_{tag}"] = f"{run.manifest['n_systems']:,}"
        numbers[f"failures_{tag}"] = str(len(run.manifest["failures"]))
    first = design.systems["decl"] == 0
    stages = {"t_sim": float(design.systems["t_sim"][first].mean())}
    for column in ("t_todcor", "t_orbit"):
        stages[column] = float(
            sum(
                design.systems[column][design.systems["decl"] == k].mean()
                for k in range(len(design.declarations))
            )
        )
    for column, value in stages.items():
        numbers[column] = f"{value:.1f}"
    numbers["seconds_per_system"] = f"{sum(stages.values()):.1f}"
    numbers["workers"] = str(design.manifest.get("workers", "n/a"))
    numbers["epochs_design"] = f"{int(design.systems['n_epochs'][first].sum()):,}"
    commit = str(design.manifest.get("commit", "unknown"))[:7]
    if design.manifest.get("uncommitted_changes"):
        commit += " with uncommitted changes"
    numbers["commit"] = commit
    created = str(design.manifest.get("created", ""))
    numbers["created"] = created[:10] + (" UTC" if created.endswith("+00:00") else "")
    return data, numbers


def _percent(value) -> str:
    return "not fitted" if not np.isfinite(value) else f"{100 * value:+.0f} percent"


def cases(design: Run, head, head_e) -> list[dict]:
    """Five systems that were not recovered, each for a different reason, and one that was."""
    from albireo.kepler import _t_peri_from_t_conj_numpy, true_anomaly_numpy

    population = design.population
    failed = head["e_worst"] >= RECOVERED[1]
    vmax = np.maximum(head["vsini1"], head["vsini2"])
    slow = vmax < 60
    rules = (
        ("a faint companion", failed & (head["light_ratio"] < 0.05) & (head["grvs"] < 10)),
        (
            "fast rotation",
            failed & (head["light_ratio"] >= 0.3) & (vmax > 100) & (head["grvs"] < 10.5),
        ),
        (
            "an eccentric orbit",
            failed
            & (head["light_ratio"] >= 0.3)
            & (head["ecc"] > 0.3)
            & slow
            & (head["grvs"] < 11),
        ),
        (
            "too few photons",
            failed
            & (head["light_ratio"] >= 0.5)
            & slow
            & (head["ecc"] == 0)
            & (head["grvs"] > 12.2)
            & (head["grvs"] < 12.8),
        ),
        (
            "a pair of alike stars",
            failed
            & (head["light_ratio"] >= 0.5)
            & slow
            & (head["ecc"] == 0)
            & (head["grvs"] < 10.5),
        ),
        (
            "recovered",
            (head["e_worst"] < 0.02)
            & (head["light_ratio"] >= 0.5)
            & (head["grvs"] > 9.5)
            & (head["grvs"] < 10.5),
        ),
    )
    out, rows = [], []
    for reason, rule in rules:
        index = np.flatnonzero(rule)
        if index.size == 0:
            continue
        row = int(index[np.argmin(head["grvs"][index])]) if reason != "recovered" else int(index[0])
        i = int(head["system"][row])
        system = population[i]
        m = head_e["system"] == i
        phase = np.linspace(0.0, 1.0, 400)
        t_conj = system.t_conj
        t_peri = _t_peri_from_t_conj_numpy(
            t_conj, period=system.period, ecc=system.ecc, omega=system.omega
        )
        nu = true_anomaly_numpy(
            t_conj + phase * system.period, period=system.period, t_peri=t_peri, ecc=system.ecc
        )
        curve = np.cos(nu + system.omega) + system.ecc * math.cos(system.omega)
        usable = head_e["usable"][m]
        out.append(
            {
                "phase": head_e["phase"][m],
                "v1": np.where(usable, head_e["ve1"][m], head_e["v1"][m]),
                "v2": np.where(usable, head_e["ve2"][m], head_e["v2"][m]),
                "s1": np.where(usable, head_e["se1"][m], 0.0),
                "s2": np.where(usable, head_e["se2"][m], 0.0),
                "usable": usable,
                "curve1": system.gamma + system.k1 * curve,
                "curve2": system.gamma - system.k2 * curve,
                "gamma": system.gamma,
                "title": reason[0].upper() + reason[1:],
                "note": (
                    f"$G_{{\\rm RVS}}$ {system.grvs:.1f}, {system.teff1:.0f} + {system.teff2:.0f} K, "
                    f"flux ratio {system.light_ratio:.2f}\n"
                    f"P {system.period:.2f} d, e {system.ecc:.2f}, "
                    f"v sin i {system.vsini1:.0f} and {system.vsini2:.0f} km/s\n"
                    f"K errors {_percent(head['e_r1'][row])} and {_percent(head['e_r2'][row])}"
                ),
            }
        )
        rows.append(
            [
                reason,
                system.name,
                system.grvs,
                system.teff1,
                system.teff2,
                system.light_ratio,
                system.period,
                system.ecc,
                system.vsini1,
                system.vsini2,
                float(head["e_r1"][row]),
                float(head["e_r2"][row]),
                int(usable.sum()),
                int(usable.size),
            ]
        )
    write_csv(
        "fig16-cases",
        [
            "case",
            "system",
            "grvs",
            "teff1_K",
            "teff2_K",
            "flux_ratio",
            "period_d",
            "ecc",
            "vsini1_kms",
            "vsini2_kms",
            "k1_relative_error",
            "k2_relative_error",
            "usable_transits",
            "transits",
        ],
        rows,
    )
    return out


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------


def figure(number: int, name: str, caption: str) -> str:
    base = f"gaia-rvs-eclipsing-binaries/figures/{name}"
    # Each rendering links to its own file, which a browser shows at full size.
    return (
        f"[![Figure {number}]({base}-light.png#only-light)]({base}-light.png)\n"
        f"[![Figure {number}]({base}-dark.png#only-dark)]({base}-dark.png)\n\n"
        f"*Figure {number}.* {caption} Table: "
        f"[`{name}.csv`](gaia-rvs-eclipsing-binaries/tables/{name}.csv).\n"
    )


PAGE = """# Gaia DR4 eclipsing binaries in the RVS: epoch velocities by correlation

Gaia DR4 publishes one RVS spectrum per field-of-view transit. This page reports a
simulation of [[n_design]] detached eclipsing binaries as that product will show them, in
which both velocities are measured at every transit by correlation against library
templates ([TODCOR](../tutorials/todcor.md)) and an orbit is fitted at the photometric
ephemeris. Nothing is disentangled. The page, its figures and every number on it are
written by `scripts/rvs_eb_report.py` from the tables of the runs (albireo [[commit]],
[[created]]).

## Summary

Five findings on the extraction of radial velocities from RVS epoch spectra. Each is
developed in a section below.

1. **Precision of a single transit.** With the injected templates the scatter of a
   transit's velocity about the velocity in its spectrum is [[scatter_over_predicted]]
   times the photon-limited prediction, and the quoted errors are calibrated: the robust
   width of the pulls is [[pull_width_1]] for the primaries and [[pull_width_2]] for the
   secondaries. With the zero point of 0.17 km/s that every transit carries, the scatter of
   a G-type primary about its orbit is [[sigma1_G_8]], [[sigma1_G_10]] and [[sigma1_G_12]]
   km/s in the half magnitudes that begin at G_RVS = 8, 10 and 12.
2. **Cost of imperfect templates.** At the bright end the result is limited by the
   templates. Brighter than G_RVS = 9, both semi-amplitudes are within 10 percent of the
   injected ones for [[rec10_injected_6_9]] percent of the systems with the injected
   templates, [[rec10_classified_6_9]] percent with templates from a classification and
   [[rec10_catalogue_6_9]] percent with one template from the colour of the pair. A-type
   primaries lose the most: among the pairs with a flux ratio above 0.2 and G_RVS below 10
   the three declarations recover [[templates_bright_injected_A]],
   [[templates_bright_classified_A]] and [[templates_bright_catalogue_A]] percent of them.
   The quoted errors do not contain the mismatch. For the recovered systems the pulls of
   the two semi-amplitudes have robust widths of [[k1_pull_classified]] and
   [[k2_pull_classified]] with classified templates.
3. **Faint limit and the scale of the noise.** With classified templates, among the
   systems whose secondary has more than a fifth of the primary's flux, both
   semi-amplitudes are within 10 percent for [[rec10b_classified_6_9]] percent brighter
   than G_RVS = 9, [[rec10b_classified_10_11]] percent at 10 to 11,
   [[rec10b_classified_11_12]] percent at 11 to 12 and [[rec10b_classified_12_13]] percent
   at 12 to 13. Half of them are recovered at G_RVS = [[half_classified]]. Over all
   systems in the catalogue's
   mixture of classes, which include every mass ratio from 0.1, the shares are
   [[rec10_classified_6_9]], [[rec10_classified_10_11]], [[rec10_classified_11_12]] and
   [[rec10_classified_12_13]] percent. A simulated transit is at the photon limit of the
   S/N that DPAC expects for it. The scatter of one transit that the errors of Gaia DR3
   imply for single dwarfs is [[single_ratio_8]], [[single_ratio_10]] and
   [[single_ratio_12]] times that of the simulated single stars at G_RVS = 8, 10 and 12,
   the equivalent of [[single_shift_8]], [[single_shift_10]] and [[single_shift_12]] mag.
   If the transits of DR4 are as DR3 measured them, a recovery quoted here at a magnitude
   holds that much brighter ([Limits](#limits)).
4. **Blended lines and faint companions.** For the pairs with a flux ratio above 0.2, at
   S/N above 30 with classified templates, [[usable_wide]] percent of the transits whose
   lines are separated by more than 1.5 line widths are usable, and [[wrong_wide]] percent
   of those have a velocity more than five quoted errors and 3 km/s from the injected one.
   Below 0.6 widths the numbers are [[usable_close]] and [[wrong_close]] percent. Where the
   secondary has less than a tenth of the primary's flux, the primary's semi-amplitude
   from its template alone is within 10 percent for [[sb1f_classified_6_9]] percent of the
   systems brighter than G_RVS = 9, [[sb1f_classified_11_12]] percent at 11 to 12 and
   [[sb1f_classified_12_13]] percent at 12 to 13.
5. **Third light and chromospheric emission.** In the model of the population
   [[pert_share_third]] percent of the systems have an unresolved third star. Left out of
   the analysis, it lowers the share recovered among them from [[pert_before_third]] to
   [[pert_after_third]] percent, and to [[pert_after_third10]] percent where it gives more
   than a tenth of the light. Its lines stand at the systemic velocity and draw the
   semi-amplitudes low: in that case the median errors of the two are [[pert_k1_third10]]
   and [[pert_k2_third10]] percent among the pairs with a flux ratio above 0.2 and G_RVS
   below 11. Emission in the Ca II cores at the strength of saturated activity, in the
   [[pert_share_ca]] percent of systems with a star that qualifies, lowers the share from
   [[pert_before_ca]] to [[pert_after_ca]] percent, with median errors of [[pert_k1_ca]]
   and [[pert_k2_ca]] percent. A background that varies between transits changes the share
   by [[pert_delta_background]] points. These are shares of a sample with equal numbers at
   every magnitude and class.

Weighted to the [[catalogue_classes]] Gaia DR3 candidates of the detached light-curve
classes with G_RVS between 6 and 13.5, the baseline gives [[orbit_12]] double-lined orbits
brighter than G_RVS = 12 and [[orbit_13p5]] brighter than 13.5, with both masses to 10
percent for [[mass10_13p5]] systems and to 3 percent for [[mass3_13p5]]. The baseline has
no third light, no emission and photon-limited transits. One system takes
[[seconds_per_system]] s for three template declarations with [[workers]] processes
running: [[t_sim]] s for the simulation, [[t_todcor]] s for the correlations and
[[t_orbit]] s for the orbit fits. The [[n_design]] systems, [[epochs_design]] transits,
took [[wall_design]] minutes on a desktop.

## Question and scope

For the detached eclipsing binaries among the sources of the RVS, which per-transit
velocities of both components can be measured by correlation alone, with what errors, and
which orbits and masses follow when the eclipse ephemeris is taken from the photometry?
The answer is given against G_RVS, which sets the S/N of a transit, and against the
spectral types, the flux ratio, the rotation and the separation of the lines.

The simulation covers detached pairs of main-sequence and subgiant stars with primaries of
4000 to 10,000 K, G_RVS from 6.0 to 13.5, and the DR4 epoch product: 961 samples from 846
to 870 nm at a resolving power near 11,500. Contact and semi-detached systems, giants and
stars hotter than 10,000 K are outside it. So is the period search: an eclipsing binary
has a photometric period.

## The simulated sample

The sample is stratified. A sample that followed the catalogue's magnitudes would have
more than half of its systems in the faintest magnitude, where little is recovered, so
every cell of fifteen bins of G_RVS by four classes of the primary's temperature holds the
same number of systems, [[per_cell]], and the catalogue enters as weights. G_RVS is
assigned and not derived from a distance, since the S/N of a transit depends on nothing
else.

Within a class the systems follow a population model
([`albireo.eclipsing.draw_eclipsing_population`](../api/eclipsing.md)):

| ingredient | prescription |
|---|---|
| primary | initial mass function and a constant star-formation rate over 10 Gyr; kept while its MIST track is in the class, from the zero-age main sequence to log g = 3.5 |
| companion | period and mass ratio of Moe & Di Stefano (2017) from q = 0.1, with their excess of twins reduced to 0.3 of the published fraction and spread from q = 0.85 (see below); the same age as the primary |
| detached | each radius below 0.75 of its Roche lobe at periastron |
| eclipses | isotropic orientations, kept where the system eclipses; the deeper eclipse at least 0.04 mag in G, and three of 93 photometric transits in an eclipse |
| brightness | kept in proportion to the volume that the combined light in the band reaches |
| eccentricity | the share of orbits above e = 0.1 of the eclipsing samples, by period and temperature |
| rotation | aligned; pseudo-synchronous below 10 d for cool stars and for hot stars with R/a above 0.1; that of a field star otherwise |
| spectra | BOSZ 2024 at R = 20,000 in three boxes, 3200 to 10,000 K, each star from the box that contains it |

**The twin excess.** With the published excess of twins, [[twin_published]] percent of
such a sample has q above 0.95, since a magnitude-limited eclipsing sample favours equal
stars twice over. Two samples of eclipsing binaries disagree with it: 30 percent of the
293 detached double-lined systems of Eker et al. (2018) have q above 0.95, and 14 percent
of the Gaia DR3 candidates of the detached classes with a primary eclipse of 0.2 mag have
the two depths in the same bin of 0.1 mag (28 percent if every single-eclipse model is a
twin at half its period). The sample drawn here has [[q95]], [[q90]] and [[q70]] percent
above 0.95, 0.9 and 0.7 among the systems with a flux ratio above 0.1, against 30, 49 and
83 in the compilation, and [[equal_depth]] percent of equal depths.

**Weights.** Each system carries the number of DR3 candidates of the light-curve classes
that Mowlavi et al. (2023) call wide (2G-A, 2G-D, 2GE-A, 1G) in its cell, divided by the
number of simulated systems there. The classes hold [[catalogue_all]] candidates between
G_RVS 6.0 and 13.5, of which [[catalogue_outside_classes]] percent have a temperature
outside the four classes; the temperatures are `teff_gspphot`, and the candidates without
one are spread over the classes of their magnitude in proportion. The weighted sample has
the magnitudes and classes of the catalogue and the periods of the model. A second set of
weights is also raked to the catalogue's periods in the bins that the model populates;
[[catalogue_outside_periods]] percent of the candidates lie in other bins, mostly at
periods below those of any detached pair of their class.

[[figure_1]]

The weighted sample has a median period of [[period_p50]] d ([[period_p16]] and
[[period_p84]] d at the 16th and 84th percentiles), [[post_ms]] percent of primaries past
the main sequence, [[ecc_small]] percent of orbits with e below 0.05, a median sum of
fractional radii of [[rho_median]], a median depth of the deeper eclipse of
[[depth_median]] mag, a median projected rotation of the primary of [[vsini_median]] km/s,
and a median of [[transits_median]] transits. The model has more systems beyond 6 d than
the DR3 catalogue, whose candidates were found with half as many photometric transits as
DR4 will have, and fewer below 0.6 d, where the catalogue's classes contain pairs closer
to contact than the model allows.

## The simulated observations

Each star is rendered from the library with its rotation and limb darkening. At every
transit the light fractions and the S/N follow from the eclipse geometry of two
limb-darkened spheres, a velocity offset of 0.17 km/s common to both stars is added, and
the transit takes the resolving power of its CCD row. Photon noise is added on the
detector pixels and the spectrum is interpolated onto the DR4 grid, as in the
[Gaia RVS simulator](../tutorials/gaia-rvs.md). A share of [[in_eclipse]] percent of the
transits falls in an eclipse. The distortion of the eclipsed star's lines is not modelled,
so those transits are measured but kept out of the orbits, as observers keep them out.

[[figure_2]]

## The analysis

Every system is measured under three declarations of the templates, on the same epochs.

| declaration | templates | light fractions |
|---|---|---|
| `injected` | the injected labels and rotation | held at the injected values |
| `classified` | labels with the errors of a classification, drawn once per system: the larger of 150 K and 3 percent in temperature, 0.2 dex in log g, 0.15 dex in [M/H], 20 percent in v sin i | measured |
| `catalogue` | one template for both stars: the temperature of the pair's colour on a lattice of 500 K, log g = 4.5, solar metallicity, synchronous rotation | measured |

The period and the time of the primary's eclipse are declared exactly. The chain is
[`albireo.survey.run_system`](../api/survey.md): templates on a grid of three pixels per
line-spread sigma; `todcor` over 450 km/s on either side with the delivered line-spread
width (11.67 km/s) and the lag-one noise correlation of the product declared; measured
light fractions taken on the transits out of eclipse and held; and
[`assign_by_ephemeris`](../api/rvorbit.md), which fits the two semi-amplitudes and the
systemic velocity with the ephemeris held and decides the order of the two velocities of a
transit by it. The eccentricity is fitted where the injected orbit is eccentric.

The definitions were fixed on a pilot of 300 systems.

- A transit is *usable* where it is out of eclipse, both velocities are measured and the
  table does not flag it after the assignment.
- A velocity is *wrong* where it is more than five quoted errors and 3 km/s from the
  injected one.
- A system is *recovered* where both semi-amplitudes are within 10 percent of the injected
  ones. Masses to 10, 3 and 1 percent need them within 4.5, 1.3 and 0.45 percent.
- The secondary is *detected* where its semi-amplitude divided by its error exceeds the
  value that one in a hundred of [[n_null]] null systems exceeds. A null system is the
  primary alone, at the system's magnitude, analysed with the two templates of the pair.
  The thresholds are [[threshold_injected]], [[threshold_classified]] and
  [[threshold_catalogue]] for the three declarations.
- An *error* is taken against the velocity of the orbit, and so contains the zero point of
  the transit. Where the estimator itself is tested (the scatter over the predicted
  error, the pulls), it is taken against the velocity that was put into the spectrum.
- A share quoted for a range of magnitudes is *weighted* to the catalogue's numbers in the
  cells it pools. The maps by cell, and the curves against S/N, line separation, rotation
  and predicted S/N, are not weighted.

## Recovery of both semi-amplitudes

[[figure_3]]

| G_RVS | injected | classified | catalogue | classified, flux ratio above 0.2 | K1 from one template, flux ratio below 0.1 |
|---|---|---|---|---|---|
| 6 to 9 | [[rec10_injected_6_9]] | [[rec10_classified_6_9]] | [[rec10_catalogue_6_9]] | [[rec10b_classified_6_9]] | [[sb1f_classified_6_9]] |
| 9 to 10 | [[rec10_injected_9_10]] | [[rec10_classified_9_10]] | [[rec10_catalogue_9_10]] | [[rec10b_classified_9_10]] | [[sb1f_classified_9_10]] |
| 10 to 11 | [[rec10_injected_10_11]] | [[rec10_classified_10_11]] | [[rec10_catalogue_10_11]] | [[rec10b_classified_10_11]] | [[sb1f_classified_10_11]] |
| 11 to 12 | [[rec10_injected_11_12]] | [[rec10_classified_11_12]] | [[rec10_catalogue_11_12]] | [[rec10b_classified_11_12]] | [[sb1f_classified_11_12]] |
| 12 to 13 | [[rec10_injected_12_13]] | [[rec10_classified_12_13]] | [[rec10_catalogue_12_13]] | [[rec10b_classified_12_13]] | [[sb1f_classified_12_13]] |
| 13 to 13.5 | [[rec10_injected_13_13p5]] | [[rec10_classified_13_13p5]] | [[rec10_catalogue_13_13p5]] | [[rec10b_classified_13_13p5]] | [[sb1f_classified_13_13p5]] |

Percent of systems with both semi-amplitudes within 10 percent, the four classes weighted
to the catalogue. The last column is for the systems whose secondary has less than a
tenth of the primary's flux: the share whose primary's semi-amplitude is within 10 percent
when the transits are measured with the primary's classified template alone. Over all
systems that share is [[sb1_classified_6_9]] percent at G_RVS 6 to 9 and
[[sb1_classified_11_12]] percent at 11 to 12, since one template measures the blend of two
sets of lines.

A stratum contains every mass ratio from 0.1: [[faint_share]] percent of the weighted
sample has a secondary with less than a fifth of the primary's flux, and [[tenth_share]]
percent less than a tenth. The recovery of the whole stratum is therefore below that of
its double-lined systems at the bright end. With the injected templates half of the
systems with a flux ratio above 0.2 are recovered at G_RVS = [[half_injected]], with
classified templates at [[half_classified]] and with one catalogue template at
[[half_catalogue]].

## Velocities of single transits

[[figure_4]]

With the injected templates the robust scatter of the primary's velocity about the
velocity in its spectrum, over all usable transits with S/N above 10, is
[[scatter_over_predicted]] times the photon-limited prediction for an isolated line system,
and [[scatter_high]] times at S/N above 100. Against the orbit, which adds the zero point of
the transit, the two numbers are [[scatter_over_predicted_orbit]] and
[[scatter_high_orbit]]. The prediction takes the noise of a transit as uniform at its
median, and the photon noise is lower in the cores of the lines, so a ratio slightly below
one is expected. The figure shows the scatter against the orbit. For G-type
primaries it is [[sigma1_G_8]], [[sigma1_G_10]] and [[sigma1_G_12]] km/s in the half
magnitudes that begin at G_RVS = 8, 10 and 12, and for A-type primaries [[sigma1_A_8]],
[[sigma1_A_10]] and [[sigma1_A_12]] km/s. The secondary of a system with a flux ratio above
0.2 has [[sigma2_G_8]], [[sigma2_G_10]] and [[sigma2_G_12]] km/s in the G class. The
squares are the scatter of one transit that the errors of Gaia DR3 imply for single dwarfs
of 5500 to 6500 K: 0.19, 0.36, 1.07 and 5.4 km/s at G_RVS = 6, 8, 10 and 12, the last an
extrapolation from 11.75. The single stars of this simulation are compared with it under
[Checks](#checks-of-the-experiment), and the difference is the first of the
[Limits](#limits).

[[figure_5]]

The pulls are taken against the velocity in the spectrum, and those of the secondaries
over the systems with a flux ratio above 0.2. With the injected templates they have a
robust width of [[pull_width_1]] (primaries) and [[pull_width_2]] (secondaries);
[[pull_beyond3_1]] and [[pull_beyond3_2]] percent lie beyond three quoted errors and
[[pull_beyond5_1]] and [[pull_beyond5_2]] percent beyond five. At S/N above 100 the width
of the primaries' pulls is [[pull_high]], and [[pull_high_orbit]] against the orbit: the
quoted errors do not contain the zero point of a transit. With classified templates the
widths are [[pull1_classified]] and [[pull2_classified]], and with the catalogue template
[[pull1_catalogue]] and [[pull2_catalogue]]: a template that does not match moves the
velocity by more than the quoted error where the S/N is high. Among the systems with a
flux ratio above 0.2 the share of usable transits with a wrong velocity is
[[wrong_injected]], [[wrong_classified]] and [[wrong_catalogue]] percent.

[[figure_6]]

With classified templates, [[both_right_8]] percent of the transits out of eclipse are
usable with both velocities right in the half magnitude that begins at G_RVS = 8,
[[both_right_10]] percent at 10, [[both_right_12]] percent at 12 and [[both_right_13]]
percent at 13. The primary's velocity is usable and right at [[primary_right_8]],
[[primary_right_10]], [[primary_right_12]] and [[primary_right_13]] percent. A transit
that the table flags counts against both.

[[figure_7]]

The lines of the two stars are resolved where their separation exceeds their width, taken
here as the quadrature sum of the line-spread FWHM (27.5 km/s) and 1.6 times the larger
v sin i. At S/N above 30, [[usable_wide]] percent of the transits beyond 1.5 widths are
usable and [[wrong_wide]] percent of those have a wrong velocity; below 0.6 widths the
numbers are [[usable_close]] and [[wrong_close]] percent.

## Detection of the secondary

[[figure_8]]

With classified templates the secondary is detected in [[detected_classified]] percent of
the weighted sample, of which more than half lies in the faintest magnitude. Of the
systems detected, [[purity_classified]] percent are recovered to 10 percent, and of the
systems recovered, [[complete_classified]] percent are detected. The
DPAC search for double lines requires a flux ratio above 0.2 and reaches G_RVS = 12.

## Rotation

[[figure_9]]

The figure shows the scatter of the primary's velocity about the velocity in its
spectrum, over its photon-limited prediction, against v sin i at S/N above 10. The
prediction contains the loss of information to rotation, so a ratio of one at every
rotation means that the estimator loses nothing more.

## Orbits and masses

[[figure_10]]

With classified templates, both minimum masses are within 10 percent for
[[mass10_below_9]] percent of the systems brighter than G_RVS = 9, [[mass10_9_to_11]]
percent at 9 to 11 and [[mass10_11_to_12p5]] percent at 11 to 12.5. They are within 3
percent for [[mass3_below_9]], [[mass3_9_to_11]] and [[mass3_11_to_12p5]] percent, and
within 1 percent for [[mass1_below_9]], [[mass1_9_to_11]] and [[mass1_11_to_12p5]]
percent. For the recovered systems the pulls of the two semi-amplitudes have robust widths
of [[k1_pull_injected]] and [[k2_pull_injected]] with the injected templates and of
[[k1_pull_classified]] and [[k2_pull_classified]] with classified ones, and the systemic
velocity a scatter of [[gamma_sigma_injected]] and [[gamma_sigma_classified]] km/s. The
quoted errors of an orbit contain the photon noise of its transits and not the mismatch
of its templates.

## What the eclipse ephemeris adds

[[figure_11]]

A fit that holds the period and the time of conjunction has two parameters fewer than one
that knows the period alone, and the ephemeris also states which star recedes after the
primary eclipse. The fit with the period alone cannot name the two stars, so its
semi-amplitudes are compared with the injected ones in the order that fits, which
[[free_exchanged]] percent of its tables needed. That comparison gives it the naming for
nothing. With it, the largest difference between the two fits in any half magnitude is
[[held_minus_free]] percentage points: at a known period the ephemeris adds the names of
the stars, and not recovered orbits. The two fits recover [[held_6_11]] and [[free_6_11]]
percent of the systems brighter than G_RVS = 11, [[held_11_12p5]] and [[free_11_12p5]]
percent at 11 to 12.5, and [[held_12p5_13p5]] and [[free_12p5_13p5]] percent beyond.

## What the templates cost

[[figure_12]]

Among the systems with a flux ratio above 0.2 and G_RVS below 10, where photons do not
limit, the injected templates recover [[templates_bright_injected_K]],
[[templates_bright_injected_G]], [[templates_bright_injected_F]] and
[[templates_bright_injected_A]] percent of the K, G, F and A classes, classified templates
[[templates_bright_classified_K]], [[templates_bright_classified_G]],
[[templates_bright_classified_F]] and [[templates_bright_classified_A]] percent, and one
catalogue template [[templates_bright_catalogue_K]], [[templates_bright_catalogue_G]],
[[templates_bright_catalogue_F]] and [[templates_bright_catalogue_A]] percent.

## Effects outside the baseline

Twenty systems of every cell were simulated again with one effect added, with the same
transit times and seeds, and compared with their baseline in the design run. The shares
are those of that subsample, which has equal numbers at every magnitude and class. The
scatter is that of the primary's velocity in km/s over the transits usable in both runs.
The third star is that of the population model: present in a share of the systems that
falls with the period from 96 percent, of 0.1 to 1.2 times the primary's mass and of its
age, at the systemic velocity of the pair to within a few km/s, and unresolved in 68
percent of cases (Tokovinin et al. 2006). The analysis is not told of it.

The emission is that of saturated activity. Every star cooler than 6500 K with a Rossby
number below 0.13 (Wright et al. 2011) has an excess equivalent width between 0.3 and
1.0 Angstrom in the core of Ca II 8542, and 0.6 and 0.8 of it in the two other lines,
which no template has. RAVE measured excesses per line from -0.2 to 1 Angstrom or more in
this band (Zerjal et al. 2013), so the run is at the strong end of observed activity: at
the middle of its range the core of the 8542 line of a G dwarf is filled to about the
continuum. The instrument run draws the background of every transit log-uniformly
between 1/5.5 and 5.5 times the mission median and leaves in the normalisation of a
transit a residual scale, slope and curvature, each with a standard deviation of 2
percent, and the analysis fits the scale of every transit.

[[figure_13]]

| effect | systems affected | recovered before [%] | recovered with it [%] | change of the primary's scatter [%] |
|---|---|---|---|---|
| an unmodelled third star | [[pert_n_third]] | [[pert_before_third]] | [[pert_after_third]] | [[pert_scatter_third]] |
| a third star above 10 percent of the light | [[pert_n_third10]] | [[pert_before_third10]] | [[pert_after_third10]] | [[pert_scatter_third10]] |
| Ca II emission in the line cores | [[pert_n_ca]] | [[pert_before_ca]] | [[pert_after_ca]] | [[pert_scatter_ca]] |
| background and normalisation of a transit | [[pert_n_background]] | [[pert_before_background]] | [[pert_after_background]] | [[pert_scatter_background]] |
| the same, systems fainter than G_RVS = 11 | [[pert_n_background11]] | [[pert_before_background11]] | [[pert_after_background11]] | [[pert_scatter_background11]] |

Third light and emission are the two effects that matter, and they act in opposite
directions. Among the affected systems with a flux ratio above 0.2 and G_RVS below 11,
the median errors of the two semi-amplitudes are [[pert_k1_third10]] and
[[pert_k2_third10]] percent under a third light above a tenth: the lines of the third
star stand at the systemic velocity and draw both measured velocities towards it. Under
the emission they are [[pert_k1_ca]] and [[pert_k2_ca]] percent. A varying background
leaves them at [[pert_k1_background]] and [[pert_k2_background]] percent, since the
transits with less background than the median gain what those with more lose.

The transits in eclipse were measured with free amplitudes. With the injected templates the
primary's velocity is within three predicted errors of the injected one at
[[eclipse_within3]] percent of them, against [[outside_within3]] percent of the transits
out of eclipse, before any distortion of the lines.

## A recovery function

[[figure_14]]

The predicted S/N of the secondary's semi-amplitude is its injected value times the square
root of the summed inverse variances of the photon-limited errors of its transits, each
weighted by the square of the velocity curve. It needs the labels, the magnitude, the
period and the number of transits, and no simulation. The share recovered to 10 percent
follows `top / (1 + exp(-(log10 x - log10 x_half) / w))` with

| declaration | top | x_half | w |
|---|---|---|---|
| injected | [[fn_top_injected]] | [[fn_mid_injected]] | [[fn_width_injected]] |
| classified | [[fn_top_classified]] | [[fn_mid_classified]] | [[fn_width_classified]] |
| catalogue | [[fn_top_catalogue]] | [[fn_mid_catalogue]] | [[fn_width_catalogue]] |

## Expected yield for the catalogue

[[figure_15]]

| limiting G_RVS | candidates | double-lined orbits | masses to 10 percent | masses to 3 percent |
|---|---|---|---|---|
| 10 | [[cand_10]] | [[orbit_10]] ([[orbit_10_lo]] to [[orbit_10_hi]]) | [[mass10_10]] | [[mass3_10]] |
| 11 | [[cand_11]] | [[orbit_11]] ([[orbit_11_lo]] to [[orbit_11_hi]]) | [[mass10_11]] | [[mass3_11]] |
| 12 | [[cand_12]] | [[orbit_12]] ([[orbit_12_lo]] to [[orbit_12_hi]]) | [[mass10_12]] | [[mass3_12]] |
| 13 | [[cand_13]] | [[orbit_13]] ([[orbit_13_lo]] to [[orbit_13_hi]]) | [[mass10_13]] | [[mass3_13]] |
| 13.5 | [[cand_13p5]] | [[orbit_13p5]] ([[orbit_13p5_lo]] to [[orbit_13p5_hi]]) | [[mass10_13p5]] | [[mass3_13p5]] |

Numbers of systems with classified templates, weighted to the DR3 candidates of the
detached classes in the four temperature classes. A double-lined orbit is one whose
secondary is detected and whose semi-amplitudes are both within 10 percent. The ranges are
the 16th to 84th percentiles of a bootstrap over the simulated systems of each cell.
Brighter than 13.5, [[orbit_13p5_pct]] percent of the candidates have a double-lined orbit
and [[mass10_13p5_pct]] percent masses to 10 percent. With the weights raked to the
catalogue's periods, which cover [[raked_coverage]] percent of the candidates, the shares
are [[orbit_13p5_raked_pct]] and [[mass10_13p5_raked_pct]] percent. Gaia DR3 has
5376 double-lined orbits, none fainter than G = 12, and 483 of them among its
eclipsing-binary candidates brighter than G = 11.

## Examples

[[figure_16]]

## Checks of the experiment

- **Recorded systems.** The [[rec_n]] systems of the recorded disentangling benchmarks,
  simulated from their recorded seeds and measured by the same code with the injected
  templates, have [[rec_usable]] percent of their epochs usable as measured, and both
  semi-amplitudes within 5 percent for [[rec_wide]] systems whose lines separate by more
  than 100 km/s and [[rec_close]] of the others. The
  [TODCOR notebook](../tutorials/gaia-rvs-todcor.ipynb) has 93.4 percent, 21 of 22 and 9
  of 11.
- **Single stars.** The primaries of the null run, measured with their own template
  alone, have a scatter about the velocity in the spectrum of [[single_scatter]] times the
  photon-limited prediction at S/N above 10, and pulls of robust width [[single_pull]].
  For those of 5500 to 6500 K that rotate below 20 km/s the scatter of a transit about the
  orbit is [[single_sigma_6]] km/s at G_RVS 6.0 to 6.5, [[single_sigma_8]] km/s at 7.5 to
  8.5, [[single_sigma_10]] km/s at 9.5 to 10.5 and [[single_sigma_12]] km/s at 11.5 to
  12.0. The errors that Gaia DR3 publishes for such stars give [[single_gaia_6]],
  [[single_gaia_8]], [[single_gaia_10]] and [[single_gaia_12]] km/s in the same ranges:
  [[single_ratio_6]], [[single_ratio_8]], [[single_ratio_10]] and [[single_ratio_12]]
  times the simulated scatter
  ([`check-single-stars.csv`](gaia-rvs-eclipsing-binaries/tables/check-single-stars.csv)).
- **Workers.** The tables of 96 systems are identical when they are run by one process and
  by six.
- **Failures.** [[failures_design]] of the [[n_design]] systems raised an error.

## Limits

- The transits of the simulation are at the photon limit. Their S/N is that of the
  science-performance model, which reproduces the S/N that DPAC expects for a source
  (`rv_expected_sig_to_noise`), with the mission's median background at every transit, and
  the velocities reach the precision that this S/N allows. The scatter that the errors of
  Gaia DR3 imply for one transit of a slowly rotating dwarf of 5500 to 6500 K is larger:
  [[single_ratio_8]], [[single_ratio_10]] and [[single_ratio_12]] times that of the
  simulated single stars at G_RVS = 8, 10 and 12. With the zero point taken from both, a
  transit of DR3 has the scatter of a simulated transit [[single_shift_8]],
  [[single_shift_10]] and [[single_shift_12]] mag fainter. The published number is a
  standard deviation over transits whose background differs by a factor of a few, and it
  contains whatever the DR3 pipeline lost; this page does not separate the two. The
  background accounts for a small part: one that is up to 5.5 times the median or below
  it, as in the instrument run, raises the standard deviation of the noise of a transit
  by [[noise_rms_10]] percent at G_RVS = 10 and [[noise_rms_12]] percent at 12 in the S/N
  model, and changes the robust scatter of the primary's velocity by
  [[pert_scatter_background11]] percent for the systems fainter than 11. If the
  transits of DR4 are as DR3 measured them, the magnitude at which a given share of
  systems is recovered is brighter than quoted here by about those amounts, and the
  yields are lower.
- The data and the templates come from one library. The `classified` and `catalogue`
  declarations and the emission run introduce mismatch, but not the difference between a
  model atmosphere and a star.
- The line-spread function is Gaussian here. The DR4 grid, the normalisation and the
  selection flags are those of the draft data model.
- The stars are spheres on Keplerian orbits: no ellipsoidal distortion, reflection or
  spots, and no distortion of the lines in eclipse.
- The tracks are of solar composition, and the metallicity enters the spectra only.
- The weights are those of DR3 candidates of light-curve classes, of which the detached
  systems are a part, and the yield assumes classified templates for every system, no
  third light and no emission.

## Reproduction

```bash
python scripts/build_mist_tracks.py --check
python scripts/rvs_eb_run.py draw
python scripts/rvs_eb_run.py run design
python scripts/rvs_eb_run.py run null
python scripts/rvs_eb_run.py run third-light
python scripts/rvs_eb_run.py run activity
python scripts/rvs_eb_run.py run instrument
python scripts/rvs_eb_run.py run recorded
python scripts/rvs_eb_report.py
```

The three BOSZ boxes must be in the albireo cache (`albireo.fetch_library`). The manifests
of the runs are in
`gaia-rvs-eclipsing-binaries/manifests`, beginning with
[`design.json`](gaia-rvs-eclipsing-binaries/manifests/design.json), and the population in
[`tables/population.csv.gz`](gaia-rvs-eclipsing-binaries/tables/population.csv.gz).

## References

- Choi, J., Dotter, A., Conroy, C., et al. 2016, ApJ, 823, 102
- Eker, Z., Bakis, V., Bilir, S., et al. 2018, MNRAS, 479, 5491
- Katz, D., Sartoretti, P., Guerrier, A., et al. 2023, A&A, 674, A5
- Meszaros, Sz., Bohlin, R., Allende Prieto, C., et al. 2024, A&A, 688, A197
- Moe, M. & Di Stefano, R. 2017, ApJS, 230, 15
- Mowlavi, N., Holl, B., Lecoeur-Taibi, I., et al. 2023, A&A, 674, A16
- Tokovinin, A., Thomas, S., Sterzik, M., & Udry, S. 2006, A&A, 450, 681
- Wright, N. J., Drake, J. J., Mamajek, E. E., & Henry, G. W. 2011, ApJ, 743, 48
- Zerjal, M., Zwitter, T., Matijevic, G., et al. 2013, ApJ, 776, 127
- Zucker, S. & Mazeh, T. 1994, ApJ, 420, 806
- Zucker, S. 2003, MNRAS, 342, 1291
"""

CAPTIONS = (
    (
        "fig01-sample-periods",
        "The simulated sample against the Gaia DR3 candidates of the detached light-curve classes. Top: periods by class of the primary, with the two sets of weights. Bottom: the flat design against the catalogue's counts; the depth of the deeper eclipse at G_RVS below 12; the mass ratio; the sum of the fractional radii.",
    ),
    (
        "fig02-spectra",
        "One transit near quadrature of a typical pair of each class at three magnitudes (grey), with the two components at the resolving power of the instrument, each scaled by its light fraction.",
    ),
    (
        "fig03-recovery",
        "Share of systems with both semi-amplitudes within 10 percent, with classified templates. Cells with fewer than five systems are empty.",
    ),
    (
        "fig04-epoch-error",
        "Robust scatter of the velocity of one transit about its orbit, with the injected templates. It contains the zero point of 0.17 km/s per transit. Bands: 16th to 84th percentiles of the absolute error. The secondary is that of systems with a flux ratio above 0.2. Squares: the scatter of one transit that the errors of Gaia DR3 imply for single dwarfs of 5500 to 6500 K, the same in every panel.",
    ),
    (
        "fig05-pulls",
        "Calibration of the quoted errors, against the velocity in the spectrum. Left: pulls with the injected templates; the secondaries are those of systems with a flux ratio above 0.2. Right: robust width of the primaries' pulls against the S/N of the transit.",
    ),
    (
        "fig06-fate",
        "Transits out of eclipse by what the table gives for them, in the catalogue's mixture of classes.",
    ),
    (
        "fig07-separation",
        "Transits of systems with a flux ratio above 0.2, with classified templates, against the separation of the lines in units of their width.",
    ),
    (
        "fig08-secondary",
        "Share of systems whose secondary is detected, with classified templates. Dashed: the flux ratio of 0.2 that the DPAC search for double lines requires. Cells with fewer than five systems are empty.",
    ),
    (
        "fig09-rotation",
        "Scatter of the primary's velocity about the velocity in its spectrum, over the photon-limited prediction, at S/N above 10.",
    ),
    (
        "fig10-orbits",
        "Cumulative shares of all systems, weighted to the catalogue, with classified templates. A system without an orbit never enters.",
    ),
    (
        "fig11-ephemeris",
        "Recovery with the ephemeris held and with the period alone, weighted to the catalogue's classes. Bars: Wilson intervals at one sigma for the effective number of systems.",
    ),
    (
        "fig12-templates",
        "Recovery of the systems with a flux ratio above 0.2 under the three declarations. Bands: Wilson intervals at one sigma.",
    ),
    (
        "fig13-perturbations",
        "Paired changes against the design run, with classified templates. Bars: 16th to 84th percentiles of a bootstrap over systems.",
    ),
    (
        "fig14-function",
        "Recovery against the predicted S/N of the secondary's semi-amplitude, with the fitted functions.",
    ),
    (
        "fig15-yield",
        "Cumulative numbers weighted to the DR3 catalogue. Bands: 16th to 84th percentiles of a bootstrap over the simulated systems of each cell.",
    ),
    (
        "fig16-cases",
        "Six systems with classified templates. Lines: the injected curves. Filled points: usable transits as assigned by the ephemeris. Open points: flagged transits as measured.",
    ),
)


def render(numbers: dict[str, str]) -> str:
    """The page with every ``[[key]]`` replaced by its number; a missing key is an error."""
    import re

    text = PAGE
    for k, (name, _) in enumerate(FIGURES):
        table, caption = CAPTIONS[k]
        block = (
            figure(k + 1, name, caption)
            .replace(f"tables/{name}.csv", f"tables/{table}.csv")
            .replace(f"`{name}.csv`", f"`{table}.csv`")
        )
        text = text.replace(f"[[figure_{k + 1}]]", block)
    return re.sub(r"\[\[(\w+)\]\]", lambda m: str(numbers[m.group(1)]), text)


def write_population(design: Run) -> None:
    fields = (
        "name klass grvs_bin grvs g_mag g_rp m1 m2 q r1 r2 teff1 teff2 logg1 logg2 mh vsini1 vsini2 "
        "period ecc omega incl gamma k1 k2 light_ratio n_transits age_gyr eep1 rho depth_mag "
        "tertiary_light weight_cell weight"
    ).split()
    path = OUT / "tables" / "population.csv.gz"
    lines = [",".join(fields)]
    for i in range(len(design.population)):
        row = []
        for field in fields:
            value = design.cols[field][i]
            row.append(str(value) if field in ("name", "klass") else f"{value:.6g}")
        lines.append(",".join(row))
    # No file name and no time in the header, so that the same population is the same file.
    with (
        open(path, "wb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as handle,
    ):
        handle.write(("\n".join(lines) + "\n").encode("utf-8"))


def write_manifests(runs) -> None:
    """The manifests of the runs, without the tracebacks of their failures.

    A traceback holds the paths of the machine that made the run. The name of the system
    and the last line of the error are kept.
    """
    for run in runs:
        manifest = json.loads((run.dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["failures"] = [
            {"name": f["name"], "error": f["error"].strip().splitlines()[-1]}
            for f in manifest["failures"]
        ]
        manifest["definitions"] = {
            "usable_epoch": "out of eclipse, both velocities measured, not flagged after the assignment by the ephemeris",
            "wrong_velocity": f"more than {WRONG_SIGMAS:g} quoted errors and {WRONG_KMS:g} km/s from the injected one",
            "recovered": f"both semi-amplitudes of the fit at the held ephemeris within {RECOVERED[1]:g} of the injected ones",
            "detected_secondary": f"K2 / sigma(K2) above the value that {FALSE_ALARM:g} of the null systems exceed",
            "double_lined_by_design": f"flux ratio of at least {BRIGHT_RATIO:g}",
        }
        target = OUT / "manifests" / f"{run.name}.json"
        target.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("--design", default="design", help="the run that stands for the design")
    parser.add_argument("--only", default="", help="comma-separated figure numbers to draw")
    parser.add_argument("--root", default="", help="the directory of the runs, for a trial")
    parser.add_argument("--out", default="", help="write the report under this directory")
    args = parser.parse_args(argv)
    global ROOT, OUT, PAGE_PATH
    if args.root:
        ROOT = Path(args.root)
    if args.out:
        OUT = Path(args.out) / "gaia-rvs-eclipsing-binaries"
        PAGE_PATH = Path(args.out) / "gaia-rvs-eclipsing-binaries.md"
    plt = _mpl()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "manifests").mkdir(parents=True, exist_ok=True)

    def maybe(name):
        path = ROOT / name / "manifest.json"
        if not path.exists() or not json.loads(path.read_text(encoding="utf-8"))["n_systems"]:
            return None
        return Run(name)

    design = Run(args.design)
    null = maybe("null")
    variants = {
        name: run for name in ("third-light", "activity", "instrument") if (run := maybe(name))
    }
    recorded = maybe("recorded")
    data, numbers = compute(design, null, variants, recorded)
    # Measured on a separate draw of 3600 systems with the published twin excess.
    numbers.setdefault("twin_published", "53")
    for key in ("wall", "failures"):
        numbers[f"{key}_design"] = numbers[f"{key}_{design.name.replace('-', '_')}"]
    wanted = {int(x) for x in args.only.split(",") if x} or set(range(1, len(FIGURES) + 1))
    for k, (name, draw) in enumerate(FIGURES):
        if k + 1 not in wanted:
            continue
        for theme_name, theme in THEMES.items():
            with plt.rc_context(style(theme)):
                fig = draw(data, theme)
                fig.savefig(OUT / "figures" / f"{name}-{theme_name}.png", dpi=150)
                plt.close(fig)
        print("drew", name, flush=True)
    write_manifests([run for run in (design, null, recorded, *variants.values()) if run])
    shutil.copyfile(ROOT / "population_summary.json", OUT / "manifests" / "population.json")
    write_population(design)
    (OUT / "numbers.json").write_text(
        json.dumps(numbers, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    try:
        PAGE_PATH.write_text(render(numbers), encoding="utf-8", newline="\n")
        print("wrote", PAGE_PATH)
    except KeyError as missing:
        print("the page was not written: no number for", missing)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
