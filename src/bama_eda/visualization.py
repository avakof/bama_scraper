"""Charts.

Three rules, all of which exist because a chart is believed more readily than a
table:

1. **Every chart states its sample size.** A bar over 3 listings and a bar over
   300 look identical otherwise.
2. **Grouped charts apply the minimum group size.** Segments below it are excluded
   from the plot and the exclusion is written into the subtitle, not silently.
3. **Axes are never truncated to exaggerate a difference.** Count axes start at
   zero; where a log scale is used it is labelled as such.

Persian labels are rendered when a font that supports Persian is installed. No
font is bundled or downloaded; when none is available the chart falls back to an
English technical label and says so, which is better than the tofu boxes that a
missing glyph produces.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display in a batch analysis
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402

#: Fonts that carry Persian glyphs, in preference order. Only used if already
#: installed on the machine; nothing is downloaded or redistributed.
_PERSIAN_FONT_CANDIDATES = (
    "Vazirmatn",
    "Vazir",
    "IRANSans",
    "Sahel",
    "Shabnam",
    "Noto Sans Arabic",
    "Noto Naskh Arabic",
    "Geeza Pro",
    "Al Bayan",
    "Arial Unicode MS",
    "Tahoma",
    "DejaVu Sans",
)

PALETTE = ["#2b6cb0", "#c05621", "#2f855a", "#6b46c1", "#b83280", "#4a5568", "#975a16"]


def detect_persian_font() -> str | None:
    """Return an installed font that can render Persian, or None."""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in _PERSIAN_FONT_CANDIDATES:
        if candidate in available:
            # DejaVu ships with matplotlib but has no Arabic-script coverage.
            if candidate == "DejaVu Sans":
                return None
            return candidate
    return None


class ChartWriter:
    """Writes charts as PNG and SVG, and records what it produced."""

    def __init__(self, out_dir: Path, *, min_group_size: int = 10) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.min_group_size = min_group_size
        self.persian_font = detect_persian_font()
        self.charts: list[dict[str, Any]] = []
        self.skipped: list[dict[str, str]] = []
        if self.persian_font:
            plt.rcParams["font.family"] = [self.persian_font]
        plt.rcParams.update(
            {
                "figure.dpi": 110,
                "savefig.dpi": 150,
                "axes.grid": True,
                "grid.alpha": 0.25,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "figure.autolayout": True,
            }
        )

    # -- helpers -----------------------------------------------------------

    def label(self, value: Any, fallback: str) -> str:
        """Persian label if renderable, otherwise a technical English fallback."""
        text = str(value)
        if self.persian_font:
            return text
        return text if text.isascii() else fallback

    def _save(self, fig: plt.Figure, name: str, *, n: int, note: str = "") -> dict[str, Any]:
        png = self.out_dir / f"{name}.png"
        svg = self.out_dir / f"{name}.svg"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig.savefig(png, bbox_inches="tight")
            fig.savefig(svg, bbox_inches="tight")
        plt.close(fig)
        record = {"name": name, "png": str(png), "svg": str(svg), "n": n, "note": note}
        self.charts.append(record)
        return record

    def _skip(self, name: str, reason: str) -> None:
        """Record a chart that was deliberately not drawn."""
        self.skipped.append({"name": name, "reason": reason})

    def _subtitle(self, ax: plt.Axes, n: int, extra: str = "") -> None:
        """Fold the title and its sample-size line into one left-aligned block.

        `set_title(loc="left")` writes a *different* text object from the centre
        title, so setting both leaves the title printed twice. The centre title is
        cleared first.
        """
        title = ax.get_title()
        ax.set_title("")
        parts = [f"n = {n:,}"]
        if extra:
            parts.append(extra)
        if not self.persian_font:
            parts.append("non-ASCII labels replaced (no Persian-capable font installed)")
        ax.set_title(f"{title}\n{'  |  '.join(parts)}", fontsize=10, loc="left")

    @staticmethod
    def _thousands(ax: plt.Axes, axis: str = "x") -> None:
        """Readable tick labels: 1,600,000 rather than 1.6 x 1e6."""
        from matplotlib.ticker import FuncFormatter

        formatter = FuncFormatter(lambda v, _: f"{v:,.0f}")
        (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(formatter)

    # -- charts ------------------------------------------------------------

    def bar_counts(
        self, series: pd.Series, name: str, title: str, *, top: int = 15, xlabel: str = "count"
    ) -> None:
        data = series.dropna().astype(str)
        if data.empty:
            self._skip(name, "no non-null values")
            return
        counts = data.value_counts()
        eligible = counts[counts >= self.min_group_size].head(top)
        dropped = int((counts < self.min_group_size).sum())
        if eligible.empty:
            self._skip(name, f"no group reaches the minimum size of {self.min_group_size}")
            return

        fig, ax = plt.subplots(figsize=(9, max(3.0, 0.42 * len(eligible))))
        labels = [self.label(v, f"item_{i + 1}") for i, v in enumerate(eligible.index)]
        ax.barh(range(len(eligible)), eligible.to_numpy(), color=PALETTE[0])
        ax.set_yticks(range(len(eligible)))
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(xlabel)
        ax.set_xlim(left=0)  # never truncate a count axis
        for i, value in enumerate(eligible.to_numpy()):
            ax.text(value, i, f" {int(value):,}", va="center", fontsize=8)
        ax.set_title(title)
        self._subtitle(
            ax,
            int(counts.sum()),
            f"{dropped} group(s) below n={self.min_group_size} excluded" if dropped else "",
        )
        self._save(fig, name, n=int(counts.sum()))

    def histogram(
        self,
        series: pd.Series,
        name: str,
        title: str,
        *,
        bins: int = 40,
        xlabel: str = "",
        log_x: bool = False,
    ) -> None:
        values = pd.to_numeric(series, errors="coerce").dropna()
        if len(values) < 5:
            self._skip(name, f"only {len(values)} numeric value(s)")
            return
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.hist(values, bins=bins, color=PALETTE[0], edgecolor="white", linewidth=0.4)
        if not log_x:
            self._thousands(ax, "x")
        ax.set_xlabel(xlabel or series.name or "")
        ax.set_ylabel("advertisements")
        ax.set_ylim(bottom=0)
        if log_x:
            ax.set_xscale("log")
            ax.set_xlabel((xlabel or "") + " (log scale)")
        median = float(values.median())
        ax.axvline(
            median, color=PALETTE[1], linestyle="--", linewidth=1.2, label=f"median = {median:,.0f}"
        )
        ax.legend(fontsize=8)
        ax.set_title(title)
        self._subtitle(ax, len(values))
        self._save(fig, name, n=int(len(values)))

    def scatter(
        self,
        frame: pd.DataFrame,
        x: str,
        y: str,
        name: str,
        title: str,
        *,
        log_y: bool = False,
        xlabel: str = "",
        ylabel: str = "",
        max_points: int = 5000,
        seed: int = 0,
    ) -> None:
        if x not in frame.columns or y not in frame.columns:
            self._skip(name, f"missing column {x!r} or {y!r}")
            return
        block = frame[[x, y]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(block) < 10:
            self._skip(name, f"only {len(block)} complete pair(s)")
            return
        total = len(block)
        # Deterministic subsample: a scatter of 100k points is a solid rectangle.
        if total > max_points:
            block = block.sample(max_points, random_state=seed)

        fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.scatter(block[x], block[y], s=8, alpha=0.28, color=PALETTE[0], edgecolors="none")
        self._thousands(ax, "x")
        ax.set_xlabel(xlabel or x)
        ax.set_ylabel((ylabel or y) + (" (log scale)" if log_y else ""))
        if log_y:
            ax.set_yscale("log")
        ax.set_title(title)
        self._subtitle(
            ax,
            total,
            f"{max_points:,} plotted (deterministic sample)" if total > max_points else "",
        )
        self._save(fig, name, n=total)

    def boxplot_by_group(
        self,
        frame: pd.DataFrame,
        value: str,
        group: str,
        name: str,
        title: str,
        *,
        top: int = 10,
        log_y: bool = True,
        ylabel: str = "",
    ) -> None:
        if value not in frame.columns or group not in frame.columns:
            self._skip(name, f"missing column {value!r} or {group!r}")
            return
        block = frame[[group, value]].copy()
        block[value] = pd.to_numeric(block[value], errors="coerce")
        block = block.dropna()
        block = block[block[value] > 0]
        if block.empty:
            self._skip(name, "no positive values")
            return

        counts = block[group].astype(str).value_counts()
        keep = counts[counts >= self.min_group_size].head(top).index.tolist()
        if not keep:
            self._skip(name, f"no group reaches n={self.min_group_size}")
            return
        subset = block[block[group].astype(str).isin(keep)]
        data = [subset.loc[subset[group].astype(str) == g, value].to_numpy() for g in keep]

        fig, ax = plt.subplots(figsize=(10, 5.5))
        box = ax.boxplot(data, patch_artist=True, showfliers=False, widths=0.6)
        for patch, colour in zip(box["boxes"], PALETTE * 5, strict=False):
            patch.set_facecolor(colour)
            patch.set_alpha(0.55)
        ax.set_xticks(range(1, len(keep) + 1))
        ax.set_xticklabels(
            [f"{self.label(g, f'group_{i + 1}')}\n(n={counts[g]:,})" for i, g in enumerate(keep)],
            fontsize=8,
        )
        if log_y:
            ax.set_yscale("log")
        ax.set_ylabel((ylabel or value) + (" (log scale)" if log_y else ""))
        ax.set_title(title)
        self._subtitle(
            ax,
            int(counts[keep].sum()),
            f"groups below n={self.min_group_size} excluded; outliers hidden",
        )
        self._save(fig, name, n=int(counts[keep].sum()))

    def grouped_share(
        self,
        comparison: pd.DataFrame,
        feature: str,
        name: str,
        title: str,
        *,
        ylabel: str = "",
        top_groups: int = 10,
    ) -> None:
        """One feature's value across the groups of a dimension.

        Bars are the per-group statistic (median, or share present for a flag),
        each labelled with its own n. A group without its sample size is a bar
        that cannot be judged.
        """
        block = comparison[comparison["feature"] == feature].copy()
        if block.empty or "value" not in block.columns:
            self._skip(name, f"no comparison rows for {feature!r}")
            return
        block = block.dropna(subset=["value"]).nlargest(top_groups, "n")
        if len(block) < 2:
            self._skip(name, f"fewer than two comparable groups for {feature!r}")
            return

        fig, ax = plt.subplots(figsize=(9, max(3.0, 0.45 * len(block))))
        labels = [
            f"{self.label(g, f'group_{i + 1}')} (n={int(n):,})"
            for i, (g, n) in enumerate(zip(block["group"], block["n"], strict=False))
        ]
        ax.barh(range(len(block)), block["value"].to_numpy(), color=PALETTE[2])
        ax.set_yticks(range(len(block)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.set_xlabel(ylabel or str(block["statistic"].iloc[0]))
        self._thousands(ax, "x")
        ax.set_title(title)
        grain = str(block["grain"].iloc[0]) if "grain" in block.columns else ""
        extra = (
            "TRIM-LEVEL feature: this is the trim mix per group, not a per-car count"
            if grain == "trim"
            else ""
        )
        dropped = (
            int(block["groups_below_min_excluded"].iloc[0])
            if ("groups_below_min_excluded" in block.columns)
            else 0
        )
        if dropped:
            extra = (
                extra + "; " if extra else ""
            ) + f"{dropped} group(s) below the minimum excluded"
        self._subtitle(ax, int(block["n"].sum()), extra)
        self._save(fig, name, n=int(block["n"].sum()), note=extra)

    def effect_ranking(
        self, ranking: pd.DataFrame, name: str, title: str, *, top: int = 15
    ) -> None:
        """Features ranked by effect size, with the test and n on each bar."""
        if ranking.empty or "effect_size" not in ranking.columns:
            self._skip(name, "no ranked features")
            return
        block = ranking.dropna(subset=["effect_size"]).head(top)
        if block.empty:
            self._skip(name, "no computable effect sizes")
            return

        fig, ax = plt.subplots(figsize=(10, max(3.0, 0.45 * len(block))))
        colours = [
            PALETTE[3] if bool(t) else PALETTE[0]
            for t in block.get("near_tautological", [False] * len(block))
        ]
        ax.barh(range(len(block)), block["effect_size"].abs().to_numpy(), color=colours)
        ax.set_yticks(range(len(block)))
        ax.set_yticklabels(
            [
                f"{self.label(row.label, f'feature_{i + 1}')} [{row.grain}] (n={int(row.n):,})"
                for i, row in enumerate(block.itertuples())
            ],
            fontsize=8,
        )
        ax.invert_yaxis()
        ax.set_xlim(left=0)
        ax.set_xlabel("|effect size| (test-specific)")
        ax.set_title(title)
        tautological = int(block.get("near_tautological", pd.Series([False])).sum())
        self._subtitle(
            ax,
            int(block["n"].max()),
            f"{tautological} bar(s) flagged near-tautological (highlighted)"
            if tautological
            else "rank-based tests; association, not causation",
        )
        self._save(fig, name, n=int(block["n"].max()))

    def correlation_heatmap(
        self,
        matrix: pd.DataFrame,
        name: str,
        title: str,
        *,
        max_features: int = 30,
        note: str = "",
    ) -> None:
        """Symmetric association heatmap on a fixed -1..1 diverging scale.

        The scale is fixed rather than fitted to the data: an auto-scaled colour
        map makes a matrix of weak correlations look identical to one of strong
        ones.
        """
        if matrix.empty or len(matrix) < 2:
            self._skip(name, "matrix too small")
            return
        block = matrix
        if len(block) > max_features:
            # Keep the most connected features rather than the first N.
            order = block.abs().sum().sort_values(ascending=False).head(max_features).index
            block = block.loc[order, order]

        fig, ax = plt.subplots(figsize=(max(7, 0.36 * len(block)), max(6, 0.34 * len(block))))
        image = ax.imshow(block.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(block)))
        ax.set_yticks(range(len(block)))
        labels = [
            self.label(c.replace("deep_", "").replace("spec_", ""), f"f{i + 1}")
            for i, c in enumerate(block.columns)
        ]
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticklabels(labels, fontsize=6)
        ax.grid(False)
        fig.colorbar(image, ax=ax, shrink=0.7, label="association (-1 to 1)")
        ax.set_title(title)
        self._subtitle(
            ax,
            len(block),
            note
            or (
                f"{len(matrix)} features, {max_features} most connected shown"
                if len(matrix) > max_features
                else ""
            ),
        )
        self._save(fig, name, n=len(block), note=note)

    def nonlinearity_scatter_panel(
        self,
        frame: pd.DataFrame,
        pairs: pd.DataFrame,
        name: str,
        title: str,
        *,
        panels: int = 6,
        seed: int = 0,
        max_points: int = 3000,
    ) -> None:
        """Scatter plots of the pairs a linear correlation would misdescribe.

        A number cannot show shape. These are the pairs where Pearson and the rank
        measures disagree most, drawn so the reader can see why.
        """
        from .feature_analysis import to_numeric

        if pairs.empty:
            self._skip(name, "no non-linear pairs")
            return
        block = pairs.head(panels)
        rows = (len(block) + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(13, 4.2 * rows), squeeze=False)
        drawn = 0
        for index, pair in enumerate(block.itertuples()):
            ax = axes[index // 3][index % 3]
            x_col, y_col = pair.feature_a, pair.feature_b
            if x_col not in frame.columns or y_col not in frame.columns:
                ax.axis("off")
                continue
            data = pd.DataFrame(
                {"x": to_numeric(frame[x_col]), "y": to_numeric(frame[y_col])}
            ).dropna()
            if len(data) < 10:
                ax.axis("off")
                continue
            if len(data) > max_points:
                data = data.sample(max_points, random_state=seed)
            ax.scatter(data["x"], data["y"], s=6, alpha=0.25, color=PALETTE[0], edgecolors="none")
            ax.set_xlabel(self.label(pair.label_a, f"x{index}"), fontsize=8)
            ax.set_ylabel(self.label(pair.label_b, f"y{index}"), fontsize=8)
            ax.set_title(
                f"r={pair.linear_pearson_r}  rho={pair.monotone_spearman_rho}  "
                f"dCor={pair.general_value}\n{pair.relationship} (n={pair.n:,})",
                fontsize=8,
                loc="left",
            )
            ax.tick_params(labelsize=7)
            drawn += 1
        for index in range(len(block), rows * 3):
            axes[index // 3][index % 3].axis("off")
        if not drawn:
            plt.close(fig)
            self._skip(name, "no pair had enough numeric data to plot")
            return
        fig.suptitle(title, fontsize=12, x=0.02, ha="left")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        self._save(
            fig, name, n=int(block["n"].max()), note="Pearson would misdescribe every pair shown"
        )

    def missingness_heatmap(
        self, frame: pd.DataFrame, name: str, title: str, *, max_columns: int = 40
    ) -> None:
        candidates = [c for c in frame.columns if frame[c].isna().any()]
        if not candidates:
            self._skip(name, "no column has missing values")
            return
        columns = (
            frame[candidates].isna().mean().sort_values(ascending=False).head(max_columns).index
        )
        matrix = frame[columns].isna().to_numpy().astype(float)

        fig, ax = plt.subplots(figsize=(11, max(4.0, 0.26 * len(columns))))
        ax.imshow(matrix.T, aspect="auto", cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
        ax.set_yticks(range(len(columns)))
        ax.set_yticklabels(columns, fontsize=7)
        ax.set_xlabel("advertisements (row order)")
        ax.set_title(title)
        ax.grid(False)
        self._subtitle(ax, len(frame), "dark = missing")
        self._save(fig, name, n=int(len(frame)))

    def line(
        self,
        x: list[Any],
        y: list[float],
        name: str,
        title: str,
        *,
        xlabel: str = "",
        ylabel: str = "",
        n: int = 0,
        note: str = "",
    ) -> None:
        if len(x) < 2:
            self._skip(name, f"only {len(x)} point(s)")
            return
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.plot(x, y, marker="o", color=PALETTE[0], linewidth=1.6, markersize=4)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.set_title(title)
        self._subtitle(ax, n or len(x), note)
        self._save(fig, name, n=n or len(x))

    def kaplan_meier(self, result: dict[str, Any], name: str, title: str) -> None:
        """Survival curve, drawn only when the estimator agreed to produce one."""
        if not result.get("available"):
            self._skip(name, result.get("reason", "not available") + f": {result.get('note', '')}")
            return
        curve: pd.DataFrame = result["curve"]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.step(
            curve["time_days"],
            curve["survival"],
            where="post",
            color=PALETTE[0],
            linewidth=1.8,
            label="survival",
        )
        ax.fill_between(
            curve["time_days"],
            curve["ci_lower"],
            curve["ci_upper"],
            step="post",
            alpha=0.18,
            color=PALETTE[0],
            label="95% CI",
        )
        ax.set_xlabel("days since first observation")
        ax.set_ylabel("P(still listed)")
        ax.set_ylim(0, 1.02)
        ax.set_xlim(left=0)
        ax.legend(fontsize=8)
        ax.set_title(title)
        self._subtitle(
            ax,
            result["subjects"],
            f"{result['events']} disappearance event(s) - NOT confirmed sales",
        )
        self._save(fig, name, n=result["subjects"], note="event = disappearance, not sale")

    def note_chart(self, name: str, title: str, message: str) -> None:
        """A deliberate placeholder explaining why a chart is absent.

        A missing chart reads as an oversight; a chart that says "not enough data"
        reads as a finding, which is what it is.
        """
        fig, ax = plt.subplots(figsize=(9, 3.2))
        ax.axis("off")
        ax.text(0.5, 0.62, title, ha="center", va="center", fontsize=13, weight="bold")
        ax.text(
            0.5, 0.30, message, ha="center", va="center", fontsize=10, wrap=True, color="#8a4b08"
        )
        self._save(fig, name, n=0, note="placeholder: analysis not supported by the data")

    def manifest(self) -> dict[str, Any]:
        return {
            "charts": self.charts,
            "skipped": self.skipped,
            "persian_font": self.persian_font,
            "persian_rendering": bool(self.persian_font),
            "font_note": (
                f"Persian labels rendered with {self.persian_font!r} (installed on this "
                "machine; not bundled)"
                if self.persian_font
                else "no Persian-capable font installed; non-ASCII labels replaced with "
                "English technical labels rather than rendered as missing glyphs"
            ),
        }
