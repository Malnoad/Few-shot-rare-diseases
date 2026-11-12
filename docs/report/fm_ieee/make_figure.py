"""Draw the proposed workflow only; no patient images or experimental results.

Style follows the figures4papers house style (github.com/ChenLiu-1996/figures4papers,
scientific-figure-making/references/design-theory.md): Arial/Helvetica text, its
blue-green-red-neutral palette, pastel rounded boxes, block arrows, white background,
300 dpi PNG. The code is written here; none of that repository's code is copied.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
W, H = 7.16, 2.38  # inches; IEEE two-column text width

# Palette from figures4papers design-theory.md section 9, plus darker text tones.
INK = "#272727"
GRAY = dict(fill="#EFEFEF", edge="#767676", title="#4D4D4D")
BLUE = dict(fill="#DCE8F6", edge="#3775BA", title="#0F4D92")
GREEN = dict(fill="#DDF3DE", edge="#5FA35F", title="#2D6A2E")
RED = dict(fill="#F6CFCB", edge="#B64342", title="#B64342")


def main() -> None:
    with plt.rc_context({"font.family": ["Arial", "DejaVu Sans"],
                         "font.size": 7.5, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(W, H))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.patch.set_facecolor("white")
        ax.set(xlim=(0, W), ylim=(0, H))
        ax.axis("off")

        def box(x, y, w, h, tone, title, body="", radius=0.08, title_size=8.2):
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                boxstyle=f"round,pad=0,rounding_size={radius}", linewidth=1.0,
                edgecolor=tone["edge"], facecolor=tone["fill"]))
            ax.text(x + w / 2, y + h - 0.09, title, ha="center", va="top",
                    fontsize=title_size, fontweight="bold", color=tone["title"])
            if body:
                ax.text(x + w / 2, y + h - 0.33, body, ha="center", va="top",
                        fontsize=7.2, linespacing=1.35, color=INK)

        def pill(x, y, w, label, tone):
            ax.add_patch(FancyBboxPatch((x, y), w, 0.19,
                boxstyle="round,pad=0,rounding_size=0.09", linewidth=0.8,
                edgecolor=tone["edge"], facecolor="white"))
            ax.text(x + w / 2, y + 0.095, label, ha="center", va="center",
                    fontsize=7.0, color=tone["title"], fontweight="bold")

        def block_arrow(start, end, tone):
            ax.add_patch(FancyArrowPatch(start, end,
                arrowstyle="simple,head_width=0.9,head_length=0.6,tail_width=0.45",
                mutation_scale=14, linewidth=0.8, edgecolor=tone["edge"],
                facecolor=tone["fill"], shrinkA=0, shrinkB=0))

        top, bottom, row_h = 1.20, 0.18, 0.84

        # 1 Locked inputs.
        box(0.05, bottom, 1.40, top + row_h - bottom, GRAY, "Locked inputs",
            "124 rare development\nimages (5 classes)\n22 legacy rare queries\n\n"
            "Hash / RGB / dHash\nduplicate screen\nVerified patient links\n"
            "Eligibility + exclusions")
        block_arrow((1.49, 1.11), (1.75, 1.11), GRAY)

        # 2 Frozen encoders, with the roster as pills.
        ex, ew = 1.79, 1.48
        box(ex, bottom, ew, top + row_h - bottom, BLUE, "Frozen encoders")
        for i, name in enumerate(["RETFound", "DINOv2", "MIRAGE (OCT only)"]):
            pill(ex + 0.14, 1.60 - i * 0.26, ew - 0.28, name, BLUE)
        pill(ex + 0.14, 1.60 - 3 * 0.26, ew - 0.28, "Inception-v3 (reference)", GRAY)
        ax.text(ex + ew / 2, 0.50, "Native transforms, FP32\nFresh caches + identities",
                ha="center", va="center", fontsize=7.2, linespacing=1.35, color=INK)

        # 3A/4A five-way branch and 3B/4B nine-way branch.
        bx, bw, ox, ow = 3.60, 1.86, 5.76, 1.36
        block_arrow((3.31, 1.42), (3.56, 1.62), GREEN)
        block_arrow((3.31, 0.80), (3.56, 0.60), RED)
        box(bx, top, bw, row_h, GREEN, "Five-way K-shot",
            "Paired, nested real supports\n5 grouped folds, K = 1, 2, 4, 8\n"
            "100 prototype draws; 25 probe draws")
        box(bx, bottom, bw, row_h, RED, "Nine-way source audit",
            "1,000 common supports/class\n+ all eligible rare fitting images\n"
            "One real-only prototype head")
        block_arrow((5.48, top + row_h / 2), (5.73, top + row_h / 2), GREEN)
        block_arrow((5.48, bottom + row_h / 2), (5.73, bottom + row_h / 2), RED)
        box(ox, top, ow, row_h, GREEN, "Report",
            "Dev-only nominee\nLegacy + eligible\nexternal queries")
        box(ox, bottom, ow, row_h, RED, "Audit",
            "NORMAL false-rare\nrate, by source:\nKermany, OCTID, OCTDL")

        # No outcome feeds back into nomination or preprocessing.
        fy = 2.21
        ax.plot([ox + ow / 2, ox + ow / 2, ex + ew / 2], [top + row_h, fy, fy],
                color=RED["edge"], linewidth=0.9, linestyle=(0, (3, 2)))
        ax.add_patch(FancyArrowPatch((ex + ew / 2, fy), (ex + ew / 2, top + row_h + 0.01),
            arrowstyle="-|>", mutation_scale=8, color=RED["edge"], linewidth=0.9,
            linestyle=(0, (3, 2)), shrinkA=0, shrinkB=0))
        ax.text((ox + ex + ew) / 2 + 0.3, fy, "×  no outcome feedback into nomination or preprocessing",
                ha="center", va="center", fontsize=7.2, style="italic",
                color=RED["title"], bbox=dict(facecolor="white", edgecolor="none", pad=1.5))

        for ext in ["pdf", "svg", "png"]:
            kwargs = {"dpi": 300} if ext == "png" else {}
            fig.savefig(HERE / f"methods_figure.{ext}", facecolor="white", **kwargs)
        plt.close(fig)
    (HERE / "verification/figure_provenance.json").write_text(json.dumps({
        "kind": "Vector workflow schematic; no measured data or patient images",
        "source": "verification/protocol_v1_0.md sections 3-8",
        "style": "figures4papers house style (palette, Arial/Helvetica, block arrows); own code",
        "dimensions_inches": [W, H],
        "formats": ["pdf", "svg", "png"], "png_dpi": 300,
        "matplotlib": matplotlib.__version__,
        "alt_text": "Locked rare-image inventories and four frozen encoders (RETFound, DINOv2, MIRAGE and an Inception-v3 reference) feed paired five-way K-shot development and reporting, and a separate deterministic nine-way real-only source audit. The first branch nominates on development data only. The second reports NORMAL false-rare rates separately by source. A dashed line marks that no outcome feeds back into nomination or preprocessing."
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
