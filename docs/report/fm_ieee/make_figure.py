"""Draw the proposed workflow only; no patient images or experimental results."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent


def main() -> None:
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 8,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(7.16, 2.38))
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        fig.patch.set_facecolor("white")
        ax.set(xlim=(0, 7.16), ylim=(0, 2.38))
        ax.axis("off")

        def box(x, y, w, h, title, body, fill="#F1F4F7", edge="#27384A"):
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                boxstyle="round,pad=0.012,rounding_size=0.035", linewidth=0.8,
                edgecolor=edge, facecolor=fill))
            ax.text(x + 0.075, y + h - 0.10, title, ha="left", va="top",
                    fontsize=8.5, fontweight="bold", color="#101820")
            ax.text(x + 0.075, y + h - 0.34, body, ha="left", va="top",
                    fontsize=7.7, linespacing=1.32, color="#101820")

        def arrow(start, end, **kwargs):
            ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                mutation_scale=9, color="#27384A", linewidth=0.95, **kwargs))

        box(0.04, 0.48, 1.43, 1.60, "1  LOCK INPUTS",
            "124 rare dev images\n22 legacy rare queries\n\nHash / duplicate screen\nVerified patient links\nEligibility + exclusions")
        box(1.72, 0.48, 1.54, 1.60, "2  FREEZE ENCODERS",
            "RETFound / DINOv2\nMIRAGE (OCT only)\nKeras Inception reference\n\nNative transforms; FP32\nFresh caches + identities")
        box(3.53, 1.21, 2.10, 0.87, "3A  FIVE-WAY K-SHOT",
            "Paired, nested real supports\n5-fold dev: K = 1, 2, 4, 8\n100 prototype draws; fixed probe",
            fill="#EAF2F7")
        box(3.53, 0.20, 2.10, 0.87, "3B  NINE-WAY AUDIT",
            "1,000 common supports/class\n+ all eligible rare fitting images\nOne real-only prototype head",
            fill="#F1F0E9")
        box(5.90, 1.21, 1.20, 0.87, "4A  REPORT",
            "Dev-only nominee\nLegacy / eligible\nexternal queries", fill="#EAF2F7")
        box(5.90, 0.20, 1.20, 0.87, "4B  AUDIT",
            "NORMAL FRR\nKermany / OCTID\n/ OCTDL by source", fill="#F1F0E9")
        arrow((1.49, 1.28), (1.70, 1.28))
        arrow((3.28, 1.48), (3.51, 1.65))
        arrow((3.28, 0.85), (3.51, 0.64))
        arrow((5.65, 1.65), (5.88, 1.65))
        arrow((5.65, 0.64), (5.88, 0.64))
        ax.text(0.04, 2.23, "PROPOSED PROTOCOL  |  No new model results",
                fontsize=9, weight="bold", color="#101820")
        ax.text(0.04, 0.39, "External inventories: terms, labels, dependence;\nfreeze query IDs before inference.",
                fontsize=7.2, color="#101820", linespacing=1.2, va="top")
        ax.text(0.04, 0.04, "No outcome-driven changes. Similarity groups are not patient IDs. All models remain visible.",
                fontsize=7.2, color="#101820")
        for ext in ["pdf", "svg", "png"]:
            kwargs = {"dpi": 300} if ext == "png" else {}
            fig.savefig(HERE / f"methods_figure.{ext}", facecolor="white", **kwargs)
        plt.close(fig)
    (HERE / "verification/figure_provenance.json").write_text(json.dumps({
        "kind": "Vector workflow schematic; no measured data or patient images",
        "source": "verification/protocol_v1_0.md sections 3-8",
        "dimensions_inches": [7.16, 2.38],
        "formats": ["pdf", "svg", "png"], "png_dpi": 300,
        "matplotlib": matplotlib.__version__,
        "alt_text": "Locked rare-image inventories and frozen encoders feed paired five-way K-shot development and reporting, and a separate deterministic nine-way real-only source audit. The first branch nominates on development data only. The second reports NORMAL false-rare rates separately by source."
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
