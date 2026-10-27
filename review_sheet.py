"""
Review sheet for looking at rare-disease images by eye  (GAN-FSL-OCT reproduction, Yoo et al. MBEC 2021)

One HTML page with two parts:
  1. CAM panels for the 22 rare test images, gan and none side by side            (REPLICATION L3, Fig. 10)
  2. 100 random accepted CycleGAN images per disease (fixed seed), each with an
     accept/reject control; the ratings export as CSV                             (REPLICATION L4, Fig. 6)

Training-free: reads test predictions, CAM panels written by `analysis.py cam`, and
data/gan/accepted/<class>/. The images it shows are copied into <page>_files/ next to the
page and referenced by relative path, so the page and that folder open anywhere: from disk,
or over HTTP (an in-app preview cannot load C:/ or D:/ paths). Missing CAM panels are shown
as missing. --jpeg Q copies the sampled GAN images as JPEG instead of PNG, for viewers with a
size cap; ratings still key on the original PNG names.

Usage (from the project root; pass --root to the checkout that holds data/):
  python analysis.py --root D:/Few-shot --out results/analysis cam --variant gan --n 22
  python analysis.py --root D:/Few-shot --out results/analysis cam --variant none --n 22
  python review_sheet.py --root D:/Few-shot --cam results/analysis
Output: .agents/reports/review-sheet.html + review-sheet_files/ (git-ignored). Open the html
file in a browser, or serve the folder and open http://localhost:8765/review-sheet.html:
  python -m http.server 8765 --bind 127.0.0.1 -d .agents/reports
Ratings persist in that browser (localStorage, per origin: keep the same port) until exported.
"""
import argparse
import json
import shutil
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

RARE = ["CSC", "MH", "MacTel", "Stargardt", "RP"]  # == stage4_evaluate.RARE (not imported: it pulls in sklearn)
CAM_VARIANTS = ["gan", "none"]


def sample_files(names, n, seed):
    """n names drawn without replacement from the code-point-sorted list (same on every OS),
    in draw order. The same (names, n, seed) always gives the same sample."""
    names = sorted(names)
    idx = np.random.default_rng(seed).choice(len(names), min(n, len(names)), replace=False)
    return [names[i] for i in idx]


def cam_rows(root, cam_dir):
    """One row per rare test image: true label, and per variant the prediction and panel path (or None)."""
    preds = {v: pd.read_csv(root / f"results/stage3_{v}/test_predictions.csv") for v in CAM_VARIANTS}
    rows = []
    for r in preds["gan"][preds["gan"].label.isin(RARE)].itertuples():
        stem = Path(str(r.path).replace("\\", "/")).stem
        row = dict(label=r.label, stem=stem, path=str(r.path).replace("\\", "/"))
        for v, t in preds.items():
            p = t[t.path == r.path].iloc[0]
            panel = cam_dir / f"cam_{v}" / f"{p.label}_as_{p.pred}_{stem}.jpg"   # stage4_evaluate.cam naming
            row[v] = dict(pred=p.pred, p=round(float(p[f"p_{p.pred}"]), 3),
                          src=panel if panel.exists() else None)
        rows.append(row)
    return sorted(rows, key=lambda r: (RARE.index(r["label"]), r["stem"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="checkout that holds data/ and results/")
    ap.add_argument("--cam", default="results/analysis", help="folder holding cam_gan/ and cam_none/")
    ap.add_argument("--out", default=".agents/reports/review-sheet.html")
    ap.add_argument("--n", type=int, default=100, help="GAN images per disease (paper Fig. 6: 100)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jpeg", type=int, default=0, help="show JPEG copies at this quality (0 = original PNGs)")
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    files = f"{out.stem}_files"                # image copies next to the page, linked by relative URL

    def put(src, rel, jpeg=0):
        """Copy src to <out dir>/<files>/<rel> (re-encoded as JPEG at quality `jpeg` if set) and
        return its URL relative to the page."""
        dst = out.parent / files / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if jpeg:
            import cv2
            dst = dst.with_suffix(".jpg")
            cv2.imwrite(str(dst), cv2.imread(str(src)), [cv2.IMWRITE_JPEG_QUALITY, jpeg])
        else:
            shutil.copyfile(src, dst)
        return dst.relative_to(out.parent).as_posix()

    cams = cam_rows(root, Path(a.cam))
    for r in cams:
        for v in CAM_VARIANTS:
            if r[v]["src"]:
                r[v]["src"] = put(r[v]["src"], f"cam_{v}/{r[v]['src'].name}")   # copied as written
    gan = {}                                   # disease -> [[original PNG name, page-relative image URL], ...]
    for c in RARE:
        d = root / "data/gan/accepted" / c
        names = sample_files([p.name for p in d.glob("*.png")], a.n, a.seed)
        gan[c] = [[f, put(d / f, f"gan/{c}/{f}", a.jpeg)] for f in names]
    data = dict(seed=a.seed, n=a.n, jpeg=a.jpeg, built=date.today().isoformat(), cams=cams, gan=gan)

    out.write_text(TEMPLATE.replace("__DATA__", json.dumps(data).replace("</", "<\\/")), encoding="utf-8")
    found = {v: sum(r[v]["src"] is not None for r in cams) for v in CAM_VARIANTS}
    print(f"review sheet: {out}  (CAM panels found {found} of {len(cams)}; "
          f"GAN images {', '.join(f'{c} {len(g)}' for c, g in gan.items())}; seed {a.seed})")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rare-disease review sheet</title>
<style>
  :root { color-scheme: dark; }
  .sheet { background: #0b0d10; color: #fff; padding: 16px; border-radius: var(--radius, 10px);
           font: 14px/1.45 var(--font-sans, system-ui, sans-serif); }
  h1 { font-size: 18px; margin: 0 0 4px; } h2 { font-size: 15px; margin: 16px 0 6px; }
  p { margin: 6px 0; } code { font-family: var(--font-mono, ui-monospace, monospace); font-size: 12px;
         background: #1c2128; padding: 1px 4px; border-radius: 4px; }
  .muted { color: #d0d7de; } .small { font-size: 12px; }
  button, input { font: inherit; color: #fff; background: #1c2128; border: 1px solid #8b949e;
                  border-radius: 6px; padding: 4px 10px; }
  button { cursor: pointer; } button:hover { border-color: #fff; }
  :focus-visible { outline: 3px solid #79c0ff; outline-offset: 2px; }
  .tabs { display: flex; flex-wrap: wrap; gap: 6px; margin: 12px 0; }
  .tabs button[aria-selected=true] { background: #fff; color: #000; border-color: #fff; font-weight: 600; }
  .note { border-left: 3px solid #79c0ff; padding: 4px 10px; background: #11161d; margin: 8px 0; }
  .warn { border-left-color: #f0b72f; }
  /* CAM rows: original | gan overlay | none overlay, cropped from the 598x299 panels with CSS */
  .cam { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; margin: 4px 0 14px; }
  .cam figure { margin: 0; } .cam figcaption { font-size: 12px; margin-top: 2px; }
  .sq { aspect-ratio: 1; background-size: 200% 100%; background-repeat: no-repeat; border: 1px solid #484f58; }
  .sq.missing { display: flex; align-items: center; justify-content: center; text-align: center; padding: 6px;
                border: 2px dashed #8b949e; font-size: 12px; }
  .ok { color: #7ee787; } .bad { color: #ffa198; }
  /* GAN rating grid */
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px; }
  .tile { border: 2px solid #484f58; border-radius: 6px; padding: 6px; background: #11161d; }
  .tile.accept { border: 2px solid #3fb950; } .tile.reject { border: 2px dashed #f85149; }
  .tile.zoom { grid-column: 1 / -1; } .tile.zoom img { max-width: 512px; }
  .tile img { display: block; width: 100%; aspect-ratio: 1; cursor: zoom-in; }
  .tile .meta { display: flex; justify-content: space-between; font-size: 12px; margin: 4px 0; gap: 4px; }
  .tile .file { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #d0d7de; }
  .tile .btns { display: flex; gap: 6px; } .tile .btns button { flex: 1; padding: 3px 4px; }
  button.acc[aria-pressed=true] { background: #0f5132; border-color: #3fb950; font-weight: 600; }
  button.rej[aria-pressed=true] { background: #67060c; border-color: #f85149; font-weight: 600; }
  table { border-collapse: collapse; margin: 8px 0; } th, td { border: 1px solid #484f58; padding: 4px 10px; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  textarea { width: 100%; box-sizing: border-box; height: 160px; background: #0b0d10; color: #fff;
             border: 1px solid #8b949e; font: 12px var(--font-mono, ui-monospace, monospace); }
  [hidden] { display: none !important; }
</style></head>
<body><div class="sheet">
<h1>Rare-disease review sheet</h1>
<p class="muted small" id="meta"></p>
<p><label>Rater <input id="rater" size="14" placeholder="initials"></label></p>
<div class="tabs" role="tablist" id="tabs"></div>
<div id="panes"></div>
</div>
<script>
const D = __DATA__;
const RARE = Object.keys(D.gan);
// Ratings are keyed and exported by the original PNG path relative to the data checkout.
for (const c of RARE) D.gan[c] = D.gan[c].map(([name, src]) => ({ src, file: `data/gan/accepted/${c}/${name}` }));
const KEY = "octfsl-gan-ratings-seed" + D.seed;
const store = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
                set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} } };
const ratings = JSON.parse(store.get(KEY) || "{}");           // file -> {rating, at}
const el = (tag, attrs = {}, ...kids) => { const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) k === "class" ? e.className = v : k.startsWith("on") ? e[k] = v : e.setAttribute(k, v);
  e.append(...kids); return e; };

document.getElementById("meta").textContent =
  `Built ${D.built} by review_sheet.py · GAN sample: ${D.n} per disease from data/gan/accepted/, seed ${D.seed}`;
const rater = document.getElementById("rater");
rater.value = store.get(KEY + "-rater") || "";
rater.oninput = () => store.set(KEY + "-rater", rater.value);

// ---------- tabs ----------
const tabs = document.getElementById("tabs"), panes = document.getElementById("panes"), tabBtn = {};
function addTab(id, label, pane) {
  pane.hidden = true; pane.id = "pane-" + id; panes.append(pane);
  tabBtn[id] = el("button", { role: "tab", "aria-selected": "false", onclick: () => show(id) }, label);
  tabs.append(tabBtn[id]);
}
function show(id) {
  for (const [k, b] of Object.entries(tabBtn)) { b.setAttribute("aria-selected", k === id); document.getElementById("pane-" + k).hidden = k !== id; }
  if (id === "export") renderExport();
}

// ---------- part 1: CAM ----------
const camPane = el("div");
const nFound = v => D.cams.filter(r => r[v].src).length;
const nRight = v => D.cams.filter(r => r[v].pred === r.label).length;
camPane.append(
  el("p", {}, `${D.cams.length} rare test images. Correct: gan ${nRight("gan")}/${D.cams.length}, none ${nRight("none")}/${D.cams.length} (seed-42 heads).`),
  el("div", { class: "note small" },
    "How to read: each overlay is the CAM of the class that variant ", el("b", {}, "predicted"),
    " (not the true class), ReLU(A·W[:, c]) on the 8×8 mixed10 map, bilinearly upsampled. Each map is scaled to its own maximum, so colour shows ",
    el("b", {}, "where"), " evidence sits within one image, not how strong it is: do not compare intensity across panels or between gan and none. Colormap: OpenCV JET, 40% over the image."));
for (const v of ["gan", "none"]) if (nFound(v) < D.cams.length) camPane.append(el("div", { class: "note warn small" },
  `${v}: ${nFound(v)} of ${D.cams.length} panels found. Generate with `,
  el("code", {}, `analysis.py --root <data checkout> --out results/analysis cam --variant ${v} --n 22`), ", then rebuild this sheet."));
let lastLabel = null;
for (const r of D.cams) {
  if (r.label !== lastLabel) { camPane.append(el("h2", {}, r.label)); lastLabel = r.label; }
  const base = r.gan.src || r.none.src;
  const sq = (src, side, label, missing) => src
    ? el("div", { class: "sq", role: "img", "aria-label": label, style: `background-image:url("${src}");background-position:${side}` })
    : el("div", { class: "sq missing" }, missing);
  const cap = v => { const ok = r[v].pred === r.label;
    return el("figcaption", {}, `${v}: `, el("span", { class: ok ? "ok" : "bad" }, `${ok ? "✓" : "✗"} ${r[v].pred}`), ` p=${r[v].p.toFixed(3)}`); };
  camPane.append(el("div", { class: "small muted" }, `${r.path} · true ${r.label}`), el("div", { class: "cam" },
    el("figure", {}, sq(base, "0 0", `${r.stem} original`, "image not available (no panel)"), el("figcaption", {}, "original (256→299 px)")),
    el("figure", {}, sq(r.gan.src, "100% 0", `${r.stem} gan CAM overlay`, "gan panel missing"), cap("gan")),
    el("figure", {}, sq(r.none.src, "100% 0", `${r.stem} none CAM overlay`, "none panel missing"), cap("none"))));
}
addTab("cam", "CAM: gan vs none", camPane);

// ---------- part 2: GAN ratings ----------
function save() { store.set(KEY, JSON.stringify(ratings)); for (const c of RARE) tabBtn[c].textContent = tabLabel(c); }
const tabLabel = c => `${c} ${D.gan[c].filter(g => ratings[g.file]).length}/${D.gan[c].length}`;
function rate(g, tile, value) {
  if (!value || ratings[g.file]?.rating === value) delete ratings[g.file];   // pressing again clears
  else ratings[g.file] = { rating: value, at: new Date().toISOString() };
  paint(g, tile); save();
}
function paint(g, tile) {
  const r = ratings[g.file]?.rating || "";
  tile.className = "tile " + r + (tile.classList.contains("zoom") ? " zoom" : "");
  tile.querySelector(".acc").setAttribute("aria-pressed", r === "accept");
  tile.querySelector(".rej").setAttribute("aria-pressed", r === "reject");
  tile.querySelector(".state").textContent = r ? r.toUpperCase() + "ED" : "unrated";
}
for (const c of RARE) {
  const grid = el("div", { class: "grid" });
  D.gan[c].forEach((g, i) => {
    const name = g.file.split("/").pop();
    const tile = el("div", { class: "tile", tabindex: "0", "aria-label": `${c} sample ${i + 1}` },
      el("img", { src: g.src, alt: `${c} CycleGAN sample ${i + 1}: ${name}`, loading: "lazy", title: "click to zoom" }),
      el("div", { class: "meta" }, el("span", {}, `#${i + 1} `, el("b", { class: "state" }, "unrated")), el("span", { class: "file", title: name }, name)),
      el("div", { class: "btns" },
        el("button", { class: "acc", "aria-pressed": "false", onclick: () => rate(g, tile, "accept") }, "✓ Accept"),
        el("button", { class: "rej", "aria-pressed": "false", onclick: () => rate(g, tile, "reject") }, "✗ Reject")));
    tile.querySelector("img").onclick = () => tile.classList.toggle("zoom");
    tile.onkeydown = e => { const k = e.key.toLowerCase();
      if (k === "a") rate(g, tile, "accept"); else if (k === "r") rate(g, tile, "reject");
      else if (k === "backspace" || k === "delete") rate(g, tile, null); };
    paint(g, tile); grid.append(tile);
  });
  addTab(c, tabLabel(c), el("div", {},
    el("div", { class: "note small" }, `Would a clinician accept this as a real ${c} B-scan? Fig. 6 protocol: 100 random samples per class. `,
      "These come from the set that already passed the automatic QC, so the acceptance rate here is conditional on QC and is not the paper's raw rate. ",
      "Keys on a focused tile: A accept, R reject, Delete clear. Click an image to zoom."),
    D.jpeg ? el("div", { class: "note warn small" }, `Shown as JPEG copies (quality ${D.jpeg}) of the PNGs to fit a size cap. `,
      "Fine artefacts may differ: for the rating of record, use the sheet built without --jpeg.") : "", grid));
}

// ---------- export ----------
const csvCell = s => /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
function csv() {
  const lines = ["rater,disease,sample,file,rating,rated_at,seed"];
  for (const c of RARE) D.gan[c].forEach((g, i) => { const r = ratings[g.file] || {};
    lines.push([rater.value, c, i + 1, g.file, r.rating || "", r.at || "", D.seed].map(x => csvCell(String(x))).join(",")); });
  return lines.join("\n") + "\n";
}
const summary = el("tbody"), text = el("textarea", { readonly: "", "aria-label": "ratings CSV" });
function renderExport() {
  summary.replaceChildren(...RARE.map(c => { const rs = D.gan[c].map(g => ratings[g.file]?.rating);
    const acc = rs.filter(r => r === "accept").length, rej = rs.filter(r => r === "reject").length;
    return el("tr", {}, el("td", {}, c), el("td", {}, `${acc + rej}/${rs.length}`), el("td", {}, acc), el("td", {}, rej),
      el("td", {}, acc + rej ? `${(100 * acc / (acc + rej)).toFixed(0)}%` : "–")); }));
  text.value = csv();
}
addTab("export", "Export CSV", el("div", {},
  el("table", {}, el("thead", {}, el("tr", {}, ...["disease", "rated", "accepted", "rejected", "accepted of rated"].map(h => el("th", {}, h)))), summary),
  el("p", {}, el("button", { onclick: () => { const a = el("a", { href: URL.createObjectURL(new Blob([csv()], { type: "text/csv" })),
      download: `gan_ratings_seed${D.seed}_${rater.value || "rater"}.csv` }); document.body.append(a); a.click(); a.remove(); } }, "Download CSV"),
    el("span", { class: "small muted" }, " If the download is blocked here, copy the text below.")),
  text,
  el("p", { class: "small muted" }, "Unrated images are exported with an empty rating, so the file always lists the full sample. Ratings are kept in this browser (localStorage) until exported.")));
show("cam");
</script></body></html>
"""

if __name__ == "__main__":
    main()
