"""
Review sheet for looking at rare-disease images by eye  (GAN-FSL-OCT reproduction, Yoo et al. MBEC 2021)

Built for a non-expert rater: every question is answered by comparing images, not by medical
knowledge. The results are a visual check by the author, not the paper's ophthalmologist rating.
One HTML page with two parts:
  1. CAM panels for the 22 rare test images, gan and none side by side. Per overlay the rater
     says where the strongest colour sits (on the abnormality / retina elsewhere / off the
     retina / can't tell)                                                         (REPLICATION L3, Fig. 10)
  2. 100 random accepted CycleGAN images per disease (fixed seed), each shown next to the
     NORMAL scan it was translated from, under a strip of the real training scans of that
     disease (data/gan/real/<class>, what the CycleGAN learned to imitate). The rater picks
     looks real / flawed (with reasons) / unsure                                  (REPLICATION L4, Fig. 6)
Answers export as one long-format CSV.

Training-free: reads test predictions, CAM panels written by `analysis.py cam`, and
data/gan/{accepted,normal_src,real}/. The images it shows are copied into <page>_files/ next
to the page and referenced by relative path, so the page and that folder open anywhere: from
disk, or over HTTP (an in-app preview cannot load C:/ or D:/ paths). Missing images are shown
as missing. --jpeg Q copies the GAN, source and real images as JPEG instead of PNG, for
viewers with a size cap; answers still key on the original PNG names.

Usage (from the project root; pass --root to the checkout that holds data/):
  python analysis.py --root D:/Few-shot --out results/analysis cam --variant gan --n 22
  python analysis.py --root D:/Few-shot --out results/analysis cam --variant none --n 22
  python review_sheet.py --root D:/Few-shot --cam results/analysis
Output: .agents/reports/review-sheet.html + review-sheet_files/ (git-ignored). Open the html
file in a browser, or serve the folder and open http://localhost:8765/review-sheet.html:
  python -m http.server 8765 --bind 127.0.0.1 -d .agents/reports
Answers persist in that browser (localStorage, per origin: keep the same port) until exported.
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


def source_name(gan_name):
    """stage2 `generate` names outputs <class>_gan_<source stem>.png after the data/gan/normal_src image."""
    return gan_name.split("_gan_", 1)[-1]


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
        return its URL relative to the page, or None if src is missing."""
        if not src.exists():
            return None
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
    gan, real = {}, {}             # gan: disease -> [[PNG name, image URL, source URL], ...]; real: disease -> [URL, ...]
    for c in RARE:
        d = root / "data/gan/accepted" / c
        names = sample_files([p.name for p in d.glob("*.png")], a.n, a.seed)
        gan[c] = [[f, put(d / f, f"gan/{c}/{f}", a.jpeg),
                   put(root / "data/gan/normal_src" / source_name(f), f"source/{source_name(f)}", a.jpeg)]
                  for f in names]
        real[c] = [put(p, f"real/{c}/{p.name}", a.jpeg) for p in sorted((root / "data/gan/real" / c).glob("*.png"))]
    data = dict(seed=a.seed, n=a.n, jpeg=a.jpeg, built=date.today().isoformat(), cams=cams, gan=gan, real=real)

    out.write_text(TEMPLATE.replace("__DATA__", json.dumps(data).replace("</", "<\\/")), encoding="utf-8")
    found = {v: sum(r[v]["src"] is not None for r in cams) for v in CAM_VARIANTS}
    print(f"review sheet: {out}  (CAM panels found {found} of {len(cams)}; "
          f"GAN images {', '.join(f'{c} {len(g)}' for c, g in gan.items())}; "
          f"sources found {sum(s is not None for g in gan.values() for _, _, s in g)}; "
          f"real scans {', '.join(f'{c} {len(r)}' for c, r in real.items())}; seed {a.seed})")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rare-disease review sheet</title>
<style>
  :root { color-scheme: dark; }
  .sheet { background: #0b0d10; color: #fff; padding: 16px; border-radius: var(--radius, 10px);
           font: 14px/1.45 var(--font-sans, system-ui, sans-serif); }
  h1 { font-size: 18px; margin: 0 0 4px; } h2 { font-size: 15px; margin: 16px 0 6px; }
  p { margin: 6px 0; } ol { margin: 4px 0; padding-left: 22px; }
  code { font-family: var(--font-mono, ui-monospace, monospace); font-size: 12px;
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
  .badge { display: inline-block; font-size: 11px; font-weight: 600; padding: 0 6px; border-radius: 9px;
           border: 1px solid #f0b72f; color: #f0b72f; margin-left: 6px; }
  /* answer buttons: pressed state = filled + bold, never colour alone */
  .opts { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
  .opts button { padding: 2px 8px; font-size: 12px; }
  button[aria-pressed=true] { background: #fff; color: #000; border-color: #fff; font-weight: 700; }
  button.real[aria-pressed=true] { background: #0f5132; color: #fff; border-color: #3fb950; }
  button.flawed[aria-pressed=true] { background: #67060c; color: #fff; border-color: #f85149; }
  button.unsure[aria-pressed=true] { background: #5a3e00; color: #fff; border-color: #f0b72f; }
  /* CAM rows: original | gan overlay | none overlay, cropped from the 598x299 panels with CSS */
  .cam { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; margin: 4px 0 14px; }
  .cam figure { margin: 0; } .cam figcaption { font-size: 12px; margin-top: 2px; }
  .sq { aspect-ratio: 1; background-size: 200% 100%; background-repeat: no-repeat; border: 1px solid #484f58; }
  .sq.missing, .missing { display: flex; align-items: center; justify-content: center; text-align: center; padding: 6px;
                          border: 2px dashed #8b949e; font-size: 12px; aspect-ratio: 1; box-sizing: border-box; }
  .ok { color: #7ee787; } .bad { color: #ffa198; }
  /* real reference scans, kept in view while scrolling a disease tab */
  .refs { position: sticky; top: 0; z-index: 2; background: #0b0d10; padding: 6px 0; border-bottom: 1px solid #484f58; }
  .strip { display: flex; gap: 4px; overflow-x: auto; padding-bottom: 4px; }
  .strip img { height: 110px; aspect-ratio: 1; cursor: zoom-in; border: 1px solid #484f58; }
  .strip img.big { height: 256px; cursor: zoom-out; }
  /* GAN rating grid: before (source NORMAL) | after (generated) */
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(330px, 1fr)); gap: 10px; margin-top: 10px; }
  .tile { border: 2px solid #484f58; border-radius: 6px; padding: 6px; background: #11161d; }
  .tile.real { border: 2px solid #3fb950; } .tile.flawed { border: 2px dashed #f85149; } .tile.unsure { border: 2px dotted #f0b72f; }
  .tile.zoom { grid-column: 1 / -1; } .tile.zoom .pair { max-width: 1040px; }
  .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; cursor: zoom-in; }
  .pair figure { margin: 0; } .pair figcaption { font-size: 11px; color: #d0d7de; }
  .pair img { display: block; width: 100%; aspect-ratio: 1; }
  .tile .meta { display: flex; justify-content: space-between; font-size: 12px; margin: 4px 0; gap: 4px; }
  .tile .file { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: #d0d7de; }
  .tile .btns { display: flex; gap: 6px; } .tile .btns button { flex: 1; padding: 3px 4px; }
  .tile .why { font-size: 12px; margin-top: 4px; }
  table { border-collapse: collapse; margin: 8px 0; } th, td { border: 1px solid #484f58; padding: 4px 10px; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  textarea { width: 100%; box-sizing: border-box; height: 160px; background: #0b0d10; color: #fff;
             border: 1px solid #8b949e; font: 12px var(--font-mono, ui-monospace, monospace); }
  [hidden] { display: none !important; }
</style></head>
<body><div class="sheet">
<h1>Rare-disease review sheet</h1>
<p class="muted small" id="meta"></p>
<div class="note">This sheet is built for a <b>non-expert</b>: every question is answered by <b>comparing images</b>, not by medical knowledge.
  When you are not sure, choose “Unsure” or “Can't tell” instead of guessing. Report the results as a visual check by the author,
  not a clinician: they do not replicate the paper's ophthalmologist rating (Fig. 6).</div>
<p><label>Rater <input id="rater" size="14" placeholder="initials"></label></p>
<div class="tabs" role="tablist" id="tabs"></div>
<div id="panes"></div>
</div>
<script>
const D = __DATA__;
const RARE = Object.keys(D.gan);
// Answers are keyed and exported by the original PNG path relative to the data checkout.
for (const c of RARE) D.gan[c] = D.gan[c].map(([name, src, source]) => ({ src, source,
  file: `data/gan/accepted/${c}/${name}`, sourceFile: `data/gan/normal_src/${name.split("_gan_").pop()}` }));
const KEY = "octfsl-review-v2-seed" + D.seed;
const store = { get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
                set(k, v) { try { localStorage.setItem(k, v); } catch (e) {} } };
const S = Object.assign({ gan: {}, cam: {} }, JSON.parse(store.get(KEY) || "{}"));   // gan: file -> {answer, reasons, at}; cam: "v|path" -> {answer, at}
const save = () => { store.set(KEY, JSON.stringify(S)); for (const c of RARE) tabBtn[c].textContent = tabLabel(c); tabBtn.cam.textContent = camLabel(); };
const now = () => new Date().toISOString();
const el = (tag, attrs = {}, ...kids) => { const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) k === "class" ? e.className = v : k.startsWith("on") ? e[k] = v : e.setAttribute(k, v);
  e.append(...kids); return e; };
// One row of mutually exclusive answer buttons; pressing the chosen one again clears it.
function options(choices, get, set, label) {
  const box = el("div", { class: "opts", role: "group", "aria-label": label });
  const paint = () => box.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", b.dataset.v === get()));
  for (const [v, text, cls] of choices)
    box.append(el("button", { class: cls || "", "data-v": v, "aria-pressed": "false", onclick: () => { set(get() === v ? null : v); paint(); } }, text));
  paint(); return box;
}

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
const WHERE = [["abnormality", "On the abnormality"], ["retina", "On the retina, elsewhere"], ["off", "Off the retina"], ["cant", "Can't tell"]];
const camKey = (v, r) => `${v}|${r.path}`;
const camLabel = () => `1. CAM ${Object.keys(S.cam).length}/${2 * D.cams.length}`;
const camPane = el("div");
const nFound = v => D.cams.filter(r => r[v].src).length;
const nRight = v => D.cams.filter(r => r[v].pred === r.label).length;
camPane.append(
  el("div", { class: "note" }, el("b", {}, "Question: where does the model look?"),
    el("ol", {},
      el("li", {}, "Look at the original scan (left). Find the part that looks unusual: compare it with the real scans of that disease in the disease tabs, or with a smooth, evenly layered retina."),
      el("li", {}, "In each overlay, find the strongest colour: red is strongest, then yellow; blue means no evidence."),
      el("li", {}, "Say where it sits: on the unusual part, on the retina elsewhere, or off the retina (black background, image edges, the black bands). Off the retina means the model may use a shortcut, not the disease."),
      el("li", {}, "Compare gan with none, especially in rows marked “models disagree”.")),
    el("span", { class: "small muted" }, "This needs no medical knowledge: you judge position, not diagnosis.")),
  el("p", {}, `${D.cams.length} rare test images. Correct: gan ${nRight("gan")}/${D.cams.length}, none ${nRight("none")}/${D.cams.length} (seed-42 heads).`),
  el("div", { class: "note small" },
    "Technical: each overlay is the CAM of the class that variant ", el("b", {}, "predicted"),
    " (not the true class), ReLU(A·W[:, c]) on the 8×8 mixed10 map, bilinearly upsampled, so blobs look finer than they are. Each map is scaled to its own maximum, so colour shows ",
    el("b", {}, "where"), " evidence sits within one image, not how strong it is: do not compare intensity across panels. Colormap: OpenCV JET, 40% over the image."));
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
  const fig = v => { const ok = r[v].pred === r.label;
    return el("figure", {}, sq(r[v].src, "100% 0", `${r.stem} ${v} CAM overlay`, `${v} panel missing`),
      el("figcaption", {}, `${v}: `, el("span", { class: ok ? "ok" : "bad" }, `${ok ? "✓" : "✗"} ${r[v].pred}`), ` p=${r[v].p.toFixed(3)}`),
      options(WHERE, () => S.cam[camKey(v, r)]?.answer,
        a => { if (a) S.cam[camKey(v, r)] = { answer: a, at: now() }; else delete S.cam[camKey(v, r)]; save(); },
        `${r.stem} ${v}: where is the strongest colour?`)); };
  camPane.append(el("div", { class: "small muted" }, `${r.path} · true ${r.label}`,
      r.gan.pred !== r.none.pred ? el("span", { class: "badge" }, "models disagree") : ""),
    el("div", { class: "cam" },
      el("figure", {}, sq(base, "0 0", `${r.stem} original`, "image not available (no panel)"), el("figcaption", {}, "original (256→299 px)")),
      fig("gan"), fig("none")));
}
addTab("cam", camLabel(), camPane);

// ---------- part 2: GAN images ----------
const ANSWERS = [["real", "✓ Looks real", "real"], ["flawed", "✗ Flawed", "flawed"], ["unsure", "? Unsure", "unsure"]];
const STATE = { real: "LOOKS REAL", flawed: "FLAWED", unsure: "UNSURE" };
const REASONS = [["artefact", "Grid, stripes or noise"], ["blur", "Blurry or smeared"], ["layers", "Broken or doubled layers"],
                 ["nochange", "No change from before"], ["unlike", "Unlike the real scans"], ["other", "Other"]];
const tabLabel = c => `${c} ${D.gan[c].filter(g => S.gan[g.file]).length}/${D.gan[c].length}`;
function answer(g, tile, a) {
  const prev = S.gan[g.file];
  if (!a || prev?.answer === a) delete S.gan[g.file];                        // pressing again clears
  else S.gan[g.file] = { answer: a, reasons: a === "flawed" ? prev?.reasons || [] : [], at: now() };
  paint(g, tile); save();
}
function toggleReason(g, tile, k) {
  const s = S.gan[g.file]; if (s?.answer !== "flawed") return;
  s.reasons = s.reasons.includes(k) ? s.reasons.filter(x => x !== k) : [...s.reasons, k];
  paint(g, tile); save();
}
function paint(g, tile) {
  const s = S.gan[g.file], a = s?.answer || "";
  tile.className = "tile " + a + (tile.classList.contains("zoom") ? " zoom" : "");
  tile.querySelectorAll(".btns button").forEach(b => b.setAttribute("aria-pressed", b.dataset.v === a));
  tile.querySelector(".state").textContent = STATE[a] || "unrated";
  tile.querySelector(".why").hidden = a !== "flawed";
  tile.querySelectorAll(".why button").forEach(b => b.setAttribute("aria-pressed", !!s?.reasons?.includes(b.dataset.v)));
}
const img = (src, alt) => src ? el("img", { src, alt, loading: "lazy" }) : el("div", { class: "missing" }, "image missing");
for (const c of RARE) {
  const grid = el("div", { class: "grid" });
  D.gan[c].forEach((g, i) => {
    const name = g.file.split("/").pop();
    const tile = el("div", { class: "tile", tabindex: "0", "aria-label": `${c} sample ${i + 1}` },
      el("div", { class: "pair", title: "click to zoom" },
        el("figure", {}, img(g.source, `source normal scan for ${c} sample ${i + 1}`), el("figcaption", {}, "before: normal scan")),
        el("figure", {}, img(g.src, `${c} generated sample ${i + 1}: ${name}`), el("figcaption", {}, `after: generated ${c}`))),
      el("div", { class: "meta" }, el("span", {}, `#${i + 1} `, el("b", { class: "state" }, "unrated")), el("span", { class: "file", title: name }, name)),
      el("div", { class: "btns" }, ...ANSWERS.map(([v, text, cls]) =>
        el("button", { class: cls, "data-v": v, "aria-pressed": "false", onclick: () => answer(g, tile, v) }, text))),
      el("div", { class: "why", hidden: "" }, "Why flawed? (pick any) ",
        el("div", { class: "opts" }, ...REASONS.map(([k, text]) =>
          el("button", { "data-v": k, "aria-pressed": "false", onclick: () => toggleReason(g, tile, k) }, text)))));
    tile.querySelector(".pair").onclick = () => tile.classList.toggle("zoom");
    tile.onkeydown = e => { if (e.target !== tile) return; const k = e.key.toLowerCase();
      if (k === "y") answer(g, tile, "real"); else if (k === "n") answer(g, tile, "flawed"); else if (k === "u") answer(g, tile, "unsure");
      else if (k === "backspace" || k === "delete") answer(g, tile, null); };
    paint(g, tile); grid.append(tile);
  });
  const strip = el("div", { class: "strip" }, ...D.real[c].map((src, i) => src ? el("img", { src, alt: `real ${c} training scan ${i + 1}`, loading: "lazy",
    title: "click to enlarge", onclick: e => e.target.classList.toggle("big") }) : ""));
  const refs = el("div", { class: "refs" },
    el("div", { class: "small" }, el("b", {}, `Real ${c} scans (${D.real[c].length}, training set): what the CycleGAN learned to imitate. `),
      el("button", { onclick: e => { strip.hidden = !strip.hidden; e.target.textContent = strip.hidden ? "Show" : "Hide"; } }, "Hide")),
    strip);
  addTab(c, tabLabel(c), el("div", {},
    el("div", { class: "note" }, el("b", {}, `Question: does the “after” image look like the real ${c} scans in the strip, without obvious flaws?`),
      el("ol", {},
        el("li", {}, `First scroll through the real ${c} scans in the strip to see what they have in common.`),
        el("li", {}, "For each pair, compare before and after: what did the CycleGAN change, and does the change resemble the real scans?"),
        el("li", {}, "Choose ✓ Looks real, ✗ Flawed (then pick why), or ? Unsure. Unsure is a valid answer: do not guess.")),
      el("span", { class: "small muted" }, "Ignore the framing: generated images usually keep the outline and black corners of the normal scan they came from, while the real scans have black bands. Judge the retina itself. ",
        "Keys on a focused tile: Y looks real, N flawed, U unsure, Delete clear. Click a pair to zoom.")),
    el("div", { class: "note small" }, "These images already passed the automatic QC, so the share that looks real here is conditional on QC and is not comparable to the paper's raw acceptance rate."),
    D.jpeg ? el("div", { class: "note warn small" }, `Shown as JPEG copies (quality ${D.jpeg}) of the PNGs to fit a size cap. `,
      "Fine artefacts may differ: for the answers of record, use the sheet built without --jpeg.") : "",
    refs, grid));
}

// ---------- export ----------
const csvCell = s => /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
function csv() {
  const lines = ["rater,task,disease,item,file,source_file,answer,reasons,answered_at,seed"];
  const row = (...xs) => lines.push([rater.value, ...xs, D.seed].map(x => csvCell(String(x))).join(","));
  for (const v of ["gan", "none"]) for (const r of D.cams) { const s = S.cam[camKey(v, r)] || {};
    row(`cam_${v}`, r.label, r.stem, r.path, "", s.answer || "", "", s.at || ""); }
  for (const c of RARE) D.gan[c].forEach((g, i) => { const s = S.gan[g.file] || {};
    row("gan", c, i + 1, g.file, g.sourceFile, s.answer || "", (s.reasons || []).join(";"), s.at || ""); });
  return lines.join("\n") + "\n";
}
const ganBody = el("tbody"), camBody = el("tbody"), text = el("textarea", { readonly: "", "aria-label": "answers CSV" });
const count = (xs, a) => xs.filter(x => x === a).length;
function renderExport() {
  ganBody.replaceChildren(...RARE.map(c => { const ss = D.gan[c].map(g => S.gan[g.file]).filter(Boolean), as = ss.map(s => s.answer);
    const real = count(as, "real"), flawed = count(as, "flawed"), why = {};
    for (const s of ss) for (const k of s.reasons || []) why[k] = (why[k] || 0) + 1;
    const top = Object.entries(why).sort((a, b) => b[1] - a[1])[0];
    return el("tr", {}, el("td", {}, c), el("td", {}, `${as.length}/${D.gan[c].length}`), el("td", {}, real), el("td", {}, flawed),
      el("td", {}, count(as, "unsure")), el("td", {}, real + flawed ? `${(100 * real / (real + flawed)).toFixed(0)}%` : "–"),
      el("td", {}, top ? `${REASONS.find(r => r[0] === top[0])[1]} (${top[1]})` : "–")); }));
  camBody.replaceChildren(...["gan", "none"].map(v => { const as = D.cams.map(r => S.cam[camKey(v, r)]?.answer).filter(Boolean);
    return el("tr", {}, el("td", {}, v), el("td", {}, `${as.length}/${D.cams.length}`), ...WHERE.map(([k]) => el("td", {}, count(as, k)))); }));
  text.value = csv();
}
const table = (heads, body) => el("table", {}, el("thead", {}, el("tr", {}, ...heads.map(h => el("th", {}, h)))), body);
addTab("export", "Export CSV", el("div", {},
  el("div", { class: "note small" }, "Report these as a visual check by the author (not a clinician), with the seed and sample size. They are not the paper's Fig. 6 acceptance rates."),
  el("h2", {}, "2. Generated images"),
  table(["disease", "answered", "looks real", "flawed", "unsure", "looks real, of decided", "most common flaw"], ganBody),
  el("h2", {}, "1. CAM: where is the strongest colour?"),
  table(["variant", "answered", ...WHERE.map(w => w[1])], camBody),
  el("p", {}, el("button", { onclick: () => { const a = el("a", { href: URL.createObjectURL(new Blob([csv()], { type: "text/csv" })),
      download: `review_answers_seed${D.seed}_${rater.value || "rater"}.csv` }); document.body.append(a); a.click(); a.remove(); } }, "Download CSV"),
    el("span", { class: "small muted" }, " If the download is blocked here, copy the text below.")),
  text,
  el("p", { class: "small muted" }, "One row per question (task cam_gan, cam_none or gan); unanswered questions are exported with an empty answer, so the file always lists the full sample. Answers are kept in this browser (localStorage) until exported.")));
show("cam");
</script></body></html>
"""

if __name__ == "__main__":
    main()
