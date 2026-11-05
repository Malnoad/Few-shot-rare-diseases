"""Build the manuscript with an existing TeX engine; never run study experiments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def run(command: list[str], label: str) -> None:
    output = HERE / "verification" / (label + ".log")
    output.parent.mkdir(exist_ok=True)
    pid_path = ROOT / ".agents/pids" / (label + ".pid")
    pid_path.parent.mkdir(exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(command, cwd=HERE, stdout=handle,
                                   stderr=subprocess.STDOUT)
        pid_path.write_text(str(process.pid), encoding="ascii")
        code = process.wait()
    if code:
        raise SystemExit(f"{label} failed ({code}); see {output}")
    print(f"{label}: PASS")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default=shutil.which("tectonic") or shutil.which("pdflatex"))
    args = parser.parse_args()
    if not args.engine:
        parser.error("Supply --engine PATH_TO_TECTONIC_OR_PDFLATEX; no installer is run.")
    run([sys.executable, str(HERE / "make_figure.py")], "fm_figure")
    engine = str(Path(args.engine).resolve())
    version = subprocess.check_output([engine, "--version"], text=True, encoding="utf-8")
    if "tectonic" in Path(engine).name.lower():
        run([engine, "--keep-logs", "--keep-intermediates", "main.tex"], "fm_compile")
    elif "pdflatex" in Path(engine).name.lower():
        bibtex = shutil.which("bibtex") or str(Path(engine).with_name("bibtex" + Path(engine).suffix))
        command = [engine, "-interaction=nonstopmode", "-halt-on-error", "main.tex"]
        run(command, "fm_compile_pass1")
        run([bibtex, "main"], "fm_bibtex")
        run(command, "fm_compile_pass2")
        run(command, "fm_compile_pass3")
    else:
        parser.error("Supported engines are tectonic and pdflatex.")
    (HERE / "verification/build_environment.json").write_text(json.dumps({
        "engine": engine, "version": version.strip(), "python": sys.version,
        "format": "Unmodified IEEEtran conference class, 10 pt, US Letter, two columns",
        "note": "Only document/figure construction; no feature extraction, model fitting or new inference."
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
