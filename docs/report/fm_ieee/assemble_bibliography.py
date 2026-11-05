"""Assemble verified batches, preserving names and documenting version choices."""
from __future__ import annotations
import json
from pathlib import Path
import re
import unicodedata

HERE = Path(__file__).resolve().parent


def tex_unicode(text: str) -> str:
    marks = {"\u0301": "'", "\u0300": "`", "\u0308": '"',
             "\u030c": "v", "\u0302": "^", "\u0303": "~", "\u0327": "c"}
    output = []
    for char in text:
        if ord(char) < 128:
            output.append(char)
        elif char in ["–", "—"]:
            output.append("--" if char == "–" else "---")
        elif char == "\u00a0":
            output.append("~")
        else:
            parts = unicodedata.normalize("NFD", char)
            if len(parts) == 2 and parts[1] in marks:
                output.append("{\\" + marks[parts[1]] + parts[0] + "}")
            else:
                raise ValueError(f"Explicit TeX encoding needed for {char!r}")
    return "".join(output)


def main() -> None:
    parent = (HERE / "verification/parent_refs/parent_refs.bib").read_text(encoding="utf-8")
    retinal = (HERE / "verification/retinal_refs/retinal_refs.bib").read_text(encoding="utf-8")
    raw = json.loads((HERE / "verification/retinal_refs/raw/A10_journal_crossref.json").read_text(encoding="utf-8"))["message"]
    assert raw["DOI"] == "10.1016/j.compeleceng.2019.106532"
    authors = " and ".join(a["family"] + ", " + a["given"] for a in raw["author"])
    octid = ("@article{OCTID2020,\n  author = {" + authors + "},\n"
             "  title = {" + raw["title"][0] + "},\n"
             "  journal = {Computers \\& Electrical Engineering},\n"
             "  volume = {81},\n  year = {2020},\n  note = {Art. no. 106532},\n"
             "  doi = {10.1016/j.compeleceng.2019.106532},\n"
             "  url = {https://doi.org/10.1016/j.compeleceng.2019.106532}\n}\n")
    retinal, replaced = re.subn(r"@misc\{OCTID2018,.*?\n\}\s*", lambda _: octid + "\n", retinal, flags=re.S)
    assert replaced == 1
    control = """@IEEEtranBSTCTL{IEEEcontrol,
  CTLuse_forced_etal = {yes},
  CTLmax_names_forced_etal = {6},
  CTLnames_show_etal = {1}
}
"""
    combined = tex_unicode(control + "\n" + parent + "\n" + retinal)
    combined = re.sub(r"(?m)^(\s*title\s*=\s*)\{(?!\{)(.*)\}(,?)$",
                      r"\1{{\2}}\3", combined)
    combined = combined.replace(" pages = {},\n", "")
    (HERE / "references.bib").write_text(combined, encoding="ascii")
    print("Assembled 21 references, full author lists retained; IEEE display abbreviates lists longer than 6.")


if __name__ == "__main__":
    main()
