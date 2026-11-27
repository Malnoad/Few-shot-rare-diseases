"""The two Colab notebooks: built from fm_notebook.py, and every command they issue is valid.

Cells run here against a temporary folder with Colab, Drive, the network and the GPU stubbed.
Each fm_*.py command a cell issues is parsed by that script's real argument parser, so a wrong
flag fails on the laptop instead of in the middle of a Colab session. No model runs, no score.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shlex
import sys
import tempfile
import types
import unittest
from unittest import mock
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import fm_evaluate  # noqa: E402
import fm_external  # noqa: E402
import fm_extract  # noqa: E402
import fm_notebook  # noqa: E402
import fm_prepare as prep  # noqa: E402

MODULES = {"fm_prepare.py": prep, "fm_extract.py": fm_extract, "fm_evaluate.py": fm_evaluate, "fm_external.py": fm_external}
ARMS = ("RETFound", "DINOv2", "MIRAGE", "Inception")


class Parsed(Exception):
    pass


def parse(argv):
    """Run the script's own parser on a notebook command; return the parsed arguments."""
    script = Path(argv[1]).name
    original = argparse.ArgumentParser.parse_args

    def stop(self, *args, **kwargs):
        raise Parsed(original(self, *args, **kwargs))

    with mock.patch.object(sys, "argv", [script, *argv[2:]]), mock.patch.object(argparse.ArgumentParser, "parse_args", stop):
        try:
            MODULES[script].main()
        except Parsed as done:
            return done.args[0]
        except SystemExit as error:
            raise AssertionError(f"the notebook issues a command its script rejects: {argv[1:]}") from error
    raise AssertionError(f"parser did not run: {argv}")


def cells(name):
    nb = json.loads((ROOT / name).read_text(encoding="utf-8"))
    return {"".join(c["source"]).split("\n")[0].split(".")[0]: "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"}


def write(path, data="x"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())


def archive(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zipped:  # fixed date: like the real exports, the same input gives the same bytes
        zipped.writestr(zipfile.ZipInfo("complete.json", (2026, 1, 1, 0, 0, 0)), json.dumps({"files": {"a": path.name}}))
    write(str(path) + ".json", json.dumps({"sha256": hashlib.sha256(path.read_bytes()).hexdigest()}))


class Colab:
    """Stand-ins for the shell, the worker processes and Drive; records every script command."""

    def __init__(self, tmp):
        self.tmp, self.commands, self.fail, self.mounts, self.interrupt_freeze = tmp, [], set(), 0, False
        colab = types.ModuleType("google.colab")
        colab.drive = types.SimpleNamespace(mount=self.mount, flush_and_unmount=lambda: None)
        colab.userdata = types.SimpleNamespace(get=lambda key: "stub")
        self.modules = {"google": types.ModuleType("google"), "google.colab": colab}

    def mount(self, *args, **kwargs):
        self.mounts += 1

    def text(self, cell):
        # Colab paths become a temporary folder; the checkout is the real one, so its scripts are the ones parsed.
        return cell.replace('pathlib.Path(ROOT, "data/external")', f'pathlib.Path({(self.tmp / "external_images").as_posix()!r})') \
            .replace("/content/Few-shot", ROOT.as_posix()).replace("/content", self.tmp.as_posix())

    def sh(self, command):
        argv = shlex.split(command)
        if len(argv) < 2 or Path(argv[1]).name not in MODULES:
            return
        args, script = parse(argv), Path(argv[1]).name
        self.commands.append(argv)
        step = args.step
        if script == "fm_external.py" and step == "candidates":
            rows = [["OCTID", "a", "NORMAL"], ["OCTID", "b", "CSC"], ["OCTDL", "c", "MH"]]
            with open(args.out, "w", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerows([["source", "path", "label"], *rows])
        elif script == "fm_prepare.py" and step == "lock":
            gate = json.loads(Path(args.gate).read_text(encoding="utf-8"))
            self.gate = gate
            lock = dict(created_utc="t", evaluation_environment=prep.math_environment(), endpoints=dict(reporting_k=[1, 2, 4, 8, 16]),
                        protocol_sha256=prep.PROTOCOL_SHA256, code=prep.code_identity(),
                        arms={row["name"]: dict(status=row["status"]) for row in gate["arms"]})
            write(Path(args.out) / "input_lock.json", json.dumps(lock))
            archive(args.archive)
        elif script == "fm_prepare.py" and step == "restore":
            write(Path(args.out) / "input_lock.json", json.dumps(self.lock))
        elif script == "fm_prepare.py" and step == "export":
            archive(args.archive)
        elif script == "fm_external.py" and step == "screen":
            write(Path(args.out) / "external_screen_summary.json", json.dumps(dict(pairs_by_block={})))
        elif script == "fm_external.py" and step == "freeze":
            source = dict(candidates=2, queries=2, excluded=0, dependence_groups=2, labels={"NORMAL": 2})
            write(Path(args.out) / "external_lock.json", json.dumps(dict(sources=dict(OCTID=source, OCTDL=source))))
            if not self.interrupt_freeze:
                write(Path(args.out) / "complete.json", "{}")
        elif step == "nominate":
            write(Path(args.out) / "nomination.json", json.dumps(dict(nominee="DINOv2")))
            write(Path(args.out) / "complete.json", "{}")
            write(Path(args.lock) / "nomination_binding.json", "{}")
        else:  # develop, report, summarize, geometry
            write(Path(args.out) / "complete.json", "{}")

    def worker(self, command, log):
        args = parse(command)
        self.commands.append(command)
        arm = Path(args.spec).stem
        if (args.step, arm) in self.fail:
            return 1, 1
        out = Path(args.out)
        write(out / "complete.json", "{}")
        if args.step == "pilot":
            repeat = dict(worst_image_difference_ratio=1e-6, min_cosine=0.99999, max_difference_ratio=1e-6)
            write(out / "timing.json", json.dumps(dict(median_seconds_per_image=0.0123, numerical=dict(real=repeat, synthetic=repeat),
                                                       costs=dict(model_loading_seconds=4.5, peak_vram_bytes=2e9))))
            write(out / "load_report.json", json.dumps(dict(missing_keys=[], unexpected_keys=[])))
        return 1, 0


class Notebooks(unittest.TestCase):
    def test_committed_notebooks_are_the_builders_output_and_every_cell_compiles(self):
        built = fm_notebook.build(fm_notebook.pinned_commit())
        for name, text in built.items():
            self.assertEqual((ROOT / name).read_text(encoding="utf-8").replace("\r\n", "\n"), text, f"rebuild {name}: python fm_notebook.py")
            for title, source in cells(name).items():
                compile(source, f"{name}: {title}", "exec")
            self.assertNotIn("@COMMIT@", text)

    def test_first_cell_records_lams_decisions_verbatim(self):
        g = {}
        exec(cells("colab_fm.ipynb")["# Fill in this cell before Run all (Step 1 stops until it is complete)"], g)
        recorded = (ROOT / "docs/research/FM_TERMS_EVIDENCE.md").read_text(encoding="utf-8")
        self.assertTrue(all(value in recorded for value in g["DECISIONS"].values()))
        self.assertIn('"Approve"', g["EXTERNAL_DECISION"])
        self.assertEqual((g["RUN_BY"], g["RETFOUND_ACCESS"], g["UNAVAILABLE"]), ("", "", {}))

    def test_extraction_session_from_external_download_to_drive(self):
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            colab, nb = Colab(tmp), cells("colab_fm.ipynb")
            with mock.patch.dict(sys.modules, colab.modules):
                g = {}
                exec(colab.text(nb["# Settings and helpers"]), g)
                exec(nb["# Fill in this cell before Run all (Step 1 stops until it is complete)"], g)
                first = colab.text(nb["# Step 1"]).split("query =")[0]
                with self.assertRaisesRegex(AssertionError, "on purpose"):
                    exec(first, g)
                g.update(RUN_BY="runner", RETFOUND_ACCESS="account x, approved 2026-11-11", UNAVAILABLE={"Inception": "no"})
                with self.assertRaisesRegex(AssertionError, "UNAVAILABLE may only name"):
                    exec(first, g)
                g["UNAVAILABLE"] = {}
                exec(first, g)
                self.assertEqual(g["RUN_ARMS"], list(ARMS))
                py = Path(sys.executable).as_posix()
                g.update(PY=py, sh=colab.sh, record=lambda row: None, SPECS=f"{g['WORK']}/specs",
                         gpu=types.SimpleNamespace(stdout="NVIDIA L4, 8.9, 23034 MiB, 550.54\n"))
                write(Path(g["REPORTS"]) / "step6_inventory.txt")

                # Step 8: external download with the network stubbed; the gate entries must satisfy the lock.
                images = {"1": b"normal", "2": b"mh", "3": b"csr"}
                listing = {"doi:10.5683/SP/WLW4ZT": ("N1.jpeg", "1"), "doi:10.5683/SP/MBMQGD": ("M1.jpeg", "2"), "doi:10.5683/SP/FVLZIL": ("C1.jpeg", "3")}
                buffer = io.BytesIO()
                with zipfile.ZipFile(buffer, "w") as zipped:
                    for name in ("OCTDL/NO/no_1.jpg", "OCTDL/VID/vid_1.jpg", "OCTDL/AMD/amd_1.jpg", "OCTDL/README.txt"):
                        zipped.writestr(name, b"i")
                payloads = {"OCTDL.zip": buffer.getvalue(), "OCTDL_labels.csv": b"file_name,disease\n"}
                md5 = lambda data: hashlib.md5(data).hexdigest()  # noqa: E731

                class Response:
                    def __init__(self, data=None, payload=None):
                        self.data, self.payload = data, payload
                    raise_for_status = lambda self: None  # noqa: E731
                    json = lambda self: self.payload  # noqa: E731
                    iter_content = lambda self, size: iter([self.data])  # noqa: E731
                    __enter__ = lambda self: self  # noqa: E731
                    __exit__ = lambda self, *error: False  # noqa: E731

                def get(url, params=None, stream=False, timeout=None):
                    if "datasets" in url:
                        name, file_id = listing[params["persistentId"]]
                        files = [dict(dataFile=dict(filename=name, filesize=1, md5=md5(images[file_id]), id=file_id)),
                                 dict(dataFile=dict(filename="note.txt", filesize=1, md5="0" * 32, id="9"))]
                        return Response(payload=dict(data=dict(latestVersion=dict(
                            metadataBlocks=dict(citation=dict(fields=[dict(typeName="title", value="t")])), files=files))))
                    if "datafile" in url:
                        return Response(data=images[url.rsplit("/", 1)[1]])
                    return Response(data=payloads[url.rsplit("/", 1)[1].split("?")[0]])

                rows = sorted([[doi, "t", name, 1, md5(images[i]), i] for doi, (name, i) in listing.items()]
                              + [[doi, "t", "note.txt", 1, "0" * 32, "9"] for doi in listing], key=lambda r: (r[0], r[2]))
                out = io.StringIO(newline="")
                csv.writer(out, lineterminator="\n").writerows([["doi", "title", "filename", "bytes", "md5", "file_id"], *rows])
                step8 = colab.text(nb["# Step 8"]) \
                    .replace("984fde2b5222c44a3c2ed7bf34b5d513a921b7d88d3a12e4518cbb28f52b6f21", hashlib.sha256(out.getvalue().encode()).hexdigest()) \
                    .replace('{"NORMAL": 206, "MH": 102, "CSR": 102}', '{"NORMAL": 1, "MH": 1, "CSR": 1}') \
                    .replace('{"NO": 332, "VID": 76}', '{"NO": 1, "VID": 1}') \
                    .replace('"dec9766fc916202c45a333c4f5a09576"', repr(md5(payloads["OCTDL.zip"]))) \
                    .replace('"2fced789038c6ca83283066c2f8ad289"', repr(md5(payloads["OCTDL_labels.csv"]))) \
                    .replace('{"OCTDL/MH": 39, "OCTDL/NORMAL": 332, "OCTID/CSC": 102, "OCTID/MH": 102, "OCTID/NORMAL": 206}',
                             '{"OCTDL/MH": 1, "OCTID/CSC": 1, "OCTID/NORMAL": 1}')
                self.assertNotEqual(step8, colab.text(nb["# Step 8"]))
                g["requests"] = types.SimpleNamespace(get=get)
                exec(step8, g)
                external = tmp / "external_images"
                self.assertEqual(sorted(p.relative_to(external).as_posix() for p in external.rglob("*") if p.is_file()),
                                 ["OCTDL/NO/no_1.jpg", "OCTDL/VID/vid_1.jpg", "OCTID/CSR/C1.jpeg", "OCTID/MH/M1.jpeg", "OCTID/NORMAL/N1.jpeg"])
                for source in g["EXTERNAL_GATE"].values():  # the fields fm_prepare.lock_inputs requires of a planned source
                    self.assertEqual(source["status"], "planned")
                    self.assertTrue(all(source[key].strip() for key in ("release", "release_sha256", "terms_evidence", "eligibility", "label_inventory")))

                # Steps 9-11: one failed pilot sequence may be retried once (A3 S-3); a third is refused.
                g["os"].environ["HF_TOKEN_TEST"] = "secret"
                step9 = colab.text(nb["# Step 9"]).replace(
                    '("/root/.cache/huggingface/token", "/root/.cache/huggingface/stored_tokens", "/root/.huggingface/token")', "()")
                exec(step9, g)
                g["os"].environ.pop("HF_TOKEN_TEST")
                self.assertNotIn("HF_TOKEN_TEST", g["SAFE_ENV"])
                g.update(worker=colab.worker, gpu_users=lambda: "")
                colab.fail = {("pilot", "MIRAGE")}
                exec(colab.text(nb["# Step 10"]), g)
                with self.assertRaisesRegex(RuntimeError, "one retry"):
                    exec(colab.text(nb["# Step 11"]), g)
                exec(colab.text(nb["# Step 10"]), g)
                with self.assertRaisesRegex(RuntimeError, "failed twice"):
                    exec(colab.text(nb["# Step 11"]), g)
                with self.assertRaisesRegex(AssertionError, "Two pilot sequences have failed"):
                    exec(colab.text(nb["# Step 10"]), g)
                self.assertEqual(len(g["ATTEMPTS"]), 2)
                # Lam lists the arm as unavailable; the completed pilots of the other arms stand.
                g["UNAVAILABLE"] = {"MIRAGE": "repeat check failed twice, values in the log"}
                exec(first, g)
                before = len(colab.commands)
                exec(colab.text(nb["# Step 10"]), g)
                self.assertEqual(len(colab.commands), before)
                exec(colab.text(nb["# Step 11"]), g)
                pilots = [c for c in colab.commands if "pilot" in c]
                self.assertEqual([Path(c[c.index("--spec") + 1]).stem for c in pilots[-4:]], list(ARMS))
                self.assertEqual([c[0] for c in pilots[-4:]], [py, py, py, sys.executable])
                self.assertEqual(len({c[c.index("--timing-session") + 1] for c in pilots[-4:]}), 1)

                # Steps 12-16: lock, save, external screen, extraction, save.
                exec(colab.text(nb["# Step 12"]), g)
                self.assertEqual({row["name"]: row["status"] for row in colab.gate["arms"]},
                                 dict(RETFound="available", DINOv2="available", MIRAGE="unavailable", Inception="available"))
                self.assertEqual(set(colab.gate["external_sources"]), {"OCTID", "OCTDL"})
                self.assertTrue(all(row.get("pilot") or row.get("reason") for row in colab.gate["arms"]))
                drive = Path(g["OUT"]) / "final" / g["ATTEMPT"]
                write(drive / "input-lock.zip.part", "cut off")  # an interrupted earlier copy must not block the retry
                exec(colab.text(nb["# Step 13"]), g)
                self.assertFalse(list(drive.glob("*.part")))
                self.assertEqual(hashlib.sha256((drive / "input-lock.zip").read_bytes()).hexdigest(), g["LOCK_SHA256"])
                deposit = (drive / f"lock-hashes-{g['ATTEMPT']}.txt").read_text(encoding="utf-8")
                for value in (g["LOCK_SHA256"], prep.PROTOCOL_SHA256, prep.code_identity()["docs/research/FM_AMENDMENTS.md"], g["COMMIT"]):
                    self.assertIn(value, deposit)
                colab.interrupt_freeze = True  # the query lock is written but the freeze did not finish
                with self.assertRaisesRegex(AssertionError, "did not finish"):
                    exec(colab.text(nb["# Step 14"]), g)
                colab.interrupt_freeze = False
                exec(colab.text(nb["# Step 14"]), g)
                steps = [c[2] for c in colab.commands if Path(c[1]).name == "fm_external.py"]
                self.assertEqual(steps, ["candidates", "screen", "freeze", "freeze"], "a rerun finishes the freeze without a second screen")
                exec(colab.text(nb["# Step 15"]), g)
                extracts = [c for c in colab.commands if "extract" in c]
                self.assertEqual(len(extracts), 6)  # three available arms, study and external images
                self.assertEqual(sum("--external" in c for c in extracts), 3)
                self.assertTrue(all(c[c.index("--prepared") + 1] == g["PREPARED"] for c in extracts))
                exec(colab.text(nb["# Step 16"]), g)
                receipt = json.loads((drive / "receipt.json").read_text(encoding="utf-8"))
                self.assertTrue(receipt["complete"])
                self.assertEqual(receipt["lock_sha256"], g["LOCK_SHA256"])
                self.assertEqual(len(receipt["files"]), 2 + 6 + 1)
                for name, entry in receipt["files"].items():
                    self.assertEqual(hashlib.sha256((drive / name).read_bytes()).hexdigest(), entry["sha256"])
                with zipfile.ZipFile(next(drive.glob("evidence-*.zip"))) as zipped:
                    names = zipped.namelist()
                self.assertTrue(any(name.startswith("pilots/") for name in names) and "model-gate.json" in names)
                self.assertFalse(any(name.startswith(("features/", "external/")) for name in names))
                exec(colab.text(nb["# Step 16"]), g)  # a rerun saves nothing new and overwrites only the receipt

                # The scoring notebook reads exactly what the extraction notebook saved.
                self.evaluate(colab, tmp, g["ATTEMPT"], g["LOCK_SHA256"], g["COMMIT"])

    def evaluate(self, colab, tmp, attempt, lock_sha256, commit):
        nb = cells("colab_fm_evaluate.ipynb")
        colab.lock = json.loads((tmp / "prepared/input_lock.json").read_text(encoding="utf-8"))
        g = {}
        exec(colab.text(nb["# Settings and helpers"]), g)
        self.assertEqual(g["COMMIT"], commit)
        exec(nb["# Fill in this cell before Run all (Step 1 stops until it is complete)"], g)
        with self.assertRaisesRegex(AssertionError, "on purpose"):
            exec(colab.text(nb["# Step 1"]), g)
        g.update(ATTEMPT=attempt, LOCK_SHA256="0" * 64, LOCK_APPROVAL="Lam, date, words", TIMESTAMP_RECORD="OSF, date", RUN_BY="runner")
        exec(colab.text(nb["# Step 1"]), g)
        g.update(PY=Path(sys.executable).as_posix(), sh=colab.sh)
        with self.assertRaisesRegex(AssertionError, "not the lock of this attempt"):
            exec(colab.text(nb["# Step 3"]), g)
        g["LOCK_SHA256"] = lock_sha256
        exec(colab.text(nb["# Step 3"]), g)
        exec(colab.text(nb["# Step 4"]), g)
        self.assertEqual(g["AVAILABLE"], ["RETFound", "DINOv2", "Inception"])
        exec(colab.text(nb["# Step 5"]), g)
        before = len(colab.commands)
        with self.assertRaisesRegex(AssertionError, "Stopped on purpose"):
            exec(colab.text(nb["# Step 6"]), g)
        saved = Path(g["OUT"]) / "final" / attempt / "evaluation"
        self.assertTrue((saved / "nomination_binding.json").exists(), "the decision reaches Drive before the stop")
        self.assertFalse(any("report" in c or "summarize" in c for c in colab.commands[before:]))
        g["REPORT_AFTER_NOMINATION"] = True
        for step in ("# Step 6", "# Step 7", "# Step 8", "# Step 9"):
            exec(colab.text(nb[step]), g)
        steps = [(Path(c[1]).name, c[2]) for c in colab.commands[before:] if c[2] not in ("export",)]
        self.assertEqual(steps[0], ("fm_evaluate.py", "nominate"))
        self.assertEqual(steps.count(("fm_evaluate.py", "report")), 3)
        self.assertEqual(steps.count(("fm_external.py", "report")), 3)
        self.assertEqual(steps.count(("fm_evaluate.py", "geometry")), 3)
        order = [s for s in steps if s[1] in ("nominate", "report", "summarize")]
        self.assertLess(order.index(("fm_evaluate.py", "nominate")), order.index(("fm_evaluate.py", "report")))
        self.assertGreater(order.index(("fm_external.py", "summarize")), max(i for i, s in enumerate(order) if s[1] == "report"))
        results = [p for p in saved.iterdir() if p.name.startswith("results-") and p.suffix == ".zip"]
        self.assertEqual(len(results), 1)
        with zipfile.ZipFile(results[0]) as zipped:
            self.assertIn("approvals.json", zipped.namelist())
        # A fresh runtime restores the original nomination from Drive instead of making a new one.
        g2 = {}
        exec(colab.text(nb["# Settings and helpers"]).replace("/fm_eval", "/fm_eval_second"), g2)
        g2.update(ATTEMPT=attempt, LOCK_SHA256=lock_sha256, LOCK_APPROVAL="a", TIMESTAMP_RECORD="t", RUN_BY="r",
                  REPORT_AFTER_NOMINATION=False, OMIT_LOGISTIC_REASON="", PY=Path(sys.executable).as_posix(), sh=colab.sh)
        exec(colab.text(nb["# Step 3"]), g2)
        exec(colab.text(nb["# Step 4"]), g2)
        self.assertTrue((Path(g2["LOCK"]) / "nomination_binding.json").exists())


if __name__ == "__main__":
    unittest.main()
