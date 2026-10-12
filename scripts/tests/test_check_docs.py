"""Host-only regression fixtures for documentation check failure handling."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


SCRIPTS = Path(__file__).resolve().parents[1]


def note_id(suffix):
    # Keep deliberately invalid fixture IDs out of the repository reference scan.
    return "CN" + "-" + suffix


class NoteChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scripts").mkdir()
        (self.root / "docs").mkdir()
        shutil.copy2(SCRIPTS / "check_note_refs.py", self.root / "scripts")

    def run_check(self, notes=None, refs=""):
        if notes is not None:
            (self.root / "docs/codebase_notes.md").write_text(notes)
        (self.root / "sample.cpp").write_text(refs)
        return subprocess.run(
            [sys.executable, str(self.root / "scripts/check_note_refs.py")],
            capture_output=True, text=True,
        )

    def test_missing_notes_fails(self):
        result = self.run_check()
        self.assertEqual(result.returncode, 1)
        self.assertIn("missing required", result.stdout)

    def test_numeric_symbolic_ids_and_placeholders(self):
        numeric, symbolic = note_id("HF-1"), note_id("NVRTC-FUSE")
        notes = f"## {numeric} — note\n## {symbolic} — note\nSee {numeric}.\n"
        refs = " ".join([numeric, symbolic, note_id("<AREA>-<n>"),
                         note_id("FUSE-*"), note_id("*")])
        result = self.run_check(notes, refs)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_malformed_ids_do_not_match_valid_prefix(self):
        valid = note_id("HF-1")
        notes = f"## {valid} — note\n"
        for suffix in ("HF-1oops", "hf-1", "HF", "HF-1_extra", "HF-1-2"):
            with self.subTest(suffix=suffix):
                result = self.run_check(notes, note_id(suffix))
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn("malformed note ID", result.stdout)

    def test_dangling_symbolic_reference_in_notes(self):
        result = self.run_check(f"## {note_id('HF-1')} — note\n"
                                f"See {note_id('NVRTC-FUSE')}.\n")
        self.assertEqual(result.returncode, 1)
        self.assertIn("not defined", result.stdout)

    def test_duplicate_and_malformed_headings(self):
        for ids in ((note_id("HF-1"), note_id("HF-1")), (note_id("hf-1"),)):
            with self.subTest(ids=ids):
                result = self.run_check("\n".join(f"## {nid} — note" for nid in ids))
                self.assertEqual(result.returncode, 1, result.stdout)

    def test_untracked_module_cmake_references(self):
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        directory = self.root / "modules/example"
        directory.mkdir(parents=True)
        for name in ("CMakeLists.txt", "helpers.cmake"):
            with self.subTest(name=name):
                path = directory / name
                path.write_text(f"# See {note_id('NVRTC-MISSING')}\n")
                result = self.run_check(f"## {note_id('HF-1')} — note\n")
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn(f"modules/example/{name}", result.stdout)
                path.unlink()


class CheckRunner(unittest.TestCase):
    def test_runner_runs_all_checks_and_propagates_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            scripts = Path(directory) / "scripts"
            scripts.mkdir()
            shutil.copy2(SCRIPTS / "check_docs.py", scripts)
            names = ("check_doc_toml.py", "check_doc_cpp.py",
                     "check_note_refs.py", "check_stage_integration.py")
            for index, name in enumerate(names):
                (scripts / name).write_text(
                    "import sys\n"
                    f"print({name!r}, sys.argv[1:])\n"
                    f"sys.exit({int(index == 1)})\n"
                )
            result = subprocess.run(
                [sys.executable, str(scripts / "check_docs.py"), "--verbose"],
                cwd="/", capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 1)
            for name in names:
                self.assertIn(name, result.stdout)
            self.assertIn("check_stage_integration.py []", result.stdout)
            self.assertIn("check_note_refs.py ['--verbose']", result.stdout)
            self.assertIn("FAIL: C++ examples", result.stdout)


if __name__ == "__main__":
    unittest.main()
