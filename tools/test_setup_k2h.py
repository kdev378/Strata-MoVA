"""Tests for setup.py's K2 Horizon install (PREVIEW_ONLY_FAMILIES, --family k2h): reachable only when named (not in the menu,
the model lists, strata launch or the app), the pinned and hashed file, and install_k2h's steps (the CPU engine's build, the
download, the tokenizer folder, the config).  Mocked build, downloads and processes - nothing is built, fetched or started.

    python -m unittest tools.test_setup_k2h
"""
from __future__ import annotations

import argparse
import contextlib
import io
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import setup  # noqa: E402

REPO = "IFM/K2-Horizon-MoVA-36B-A4B-GGUF"


def args(**kw):
    base = dict(family="k2h", model=None, yes=True, models_dir=None, gguf_dir=None, context=None, host=None, api_key=None,
                browser=None, data_dir=None, port=None, skip_build=False)
    base.update(kw)
    return argparse.Namespace(**base)


def run(fn, *a, **kw):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            return fn(*a, **kw), out.getvalue()
        except SystemExit as e:
            return e.code, out.getvalue()


class Entries(unittest.TestCase):
    def test_pinned_and_hashed(self):
        fam = setup.PREVIEW_ONLY_FAMILIES["k2h"]
        self.assertRegex(setup.hf(REPO), r"/resolve/[0-9a-f]{40}/$")
        for name, (size, sha) in fam["sha256"].items():
            self.assertTrue(name.startswith("K2-Horizon-MoVA-36B-A4B-") and size > 1e9 and re.fullmatch(r"[0-9a-f]{64}", sha), name)
        for model in fam["models"]:
            self.assertIn(fam["file"].format(q=model), fam["sha256"])
        self.assertEqual(fam["arch"], "k2-horizon")

    def test_only_when_named(self):
        """Nothing that lists or offers models reads PREVIEW_ONLY_FAMILIES: the menu, strata launch and the app see what they saw."""
        for table in (setup.FAMILIES, setup.OWN_PATH_FAMILIES, setup.PREVIEW_FAMILIES, setup.MULTI_FILE_FAMILIES):
            self.assertNotIn("k2h", table)
        for name, size in setup.MODELS.items():
            self.assertNotIn("k2h", size.get("families", ()), name)


class Install(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.calls = []
        self.cfg = None

        def build(src, bdir, target, defs, vcvars, bat):
            self.calls.append(("build", target))
            (bdir).mkdir(parents=True, exist_ok=True)
            (bdir / "strata-k2h").write_text("")

        def write_cfg(path, cfg, source=None, explicit=None):
            self.cfg = cfg
            self.calls.append(("config", path.name))

        patches = [mock.patch.object(setup, "download", lambda url, dst, what=None: self.calls.append(("get", url, dst))),
                   mock.patch.object(setup, "verify_sha256", lambda s, size, sha, own=False: self.calls.append(("sha", s.name, size, sha))),
                   mock.patch.object(setup, "free_gb", lambda p: 500.0),
                   mock.patch.object(setup, "cmake_build", build),
                   mock.patch.object(setup, "run", lambda cmd, **k: self.calls.append(("run", cmd))),
                   mock.patch.object(setup, "data_folder", lambda d, *a, **k: (self.root / "data", [])),
                   mock.patch.object(setup, "write_setup_config", write_cfg),
                   mock.patch.object(setup, "write_run_script", lambda tag, p, port, o=True: self.root / f"run-{tag}.sh"),
                   mock.patch.object(setup, "ram_gb", lambda: 48.0),
                   mock.patch.object(setup, "ROOT", self.root)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_builds_downloads_and_writes_the_config(self):
        rc, out = run(setup.install_k2h, args(models_dir=str(self.root / "models")), "k2h")
        self.assertEqual(rc, 0, out)
        self.assertIn(("build", "strata-k2h"), self.calls)
        gguf = self.root / "models" / "k2h-Q4_K_M" / "K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf"
        get = [c for c in self.calls if c[0] == "get"]
        self.assertEqual(get, [("get", setup.hf(REPO) + gguf.name, gguf)])
        size, sha = setup.K2H_FILES[gguf.name]
        self.assertIn(("sha", gguf.name, size, sha), self.calls)
        tok = [c for c in self.calls if c[0] == "run"]
        self.assertTrue(tok and tok[0][1][1].endswith("make_pack.py") and tok[0][1][2] == str(gguf))
        a = self.cfg["args"]
        self.assertEqual(a[a.index("--model") + 1], str(gguf))
        self.assertEqual(a[a.index("--ctx") + 1], "2048")
        self.assertEqual(self.cfg["exe"], str(self.root / "build-k2h" / "strata-k2h"))
        self.assertEqual(self.cfg["arch"], "k2-horizon")
        self.assertEqual(self.cfg["model_name"], "k2-horizon-mova-36b-a4b-q4_k_m")
        self.assertTrue(self.cfg["tokenizer"].endswith("k2h-q4_k_m/tokenizer"))
        self.assertIn(("config", "strata-k2h-q4_k_m.json"), self.calls)
        self.assertIn("in development", out)

    def test_gguf_dir_takes_the_persons_file(self):
        own = self.root / "mine"
        own.mkdir()
        (own / "K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf").write_bytes(b"x")
        rc, out = run(setup.install_k2h, args(gguf_dir=str(own), context=4096), "k2h")
        self.assertEqual(rc, 0, out)
        self.assertFalse([c for c in self.calls if c[0] == "get"])
        self.assertEqual(self.cfg["args"][self.cfg["args"].index("--ctx") + 1], "4096")

    def test_gguf_dir_without_the_file_says_it_is_missing(self):
        rc, out = run(setup.install_k2h, args(gguf_dir=str(self.root / "empty")), "k2h")
        self.assertNotEqual(rc, 0)
        self.assertIn("is missing", out)

    def test_low_ram_is_said_not_refused(self):
        with mock.patch.object(setup, "ram_gb", lambda: 16.0):
            rc, out = run(setup.install_k2h, args(models_dir=str(self.root / "models")), "k2h")
        self.assertEqual(rc, 0, out)
        self.assertIn("runs best with about 24 GB", out)


if __name__ == "__main__":
    unittest.main()
