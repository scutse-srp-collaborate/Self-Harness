"""SWE-bench manifest and split validation tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from self_harness_eval.manifest import (
    DEFAULT_MANIFEST_PATH,
    EXPECTED_HELDOUT,
    EXPECTED_TOTAL,
    EXPECTED_TRAIN,
    ManifestError,
    load_swebench_manifest,
    select_task_ids,
    validate_swebench_manifest,
)
from self_harness_eval.tb2_manifest import (
    Tb2ManifestError,
    load_tb2_manifest,
    validate_tb2_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _raw_manifest() -> dict:
    return json.loads(DEFAULT_MANIFEST_PATH.read_text(encoding="utf-8"))


class SwebenchManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.manifest = load_swebench_manifest()

    def test_counts_uniqueness_disjointness_and_order(self) -> None:
        self.assertEqual(len(self.manifest.held_in_task_ids), EXPECTED_TRAIN)
        self.assertEqual(len(self.manifest.held_out_task_ids), EXPECTED_HELDOUT)
        self.assertEqual(len(self.manifest.task_ids), EXPECTED_TOTAL)
        self.assertEqual(len(set(self.manifest.task_ids)), EXPECTED_TOTAL)
        self.assertFalse(set(self.manifest.held_in_task_ids) & set(self.manifest.held_out_task_ids))
        raw = _raw_manifest()
        self.assertEqual(list(self.manifest.held_in_task_ids), raw["held_in_task_ids"])
        self.assertEqual(list(self.manifest.held_out_task_ids), raw["held_out_task_ids"])
        self.assertEqual(self.manifest.count_by_split(), {"train": 67, "heldout": 33, "total": 100})

    def test_distributions_follow_five_domain_stratification(self) -> None:
        distributions = self.manifest.distributions()
        self.assertIn("stratified", distributions["sampling_note"])
        self.assertEqual(
            distributions["by_domain_total"],
            {"Dev_Tooling": 15, "ML_Data_Processing": 11, "Scientific_Symbolic": 19,
             "Visualization": 7, "Web_Networking": 48},
        )
        declared = self.manifest.repo_distribution_in_100
        self.assertEqual(sum(declared.values()), 100)
        self.assertGreater(declared["django/django"], declared["sympy/sympy"])

    def test_task_repository_and_domain_mapping(self) -> None:
        self.assertEqual(self.manifest.repository_of("django__django-14017"), "django/django")
        self.assertEqual(self.manifest.domain_of("django__django-14017"), "Web_Networking")
        self.assertEqual(
            self.manifest.repository_of("sphinx-doc__sphinx-8548"), "sphinx-doc/sphinx"
        )
        self.assertEqual(self.manifest.domain_of("sympy__sympy-21930"), "Scientific_Symbolic")

    def test_validation_rejects_wrong_counts(self) -> None:
        raw = _raw_manifest()
        raw["held_in_task_ids"] = raw["held_in_task_ids"][:-1]
        with self.assertRaises(ManifestError):
            self._validate(raw)

    def test_validation_rejects_duplicate_ids(self) -> None:
        raw = _raw_manifest()
        raw["held_in_task_ids"][1] = raw["held_in_task_ids"][0]
        with self.assertRaises(ManifestError):
            self._validate(raw)

    def test_validation_rejects_split_overlap(self) -> None:
        raw = _raw_manifest()
        moved = raw["held_in_task_ids"].pop()
        raw["held_out_task_ids"][0] = moved
        with self.assertRaises(ManifestError):
            self._validate(raw)

    def test_validation_rejects_unknown_repository(self) -> None:
        raw = _raw_manifest()
        raw["held_in_task_ids"][0] = "unknown__repo-123"
        raw["repo_distribution_in_100"] = {
            key: value for key, value in raw["repo_distribution_in_100"].items()
        }
        with self.assertRaises(ManifestError):
            self._validate(raw)

    def test_validation_rejects_repo_distribution_mismatch(self) -> None:
        raw = _raw_manifest()
        raw["repo_distribution_in_100"]["django/django"] += 1
        raw["repo_distribution_in_100"]["sympy/sympy"] -= 1
        with self.assertRaises(ManifestError):
            self._validate(raw)

    def test_manifest_path_override(self) -> None:
        target = Path(self.tmp.name) / "manifest.json"
        target.write_text(json.dumps(_raw_manifest()), encoding="utf-8")
        manifest = load_swebench_manifest(target)
        self.assertEqual(manifest.path, target.resolve())
        self.assertEqual(manifest.task_ids, self.manifest.task_ids)

    def test_select_task_ids_preserves_manifest_order(self) -> None:
        selected = select_task_ids(
            self.manifest,
            splits=("train",),
            task_ids=("sympy__sympy-21930", "astropy__astropy-13398"),
        )
        self.assertEqual(
            list(selected["train"]), ["astropy__astropy-13398", "sympy__sympy-21930"]
        )
        with self.assertRaises(ManifestError):
            select_task_ids(self.manifest, splits=("train",), task_ids=("astropy__astropy-7336",))
        with self.assertRaises(ManifestError):
            select_task_ids(self.manifest, splits=("train",), task_ids=("not-a-task",))

    def test_task_id_hash_is_order_sensitive(self) -> None:
        first = self.manifest.task_ids_sha256
        raw = _raw_manifest()
        raw["held_in_task_ids"].reverse()
        swapped = self._validate(raw)
        self.assertNotEqual(first, swapped.task_ids_sha256)

    def _validate(self, raw: dict):
        path = Path(self.tmp.name) / "mutated.json"
        path.write_text(json.dumps(raw), encoding="utf-8")
        return validate_swebench_manifest(raw, path=path)


class Tb2ManifestTest(unittest.TestCase):
    def test_clean64_manifest_counts_and_splits(self) -> None:
        manifest = load_tb2_manifest()
        self.assertEqual(manifest.counts(), {"train": 43, "heldout": 21, "total": 64})
        self.assertEqual(manifest.version, "2.0-clean64")
        self.assertEqual(len(set(manifest.task_ids)), 64)
        self.assertIn("build_cython_ext", manifest.split_task_ids("train"))
        self.assertIn("headless_terminal", manifest.split_task_ids("heldout"))
        self.assertTrue(manifest.strata())

    def test_manifest_matches_readme_lists(self) -> None:
        readme = (REPO_ROOT / "eval" / "README.md").read_text(encoding="utf-8")
        train_block = readme.split("## Train (43)")[1].split("```text")[1].split("```")[0]
        held_block = readme.split("## Heldout (21)")[1].split("```text")[1].split("```")[0]
        train = [line.strip() for line in train_block.strip().splitlines() if line.strip()]
        heldout = [line.strip() for line in held_block.strip().splitlines() if line.strip()]
        manifest = load_tb2_manifest()
        self.assertEqual(list(manifest.split_task_ids("train")), train)
        self.assertEqual(list(manifest.split_task_ids("heldout")), heldout)

    def test_validation_rejects_bad_split_and_counts(self) -> None:
        raw = json.loads(
            (REPO_ROOT / "eval" / "manifests" / "tb2_clean64.json").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tb2.json"
            bad = json.loads(json.dumps(raw))
            bad["tasks"][0]["split"] = "validation"
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(Tb2ManifestError):
                validate_tb2_manifest(bad, path=path)

            bad = json.loads(json.dumps(raw))
            bad["tasks"] = bad["tasks"][:-1]
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(Tb2ManifestError):
                validate_tb2_manifest(bad, path=path)

            bad = json.loads(json.dumps(raw))
            bad["tasks"][1]["task_id"] = bad["tasks"][0]["task_id"]
            path.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(Tb2ManifestError):
                validate_tb2_manifest(bad, path=path)


if __name__ == "__main__":
    unittest.main()
