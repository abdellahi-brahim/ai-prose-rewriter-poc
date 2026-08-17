import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.prepare_arb_dataset import (
    BuildConfig,
    apply_semantic_filter,
    basic_rejection_reasons,
    filter_basic_pairs,
    grouped_split,
    prepare_arb_rows,
    sample_examples,
    unique_human_records,
    write_outputs,
)


def arb_frame(groups: int = 20, rewrites_per_group: int = 2) -> pd.DataFrame:
    rows = []
    for group in range(groups):
        human = f"The project group {group} met on Tuesday and agreed to publish 12 reports next month."
        for rewrite in range(rewrites_per_group):
            variant = chr(ord("a") + rewrite)
            rows.append(
                {
                    "pair_id": f"pair-{group}",
                    "text": (
                        f"Project group {group} convened on Tuesday and reached an agreement "
                        f"to publish 12 reports during the following month, wording {variant}."
                    ),
                    "source_text": human,
                    "source_dataset": "fixture",
                    "generator_model": f"model-{rewrite}",
                    "regime": "h2l",
                }
            )
    return pd.DataFrame(rows)


class PrepareArbDatasetTest(unittest.TestCase):
    def setUp(self):
        self.config = BuildConfig(
            rewrite_count=20,
            identity_count=5,
            review_count=10,
            min_words=4,
            skip_semantic_filter=True,
        )

    def test_protected_numbers_are_rejected(self):
        reasons = basic_rejection_reasons(
            "The team published 12 reports last year.",
            "The team published 13 reports last year.",
            self.config,
        )
        self.assertIn("changed_numbers", reasons)

    def test_identity_and_rewrite_groups_are_disjoint(self):
        raw = prepare_arb_rows(arb_frame())
        rewrites, rejected = filter_basic_pairs(raw, self.config)
        self.assertTrue(rejected.empty)
        identities = unique_human_records(raw, self.config)
        selected_rewrites, selected_identities = sample_examples(
            rewrites, identities, self.config
        )
        self.assertFalse(
            set(selected_rewrites["group_id"]) & set(selected_identities["group_id"])
        )

    def test_automatic_identity_fraction_converges(self):
        config = BuildConfig(
            rewrite_count=0,
            identity_count=0,
            identity_fraction=0.2,
            min_words=4,
            skip_semantic_filter=True,
        )
        raw = prepare_arb_rows(arb_frame(groups=50))
        rewrites, _ = filter_basic_pairs(raw, config)
        identities = unique_human_records(raw, config)
        selected_rewrites, selected_identities = sample_examples(rewrites, identities, config)
        self.assertEqual(len(selected_identities), round(len(selected_rewrites) * 0.2))
        self.assertFalse(
            set(selected_rewrites["group_id"]) & set(selected_identities["group_id"])
        )

    def test_semantic_filter_records_rejection(self):
        raw = prepare_arb_rows(arb_frame(groups=2, rewrites_per_group=1))
        accepted, rejected = filter_basic_pairs(raw, self.config)
        config = BuildConfig(min_words=4, semantic_threshold=0.85)
        with patch(
            "src.prepare_arb_dataset.semantic_similarities",
            return_value=np.array([0.95, 0.70]),
        ):
            accepted, rejected = apply_semantic_filter(accepted, rejected, config)
        self.assertEqual(1, len(accepted))
        self.assertIn("semantic_similarity", set(rejected["rejection_reason"]))

    def test_splits_have_no_group_leakage_and_write_expected_files(self):
        raw = prepare_arb_rows(arb_frame())
        rewrites, rejected = filter_basic_pairs(raw, self.config)
        rewrites["semantic_similarity"] = 0.95
        identities = unique_human_records(raw, self.config)
        rewrites, identities = sample_examples(rewrites, identities, self.config)
        splits = grouped_split(pd.concat([rewrites, identities], ignore_index=True), self.config)

        group_sets = [set(frame["group_id"]) for frame in splits.values()]
        self.assertFalse(group_sets[0] & group_sets[1])
        self.assertFalse(group_sets[0] & group_sets[2])
        self.assertFalse(group_sets[1] & group_sets[2])

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            write_outputs(
                splits,
                rejected,
                self.config,
                output,
                input_rows=len(raw),
                eligible_rewrite_rows=len(rewrites),
                available_identity_rows=len(identities),
            )
            expected = {
                "train.csv",
                "validation.csv",
                "test.csv",
                "test_rewrite.csv",
                "test_identity.csv",
                "manual_review_sample.csv",
                "rejected_pairs.csv",
                "dataset_report.json",
            }
            self.assertEqual(expected, {path.name for path in output.iterdir()})


if __name__ == "__main__":
    unittest.main()
