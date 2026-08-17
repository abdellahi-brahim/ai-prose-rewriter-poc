#!/usr/bin/env python3
"""Build grouped AI-style-to-human rewrite data from ARB."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


REQUIRED_ARB_COLUMNS = {
    "pair_id",
    "text",
    "source_text",
    "source_dataset",
    "generator_model",
    "regime",
}
NUMBER_PATTERN = re.compile(r"(?<!\w)[+-]?(?:\d[\d,.]*%?)(?!\w)")
URL_PATTERN = re.compile(r"https?://[^\s)>\]}]+", re.IGNORECASE)
SPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class BuildConfig:
    dataset: str = "giper45/ARB-Dataset"
    split: str = "train"
    rewrite_count: int = 0
    identity_count: int = 0
    identity_fraction: float = 0.20
    review_count: int = 200
    semantic_threshold: float = 0.85
    min_words: int = 8
    max_words: int = 350
    min_length_ratio: float = 0.55
    max_length_ratio: float = 1.80
    train_fraction: float = 0.80
    validation_fraction: float = 0.10
    seed: int = 42
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_batch_size: int = 128
    skip_semantic_filter: bool = False


def validate_config(config: BuildConfig) -> None:
    if config.rewrite_count < 0 or config.identity_count < 0:
        raise ValueError("Example counts cannot be negative.")
    if not 0.0 <= config.identity_fraction <= 1.0:
        raise ValueError("identity_fraction must be between 0 and 1.")
    if not 0.0 < config.semantic_threshold <= 1.0:
        raise ValueError("semantic_threshold must be greater than 0 and at most 1.")
    if not 0.0 < config.train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1.")
    if not 0.0 <= config.validation_fraction < 1.0:
        raise ValueError("validation_fraction must be at least 0 and less than 1.")
    if config.train_fraction + config.validation_fraction >= 1.0:
        raise ValueError("The train and validation fractions must leave room for a test split.")


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return SPACE_PATTERN.sub(" ", str(value)).strip()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.casefold().encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    return len(text.split())


def length_ratio(source: str, target: str) -> float:
    target_words = max(1, word_count(target))
    return word_count(source) / target_words


def protected_values(text: str) -> dict[str, Counter[str]]:
    return {
        "numbers": Counter(NUMBER_PATTERN.findall(text)),
        "urls": Counter(URL_PATTERN.findall(text)),
    }


def changed_protected_values(source: str, target: str) -> list[str]:
    source_values = protected_values(source)
    target_values = protected_values(target)
    return [name for name in source_values if source_values[name] != target_values[name]]


def basic_rejection_reasons(
    source: str,
    target: str,
    config: BuildConfig,
    *,
    identity: bool = False,
) -> list[str]:
    reasons: list[str] = []
    source_words = word_count(source)
    target_words = word_count(target)
    if not source or not target:
        reasons.append("empty_text")
        return reasons
    if min(source_words, target_words) < config.min_words:
        reasons.append("too_short")
    if max(source_words, target_words) > config.max_words:
        reasons.append("too_long")
    if not identity:
        ratio = length_ratio(source, target)
        if ratio < config.min_length_ratio or ratio > config.max_length_ratio:
            reasons.append("length_ratio")
        reasons.extend(f"changed_{name}" for name in changed_protected_values(source, target))
    return reasons


def validate_arb_frame(frame: pd.DataFrame) -> None:
    missing = REQUIRED_ARB_COLUMNS - set(frame.columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"ARB is missing required columns: {names}")


def prepare_arb_rows(frame: pd.DataFrame) -> pd.DataFrame:
    validate_arb_frame(frame)
    prepared = frame.copy()
    prepared["pair_id"] = prepared["pair_id"].map(normalize_text)
    prepared["text"] = prepared["text"].map(normalize_text)
    prepared["source_text"] = prepared["source_text"].map(normalize_text)
    prepared["regime"] = prepared["regime"].map(normalize_text).str.casefold()
    prepared = prepared[(prepared["regime"] == "h2l") & (prepared["pair_id"] != "")]
    return prepared.reset_index(drop=True)


def filter_basic_pairs(
    frame: pd.DataFrame,
    config: BuildConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    seen_pairs: set[tuple[str, str]] = set()

    for row in frame.to_dict(orient="records"):
        source = normalize_text(row["text"])
        target = normalize_text(row["source_text"])
        reasons = basic_rejection_reasons(source, target, config)
        fingerprint = (text_hash(source), text_hash(target))
        if fingerprint in seen_pairs:
            reasons.append("duplicate_pair")
        else:
            seen_pairs.add(fingerprint)

        candidate = {
            "group_id": row["pair_id"],
            "source": source,
            "target": target,
            "pair_type": "rewrite",
            "source_dataset": normalize_text(row["source_dataset"]),
            "generator_model": normalize_text(row["generator_model"]),
            "length_ratio": length_ratio(source, target) if target else np.nan,
        }
        if reasons:
            rejected.append({**candidate, "rejection_reason": ";".join(sorted(set(reasons)))})
        else:
            accepted.append(candidate)

    return pd.DataFrame(accepted), pd.DataFrame(rejected)


def semantic_similarities(
    sources: Sequence[str],
    targets: Sequence[str],
    model_name: str,
    batch_size: int,
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    source_vectors = model.encode(
        list(sources), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
    )
    target_vectors = model.encode(
        list(targets), batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
    )
    return np.sum(source_vectors * target_vectors, axis=1)


def apply_semantic_filter(
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    config: BuildConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if accepted.empty:
        return accepted, rejected
    if config.skip_semantic_filter:
        accepted = accepted.copy()
        accepted["semantic_similarity"] = np.nan
        return accepted, rejected

    accepted = accepted.copy()
    accepted["semantic_similarity"] = semantic_similarities(
        accepted["source"].tolist(),
        accepted["target"].tolist(),
        config.embedding_model,
        config.embedding_batch_size,
    )
    failed = accepted[accepted["semantic_similarity"] < config.semantic_threshold].copy()
    failed["rejection_reason"] = "semantic_similarity"
    kept = accepted[accepted["semantic_similarity"] >= config.semantic_threshold].copy()
    rejected = pd.concat([rejected, failed], ignore_index=True)
    return kept.reset_index(drop=True), rejected.reset_index(drop=True)


def unique_human_records(frame: pd.DataFrame, config: BuildConfig) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    seen_groups: set[str] = set()
    seen_texts: set[str] = set()
    for row in frame.to_dict(orient="records"):
        group_id = normalize_text(row["pair_id"])
        text = normalize_text(row["source_text"])
        fingerprint = text_hash(text)
        if group_id in seen_groups or fingerprint in seen_texts:
            continue
        if basic_rejection_reasons(text, text, config, identity=True):
            continue
        seen_groups.add(group_id)
        seen_texts.add(fingerprint)
        records.append(
            {
                "group_id": group_id,
                "source": text,
                "target": text,
                "pair_type": "identity",
                "source_dataset": normalize_text(row["source_dataset"]),
                "generator_model": "",
                "length_ratio": 1.0,
                "semantic_similarity": 1.0,
            }
        )
    return pd.DataFrame(records)


def sample_examples(
    rewrite_candidates: pd.DataFrame,
    identity_candidates: pd.DataFrame,
    config: BuildConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled_rewrites = rewrite_candidates.sample(frac=1.0, random_state=config.seed)
    identity_groups = identity_candidates["group_id"].drop_duplicates().tolist()
    random.Random(config.seed).shuffle(identity_groups)

    initial_rewrite_count = config.rewrite_count or len(shuffled_rewrites)
    desired_identity_count = config.identity_count or round(
        initial_rewrite_count * config.identity_fraction
    )
    rewrites = shuffled_rewrites
    for _ in range(25):
        if desired_identity_count > len(identity_groups):
            break
        selected_identity_groups = set(identity_groups[:desired_identity_count])
        rewrites = shuffled_rewrites[
            ~shuffled_rewrites["group_id"].isin(selected_identity_groups)
        ]
        if config.rewrite_count > 0:
            rewrites = rewrites.head(config.rewrite_count)
        revised_count = config.identity_count or round(len(rewrites) * config.identity_fraction)
        if revised_count == desired_identity_count:
            break
        desired_identity_count = revised_count

    selected_identity_groups = set(identity_groups[:desired_identity_count])
    identity = identity_candidates[
        identity_candidates["group_id"].isin(selected_identity_groups)
    ].copy()
    rewrites = shuffled_rewrites[
        ~shuffled_rewrites["group_id"].isin(selected_identity_groups)
    ]
    if config.rewrite_count > 0:
        rewrites = rewrites.head(config.rewrite_count)
    rewrites = rewrites.copy()

    if desired_identity_count > len(identity_groups) or len(identity) < desired_identity_count:
        raise ValueError(
            f"Requested {desired_identity_count} identity examples, but only "
            f"{len(identity_groups)} distinct human groups are available."
        )
    if config.rewrite_count > 0 and len(rewrites) < config.rewrite_count:
        raise ValueError(
            f"Requested {config.rewrite_count} rewrite examples, but only {len(rewrites)} are available."
        )
    if set(identity["group_id"]) & set(rewrites["group_id"]):
        raise AssertionError("Identity and rewrite examples share a group_id.")
    return rewrites.reset_index(drop=True), identity.reset_index(drop=True)


def split_group_ids(
    group_ids: Iterable[str],
    config: BuildConfig,
    seed_offset: int,
) -> dict[str, set[str]]:
    groups = sorted(set(group_ids))
    random.Random(config.seed + seed_offset).shuffle(groups)
    total = len(groups)
    train_end = int(total * config.train_fraction)
    validation_end = train_end + int(total * config.validation_fraction)
    return {
        "train": set(groups[:train_end]),
        "validation": set(groups[train_end:validation_end]),
        "test": set(groups[validation_end:]),
    }


def grouped_split(examples: pd.DataFrame, config: BuildConfig) -> dict[str, pd.DataFrame]:
    group_splits: dict[str, set[str]] = defaultdict(set)
    for offset, pair_type in enumerate(("rewrite", "identity")):
        ids = examples.loc[examples["pair_type"] == pair_type, "group_id"]
        split_ids = split_group_ids(ids, config, seed_offset=offset)
        for split_name, values in split_ids.items():
            group_splits[split_name].update(values)

    result = {
        name: examples[examples["group_id"].isin(group_ids)]
        .sample(frac=1.0, random_state=config.seed)
        .reset_index(drop=True)
        for name, group_ids in group_splits.items()
    }
    train_groups = set(result["train"]["group_id"])
    validation_groups = set(result["validation"]["group_id"])
    test_groups = set(result["test"]["group_id"])
    if train_groups & validation_groups or train_groups & test_groups or validation_groups & test_groups:
        raise AssertionError("A group_id appears in more than one split.")
    return result


def make_review_sample(examples: pd.DataFrame, config: BuildConfig) -> pd.DataFrame:
    count = min(config.review_count, len(examples))
    rewrite_count = min(round(count * 0.8), int((examples["pair_type"] == "rewrite").sum()))
    identity_count = min(count - rewrite_count, int((examples["pair_type"] == "identity").sum()))
    remaining = count - rewrite_count - identity_count
    rewrite_count += min(remaining, int((examples["pair_type"] == "rewrite").sum()) - rewrite_count)

    parts = []
    for pair_type, size in (("rewrite", rewrite_count), ("identity", identity_count)):
        if size:
            parts.append(
                examples[examples["pair_type"] == pair_type].sample(
                    n=size, random_state=config.seed + len(parts)
                )
            )
    review = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=config.seed)
    review = review.reset_index(drop=True)
    review.insert(0, "review_id", np.arange(len(review)))
    review["meaning_preserved_Y_N"] = ""
    review["target_more_natural_Y_N_TIE"] = ""
    review["facts_changed_Y_N"] = ""
    review["accept_pair_Y_N"] = ""
    review["notes"] = ""
    return review


def rejection_counts(rejected: pd.DataFrame) -> dict[str, int]:
    counts: Counter[str] = Counter()
    if "rejection_reason" not in rejected:
        return {}
    for reasons in rejected["rejection_reason"].dropna():
        counts.update(str(reasons).split(";"))
    return dict(sorted(counts.items()))


def write_outputs(
    splits: dict[str, pd.DataFrame],
    rejected: pd.DataFrame,
    config: BuildConfig,
    output_dir: Path,
    input_rows: int,
    eligible_rewrite_rows: int,
    available_identity_rows: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        "group_id",
        "source",
        "target",
        "pair_type",
        "source_dataset",
        "generator_model",
        "length_ratio",
        "semantic_similarity",
    ]
    splits["train"][columns].to_csv(output_dir / "train.csv", index=False)
    splits["validation"][columns].to_csv(output_dir / "validation.csv", index=False)
    test = splits["test"][columns]
    test.to_csv(output_dir / "test.csv", index=False)
    test[test["pair_type"] == "rewrite"].to_csv(output_dir / "test_rewrite.csv", index=False)
    test[test["pair_type"] == "identity"].to_csv(output_dir / "test_identity.csv", index=False)
    rejected.to_csv(output_dir / "rejected_pairs.csv", index=False)

    all_examples = pd.concat(splits.values(), ignore_index=True)
    make_review_sample(all_examples, config).to_csv(
        output_dir / "manual_review_sample.csv", index=False
    )
    report = {
        "config": asdict(config),
        "input_rows": input_rows,
        "eligible_rewrite_rows": eligible_rewrite_rows,
        "available_identity_rows": available_identity_rows,
        "accepted_rows": len(all_examples),
        "rejected_rows": len(rejected),
        "rejection_counts": rejection_counts(rejected),
        "splits": {
            name: {
                "rows": len(frame),
                "groups": int(frame["group_id"].nunique()),
                "rewrite_rows": int((frame["pair_type"] == "rewrite").sum()),
                "identity_rows": int((frame["pair_type"] == "identity").sum()),
            }
            for name, frame in splits.items()
        },
    }
    (output_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_arb(dataset_name: str, split: str) -> pd.DataFrame:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    return dataset.to_pandas()


def build_dataset(config: BuildConfig, output_dir: Path) -> dict[str, pd.DataFrame]:
    validate_config(config)
    raw = prepare_arb_rows(load_arb(config.dataset, config.split))
    rewrite_candidates, rejected = filter_basic_pairs(raw, config)
    rewrite_candidates, rejected = apply_semantic_filter(rewrite_candidates, rejected, config)
    identity_candidates = unique_human_records(raw, config)
    rewrites, identities = sample_examples(rewrite_candidates, identity_candidates, config)
    examples = pd.concat([rewrites, identities], ignore_index=True)
    splits = grouped_split(examples, config)
    write_outputs(
        splits,
        rejected,
        config,
        output_dir,
        input_rows=len(raw),
        eligible_rewrite_rows=len(rewrite_candidates),
        available_identity_rows=len(identity_candidates),
    )
    return splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paired prose-rewriting data from ARB's H2L records."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/arb"))
    parser.add_argument("--dataset", default=BuildConfig.dataset)
    parser.add_argument("--split", default=BuildConfig.split)
    parser.add_argument("--rewrite-count", type=int, default=BuildConfig.rewrite_count)
    parser.add_argument("--identity-count", type=int, default=BuildConfig.identity_count)
    parser.add_argument("--identity-fraction", type=float, default=BuildConfig.identity_fraction)
    parser.add_argument("--review-count", type=int, default=BuildConfig.review_count)
    parser.add_argument("--semantic-threshold", type=float, default=BuildConfig.semantic_threshold)
    parser.add_argument("--min-words", type=int, default=BuildConfig.min_words)
    parser.add_argument("--max-words", type=int, default=BuildConfig.max_words)
    parser.add_argument("--min-length-ratio", type=float, default=BuildConfig.min_length_ratio)
    parser.add_argument("--max-length-ratio", type=float, default=BuildConfig.max_length_ratio)
    parser.add_argument("--seed", type=int, default=BuildConfig.seed)
    parser.add_argument("--embedding-model", default=BuildConfig.embedding_model)
    parser.add_argument("--embedding-batch-size", type=int, default=BuildConfig.embedding_batch_size)
    parser.add_argument("--skip-semantic-filter", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BuildConfig(
        dataset=args.dataset,
        split=args.split,
        rewrite_count=args.rewrite_count,
        identity_count=args.identity_count,
        identity_fraction=args.identity_fraction,
        review_count=args.review_count,
        semantic_threshold=args.semantic_threshold,
        min_words=args.min_words,
        max_words=args.max_words,
        min_length_ratio=args.min_length_ratio,
        max_length_ratio=args.max_length_ratio,
        seed=args.seed,
        embedding_model=args.embedding_model,
        embedding_batch_size=args.embedding_batch_size,
        skip_semantic_filter=args.skip_semantic_filter,
    )
    splits = build_dataset(config, args.output_dir)
    summary = ", ".join(f"{name}={len(frame):,}" for name, frame in splits.items())
    print(f"Wrote {summary} to {args.output_dir}")


if __name__ == "__main__":
    main()
