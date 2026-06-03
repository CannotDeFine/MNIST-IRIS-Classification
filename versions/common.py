"""Shared helpers for roadmap-version experiment runners."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from classification_experiments import (  # noqa: E402
    ExperimentConfig,
    build_experiment_configs,
    run_experiment,
    run_suite,
)
from classification_nn import (  # noqa: E402
    _metrics_for_json,
    evaluate_checkpoint,
    save_checkpoint,
    save_metrics,
    train_dataset,
)


RESULTS_DIR = ROOT / "results" / "versions"


def build_version_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--dataset", choices=["iris", "mnist", "all"], default="all")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser


def selected_ablation_configs(dataset: str, profile: str, names: Iterable[str]) -> list[ExperimentConfig]:
    name_set = set(names)
    return [
        config
        for config in build_experiment_configs(dataset, profile=profile)
        if config.name in name_set
    ]


def _datasets(dataset: str) -> list[str]:
    return ["iris", "mnist"] if dataset == "all" else [dataset]


def run_version_configs(
    version_name: str,
    names: Iterable[str],
    dataset: str,
    profile: str,
    output_dir: Path | None = None,
    verbose: bool = False,
) -> list[dict]:
    output_dir = output_dir or RESULTS_DIR / version_name
    output_dir.mkdir(parents=True, exist_ok=True)

    configs: list[ExperimentConfig] = []
    for dataset_name in _datasets(dataset):
        configs.extend(selected_ablation_configs(dataset_name, profile=profile, names=names))

    if not configs:
        raise ValueError(f"no experiments selected for dataset={dataset!r}, profile={profile!r}")

    results = [run_experiment(config, verbose=verbose) for config in configs]
    json_path = output_dir / f"{profile}.json"
    save_metrics(results, json_path)

    print(f"results saved to {json_path}")
    return results


def run_component_suites(
    version_name: str,
    suites: Iterable[str],
    dataset: str,
    profile: str,
    output_dir: Path | None = None,
    verbose: bool = False,
) -> list[dict]:
    output_dir = output_dir or RESULTS_DIR / version_name
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []
    for suite in suites:
        results: list[dict] = []
        for dataset_name in _datasets(dataset):
            results.extend(run_suite(dataset_name, profile=profile, suite=suite, verbose=verbose))
        json_path = output_dir / f"{suite}_{profile}.json"
        save_metrics(results, json_path)
        print(f"{suite}: results saved to {json_path}")
        all_results.extend(results)
    return all_results


def build_final_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the final selected MNIST MLP.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "final_selected_model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mnist-limit", type=int, default=None)
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def run_final_mnist(args: argparse.Namespace) -> list[dict]:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "final_mnist_mlp.npz"
    train_path = output_dir / "train_val_metrics.json"
    test_path = output_dir / "test_metrics.json"

    train_metrics = train_dataset(
        dataset="mnist",
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        l2=0.0,
        optimizer="momentum",
        momentum=args.momentum,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        mnist_limit=args.mnist_limit,
        verbose=args.verbose,
        return_artifacts=True,
    )
    save_checkpoint(
        train_metrics["model"],
        "mnist",
        train_metrics["meta"],
        train_metrics["config"],
        model_path,
    )
    clean_train_metrics = _metrics_for_json(train_metrics)
    save_metrics([clean_train_metrics], train_path)
    print(f"train/validation metrics saved to {train_path}")
    print(f"checkpoint saved to {model_path}")

    results = [clean_train_metrics]
    if not args.skip_test:
        test_metrics = evaluate_checkpoint(model_path, split="test", mnist_limit=args.mnist_limit)
        save_metrics([test_metrics], test_path)
        print(f"test metrics saved to {test_path}")
        results.append(test_metrics)
    return results
