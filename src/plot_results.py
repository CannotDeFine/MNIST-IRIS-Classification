"""Plot experiment outputs and checkpoint evaluations."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
import numpy as np

try:
    from .classification_nn import evaluate_checkpoint, load_checkpoint, prepare_iris_split, prepare_mnist_split
    from .metrics import confusion_matrix, per_class_accuracy
except ImportError:  # pragma: no cover - direct script execution.
    from classification_nn import evaluate_checkpoint, load_checkpoint, prepare_iris_split, prepare_mnist_split
    from metrics import confusion_matrix, per_class_accuracy


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
INK = "black"
GRID = "grey"
BLUE = "#98A1B1"
LIGHT_BLUE = "#AEBFC1"
GREEN = "#8FA998"
RED = "#B36A6F"
PURPLE = "#9E95B3"
GOLD = "#B8A36F"
GRAY = "#8E8E8E"
LIGHT_GRAY = "#D5D5D5"
BAR = "#AEBFC1"
BAR_GRADIENT = ["#D7DFE0", "#C8D4D6", "#B8C8CA", "#A8BABD", "#B36A6F"]


def _preferred_font() -> str:
    for font in font_manager.fontManager.ttflist:
        if font.name == "Arial":
            return "Arial"
    return "DejaVu Sans"


def _load_metrics(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return payload
    return [payload]


def _apply_paper_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": _preferred_font(),
            "font.size": 8,
            "font.weight": "bold",
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "axes.labelweight": "bold",
            "axes.edgecolor": "black",
            "axes.linewidth": 1.2,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "axes.spines.bottom": True,
            "axes.spines.left": True,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.5,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _metric_label(item: dict[str, Any], source: Path) -> str:
    if "name" in item:
        return str(item["name"])
    dataset = str(item.get("dataset", "")).strip()
    stem = source.stem
    return f"{dataset}_{stem}" if dataset and dataset not in stem else stem


def _collect_comparison_rows(
    metrics_paths: list[Path],
    dataset: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in metrics_paths:
        for item in _load_metrics(path):
            if dataset is not None and item.get("dataset") != dataset:
                continue
            history = item.get("history", {})
            epochs_trained = item.get("epochs_trained")
            if epochs_trained is None and "loss" in history:
                epochs_trained = len(history["loss"]) - 1
            stopped_early = item.get("stopped_early")
            max_epochs = item.get("config", {}).get("epochs")
            if stopped_early is None and epochs_trained is not None and max_epochs is not None:
                stopped_early = epochs_trained < max_epochs
            rows.append(
                {
                    "label": _metric_label(item, path),
                    "source": str(path),
                    "dataset": item.get("dataset", ""),
                    "val_accuracy": item.get("final_val_accuracy", item.get("accuracy")),
                    "train_accuracy": item.get("final_train_accuracy"),
                    "val_loss": item.get("final_val_loss", item.get("loss")),
                    "train_loss": item.get("final_train_loss"),
                    "epochs_trained": epochs_trained,
                    "stopped_early": stopped_early,
                }
            )
    return rows


def _write_comparison_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label",
        "dataset",
        "val_accuracy",
        "train_accuracy",
        "val_loss",
        "train_loss",
        "epochs_trained",
        "stopped_early",
        "source",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def plot_metric_comparison(
    metrics_paths: list[Path],
    output_dir: Path,
    output_prefix: str,
    dataset: str | None = None,
    title: str | None = None,
) -> list[Path]:
    """Create paper-style comparison figures from one or more metrics JSON files."""
    rows = _collect_comparison_rows(metrics_paths, dataset=dataset)
    if not rows:
        raise ValueError("no matching metrics found")

    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / f"{output_prefix}_summary.csv"
    _write_comparison_csv(rows, table_path)

    _apply_paper_style()
    outputs = [table_path]
    labels = [row["label"] for row in rows]
    y = np.arange(len(rows))
    height = max(3.6, 0.42 * len(rows) + 1.5)
    base_title = title or output_prefix.replace("_", " ").title()

    accuracy_values = [row["val_accuracy"] for row in rows]
    if all(value is not None for value in accuracy_values):
        fig, ax = plt.subplots(figsize=(8.2, height))
        values = np.asarray(accuracy_values, dtype=np.float64)
        colors = plt.cm.viridis(np.linspace(0.20, 0.82, len(values)))
        ax.barh(y, values, color=colors, edgecolor="#202020", linewidth=0.4)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("validation accuracy")
        ax.set_xlim(max(0.0, min(values) - 0.05), min(1.0, max(values) + 0.03))
        ax.set_title(f"{base_title}: validation accuracy")
        for idx, value in enumerate(values):
            ax.text(value + 0.002, idx, f"{value:.4f}", va="center", fontsize=8)
        fig.tight_layout()
        path = output_dir / f"{output_prefix}_val_accuracy.png"
        fig.savefig(path)
        plt.close(fig)
        outputs.append(path)

    epoch_values = [row["epochs_trained"] for row in rows]
    if all(value is not None for value in epoch_values):
        fig, ax = plt.subplots(figsize=(8.2, height))
        values = np.asarray(epoch_values, dtype=np.float64)
        colors = plt.cm.cividis(np.linspace(0.20, 0.82, len(values)))
        ax.barh(y, values, color=colors, edgecolor="#202020", linewidth=0.4)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_xlabel("epochs trained")
        ax.set_title(f"{base_title}: optimization effort")
        ax.set_xlim(0, max(values) * 1.15 if max(values) > 0 else 1)
        for idx, value in enumerate(values):
            ax.text(value + max(values) * 0.015 + 0.02, idx, f"{int(value)}", va="center", fontsize=8)
        fig.tight_layout()
        path = output_dir / f"{output_prefix}_epochs_trained.png"
        fig.savefig(path)
        plt.close(fig)
        outputs.append(path)

    return outputs


def _items_by_name(metrics_path: Path) -> dict[str, dict[str, Any]]:
    return {str(item["name"]): item for item in _load_metrics(metrics_path) if "name" in item}


def _final_item(metrics_path: Path) -> dict[str, Any]:
    items = _load_metrics(metrics_path)
    if not items:
        raise ValueError(f"empty metrics file: {metrics_path}")
    return items[0]


def _save_report_table(rows: list[dict[str, Any]], path: Path) -> Path:
    if not rows:
        raise ValueError("empty report table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _annotate_points(ax, xs: list[float], ys: list[float], labels: list[str]) -> None:
    for x, y_value, label in zip(xs, ys, labels, strict=True):
        ax.annotate(
            label,
            (x, y_value),
            textcoords="offset points",
            xytext=(0, 9),
            ha="center",
            fontsize=8,
        )


def _panel_label(ax, label: str) -> None:
    ax.text(
        -0.13,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
    )


def _save_figure_pair(fig, output_dir: Path, stem: str) -> list[Path]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return [png, pdf]


def _style_axes(ax) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")
    ax.tick_params(axis="both", labelsize=7, width=1.0)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    ax.set_axisbelow(True)


def _save_single_report_figure(fig, output_dir: Path, stem: str) -> list[Path]:
    paths = _save_figure_pair(fig, output_dir, stem)
    # Keep the PNG as the first-class report artifact while preserving a PDF
    # for final document assembly.
    return paths


def plot_report_figures(results_dir: Path, output_dir: Path) -> list[Path]:
    """Generate narrative figures for the report."""
    _apply_paper_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    baseline = _items_by_name(results_dir / "baseline_progression_full.json")
    final = _final_item(results_dir / "mnist_full_es100_width256_momentum_no_l2.json")
    progression = [
        ("Random\nnetwork", baseline["mnist_untrained_mlp"]),
        ("Linear\nsoftmax", baseline["mnist_linear_softmax"]),
        ("Small\nMLP", baseline["mnist_small_hidden"]),
        ("Stable\nMLP", baseline["mnist_current_mlp"]),
        ("Final\ntuned MLP", final),
    ]
    labels = [label for label, _ in progression]
    x = list(range(len(progression)))
    val_acc = [float(item["final_val_accuracy"]) for _, item in progression]
    train_acc = [float(item["final_train_accuracy"]) for _, item in progression]

    fig, ax = plt.subplots(figsize=(3.6, 1.8))
    ax.plot(x, val_acc, marker="o", markersize=3.6, linewidth=1.5, color=RED, label="validation", zorder=3)
    ax.plot(x, train_acc, marker="s", markersize=3.2, linewidth=1.2, color=GRAY, linestyle="--", label="train", zorder=3)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.05, 1.02)
    ax.set_ylabel("Accuracy", fontsize=8, fontweight="bold")
    ax.set_title("MNIST Model Evolution", fontsize=8, fontweight="bold")
    ax.legend(loc="lower right", fontsize=6.5, handlelength=1.1, handletextpad=0.35)
    _style_axes(ax)
    fig.tight_layout()
    paths = _save_single_report_figure(fig, output_dir, "report_mnist_method_progression")
    plt.close(fig)
    outputs.extend(paths)
    outputs.append(
        _save_report_table(
            [
                {
                    "stage": label.replace("\n", " "),
                    "val_accuracy": item["final_val_accuracy"],
                    "train_accuracy": item["final_train_accuracy"],
                    "epochs_trained": item.get("epochs_trained", len(item.get("history", {}).get("loss", [])) - 1),
                }
                for label, item in progression
            ],
            output_dir / "report_mnist_method_progression.csv",
        )
    )

    cumulative_stages = [
        ("Random\ninit", baseline["mnist_untrained_mlp"], "initialization only"),
        ("+ Linear\nsoftmax", baseline["mnist_linear_softmax"], "trainable softmax classifier"),
        ("+ Hidden\nlayer", baseline["mnist_small_hidden"], "nonlinear MLP capacity"),
        ("+ Stable\ntraining", baseline["mnist_current_mlp"], "scaling + He init + mini-batch"),
        ("+ Unified\ntuning", final, "width 256 + momentum + early stopping"),
    ]
    cumulative_labels = [label for label, _, _ in cumulative_stages]
    cumulative_values = [float(item["final_val_accuracy"]) for _, item, _ in cumulative_stages]
    cumulative_epochs = [
        int(item.get("epochs_trained", len(item.get("history", {}).get("loss", [])) - 1))
        for _, item, _ in cumulative_stages
    ]
    cumulative_delta = [0.0] + [
        cumulative_values[idx] - cumulative_values[idx - 1]
        for idx in range(1, len(cumulative_values))
    ]
    x_bar = np.arange(len(cumulative_stages))
    bar_colors = BAR_GRADIENT
    fig, ax = plt.subplots(figsize=(3.7, 1.8))
    ax.bar(x_bar, cumulative_values, width=0.36, color=bar_colors, edgecolor=INK, linewidth=0.8, zorder=2)
    ax.set_xticks(x_bar, cumulative_labels)
    ax.set_ylim(0, 1.14)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Val. Acc.", fontsize=8, fontweight="bold")
    ax.set_title("Cumulative Improvements", fontsize=8, fontweight="bold")
    for idx, (value, delta) in enumerate(zip(cumulative_values, cumulative_delta, strict=True)):
        label = f"{value:.3f}" if idx == 0 else f"+{delta:.3f}"
        ax.text(idx, value + 0.018, label, ha="center", va="bottom", fontsize=6.5, fontweight="bold")
    ax_epoch = ax.twinx()
    ax_epoch.plot(
        x_bar,
        cumulative_epochs,
        color=INK,
        marker="D",
        markersize=3.0,
        linewidth=1.1,
        zorder=4,
    )
    ax_epoch.set_ylim(0, max(cumulative_epochs) * 1.25 if max(cumulative_epochs) > 0 else 1)
    ax_epoch.set_ylabel("Epochs", fontsize=8, fontweight="bold")
    ax_epoch.tick_params(axis="y", labelsize=7, width=1.0)
    for label in ax_epoch.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax_epoch.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")
    ax_epoch.grid(False)
    _style_axes(ax)
    fig.tight_layout()
    paths = _save_single_report_figure(fig, output_dir, "report_mnist_cumulative_ablation")
    plt.close(fig)
    outputs.extend(paths)
    outputs.append(
        _save_report_table(
            [
                {
                    "stage": label.replace("\n", " "),
                    "added_method": method,
                    "val_accuracy": item["final_val_accuracy"],
                    "epochs_trained": item.get("epochs_trained", len(item.get("history", {}).get("loss", [])) - 1),
                    "delta_vs_previous": delta,
                }
                for (label, item, method), delta in zip(cumulative_stages, cumulative_delta, strict=True)
            ],
            output_dir / "report_mnist_cumulative_ablation.csv",
        )
    )

    final_paths = [
        results_dir / "mnist_full_es100_current_mlp.json",
        results_dir / "mnist_full_es100_width_256.json",
        results_dir / "mnist_full_es100_momentum.json",
        results_dir / "mnist_full_es100_l2_1e-3.json",
        results_dir / "mnist_full_es100_width256_momentum_no_l2.json",
    ]
    final_labels = ["128+SGD", "256+SGD", "128+Mom.", "128+L2", "256+Mom."]
    final_items = [_final_item(path) for path in final_paths]
    final_epochs = [int(item.get("epochs_trained", len(item.get("history", {}).get("loss", [])) - 1)) for item in final_items]
    final_acc = [float(item["final_val_accuracy"]) for item in final_items]

    loss_items = _items_by_name(results_dir / "loss_comparison.json")
    loss_names = ["mnist_sigmoid_mse", "mnist_softmax_cross_entropy"]
    loss_labels = ["Sigmoid + MSE", "Softmax + CE"]
    loss_values = [float(loss_items[name]["final_val_accuracy"]) for name in loss_names]
    fig, ax = plt.subplots(figsize=(2.8, 1.75))
    ax.bar(loss_labels, loss_values, width=0.38, color=[LIGHT_GRAY, RED], edgecolor=INK, linewidth=0.8, zorder=2)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Val. Acc.", fontsize=8, fontweight="bold")
    ax.set_title("Loss Function", fontsize=8, fontweight="bold")
    for idx, value in enumerate(loss_values):
        ax.text(idx, value + 0.025, f"{value:.3f}", ha="center", fontsize=6.8, fontweight="bold")
    _style_axes(ax)
    fig.tight_layout()
    paths = _save_single_report_figure(fig, output_dir, "report_mnist_loss_function_check")
    plt.close(fig)
    outputs.extend(paths)

    l2_items = _load_metrics(results_dir / "l2_sweep.json")
    mnist_l2 = [item for item in l2_items if item.get("dataset") == "mnist"]
    l2_labels = [str(item["config"]["l2"]) for item in mnist_l2]
    l2_values = [float(item["final_val_accuracy"]) for item in mnist_l2]
    fig, ax = plt.subplots(figsize=(3.2, 1.75))
    ax.plot(range(len(l2_values)), l2_values, marker="o", markersize=3.5, linewidth=1.4, color=PURPLE, zorder=3)
    ax.set_xticks(range(len(l2_labels)), l2_labels)
    ax.set_xlabel("L2 lambda", fontsize=8, fontweight="bold")
    ax.set_ylabel("Val. Acc.", fontsize=8, fontweight="bold")
    ax.set_title("L2 Sweep", fontsize=8, fontweight="bold")
    _style_axes(ax)
    fig.tight_layout()
    paths = _save_single_report_figure(fig, output_dir, "report_mnist_l2_sweep")
    plt.close(fig)
    outputs.extend(paths)

    lr_items = [item for item in _load_metrics(results_dir / "lr_sweep.json") if item.get("dataset") == "mnist"]
    width_items = [item for item in _load_metrics(results_dir / "width_sweep.json") if item.get("dataset") == "mnist"]
    fig, ax = plt.subplots(figsize=(3.2, 1.75))
    lr_x = [float(item["config"]["learning_rate"]) for item in lr_items]
    lr_y = [float(item["final_val_accuracy"]) for item in lr_items]
    ax.plot(lr_x, lr_y, marker="o", markersize=3.5, linewidth=1.4, color=RED, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("Learning rate", fontsize=8, fontweight="bold")
    ax.set_ylabel("Val. Acc.", fontsize=8, fontweight="bold")
    ax.set_title("Learning-rate Sweep", fontsize=8, fontweight="bold")
    _style_axes(ax)
    fig.tight_layout()
    paths = _save_single_report_figure(fig, output_dir, "report_mnist_learning_rate_sweep")
    plt.close(fig)
    outputs.extend(paths)

    fig, ax = plt.subplots(figsize=(3.2, 1.75))
    width_x = [int(item["config"]["hidden_dim"]) for item in width_items]
    width_y = [float(item["final_val_accuracy"]) for item in width_items]
    ax.plot(width_x, width_y, marker="o", markersize=3.5, linewidth=1.4, color=GREEN, zorder=3)
    ax.set_xlabel("Hidden units", fontsize=8, fontweight="bold")
    ax.set_ylabel("Val. Acc.", fontsize=8, fontweight="bold")
    ax.set_title("Hidden-width Sweep", fontsize=8, fontweight="bold")
    _style_axes(ax)
    fig.tight_layout()
    paths = _save_single_report_figure(fig, output_dir, "report_mnist_hidden_width_sweep")
    plt.close(fig)
    outputs.extend(paths)

    strategy_items = [item for item in _load_metrics(results_dir / "training_strategies.json") if item.get("dataset") == "mnist"]
    strategy_labels = ["SGD", "Momentum", "LR decay", "Early stopping"]
    strategy_values = [float(item["final_val_accuracy"]) for item in strategy_items]
    fig, ax = plt.subplots(figsize=(3.2, 1.75))
    strategy_colors = [LIGHT_GRAY, BLUE, GRAY, GOLD]
    ax.bar(strategy_labels, strategy_values, width=0.42, color=strategy_colors, edgecolor=INK, linewidth=0.8, zorder=2)
    ax.set_ylim(min(strategy_values) - 0.03, max(strategy_values) + 0.03)
    ax.set_ylabel("Val. Acc.", fontsize=8, fontweight="bold")
    ax.set_title("Training Strategy", fontsize=8, fontweight="bold")
    for idx, value in enumerate(strategy_values):
        ax.text(idx, value + 0.005, f"{value:.3f}", ha="center", fontsize=6.5, fontweight="bold")
    _style_axes(ax)
    fig.tight_layout()
    paths = _save_single_report_figure(fig, output_dir, "report_mnist_training_strategies")
    plt.close(fig)
    outputs.extend(paths)

    final_labels = ["128 + SGD", "256 + SGD", "128 + momentum", "128 + L2", "256 + momentum"]
    fig, ax = plt.subplots(figsize=(4.6, 2.25))
    sizes = [42, 42, 48, 42, 62]
    colors = [LIGHT_GRAY, LIGHT_BLUE, GOLD, RED, GREEN]
    ax.scatter(final_epochs, final_acc, s=sizes, c=colors, edgecolors=INK, linewidths=0.7, zorder=3)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5.0,
            markerfacecolor=color,
            markeredgecolor=INK,
            markeredgewidth=0.7,
            label=label,
        )
        for label, color in zip(final_labels, colors, strict=True)
    ]
    ax.legend(
        handles=legend_handles,
        ncol=1,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=6.4,
        labelspacing=0.6,
        handlelength=0.75,
        handletextpad=0.3,
        borderaxespad=0.0,
    )
    ax.set_xlim(min(final_epochs) - 2.2, max(final_epochs) + 2.2)
    ax.set_ylim(min(final_acc) - 0.00045, max(final_acc) + 0.00085)
    ax.set_xlabel("Epochs before stop", fontsize=8, fontweight="bold")
    ax.set_ylabel("Val. Acc.", fontsize=8, fontweight="bold")
    ax.set_title("Accuracy-Effort Tradeoff", fontsize=8, fontweight="bold")
    _style_axes(ax)
    fig.subplots_adjust(left=0.13, right=0.70, bottom=0.25, top=0.82)
    paths = _save_single_report_figure(fig, output_dir, "report_mnist_final_tradeoff")
    plt.close(fig)
    outputs.extend(paths)
    outputs.append(
        _save_report_table(
            [
                {
                    "candidate": label,
                    "val_accuracy": item["final_val_accuracy"],
                    "val_loss": item["final_val_loss"],
                    "epochs_trained": epoch,
                    "stopped_early": item.get("stopped_early"),
                }
                for label, item, epoch in zip(final_labels, final_items, final_epochs, strict=True)
            ],
            output_dir / "report_mnist_final_tradeoff.csv",
        )
    )

    return outputs


def plot_learning_curves(metrics_path: Path, output_dir: Path) -> list[Path]:
    """Plot loss and accuracy curves for every item in a metrics JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for item in _load_metrics(metrics_path):
        history = item.get("history")
        if not history:
            continue
        dataset = item.get("dataset", "dataset")
        stem = f"{metrics_path.stem}_{dataset}"
        epochs = np.arange(len(history["loss"]))

        loss_path = output_dir / f"{stem}_loss.png"
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, history["loss"], label="train")
        if "val_loss" in history:
            plt.plot(epochs, history["val_loss"], label="validation")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(f"{dataset} loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(loss_path, dpi=160)
        plt.close()
        outputs.append(loss_path)

        acc_path = output_dir / f"{stem}_accuracy.png"
        plt.figure(figsize=(7, 4))
        plt.plot(epochs, history["accuracy"], label="train")
        if "val_accuracy" in history:
            plt.plot(epochs, history["val_accuracy"], label="validation")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title(f"{dataset} accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(acc_path, dpi=160)
        plt.close()
        outputs.append(acc_path)
    return outputs


def _checkpoint_split(path: Path, split: str, mnist_limit: int | None = None):
    model, dataset, meta, _ = load_checkpoint(path)
    if dataset == "iris":
        X, y, _ = prepare_iris_split(split, meta=meta)
    elif dataset == "mnist":
        X, y, _ = prepare_mnist_split(split, limit=mnist_limit, meta=meta)
    else:
        raise ValueError("checkpoint dataset must be 'iris' or 'mnist'")
    return model, dataset, X, y


def save_confusion_outputs(
    checkpoint: Path,
    split: str,
    figure_dir: Path,
    table_dir: Path,
    mnist_limit: int | None = None,
) -> tuple[Path, Path]:
    """Save confusion matrix as CSV and heatmap PNG for a checkpoint."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    model, dataset, X, y = _checkpoint_split(checkpoint, split, mnist_limit=mnist_limit)
    y_pred = model.predict(X)
    matrix = confusion_matrix(y, y_pred, num_classes=model.output_dim)
    class_acc = per_class_accuracy(matrix)
    stem = f"{checkpoint.stem}_{dataset}_{split}_confusion"

    csv_path = table_dir / f"{stem}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + [str(i) for i in range(matrix.shape[1])] + ["class_accuracy"])
        for idx, row in enumerate(matrix):
            writer.writerow([idx] + row.tolist() + [float(class_acc[idx])])

    png_path = figure_dir / f"{stem}.png"
    plt.figure(figsize=(6, 5))
    plt.imshow(matrix, cmap="Blues")
    plt.title(f"{dataset} {split} confusion matrix")
    plt.xlabel("predicted")
    plt.ylabel("true")
    plt.colorbar()
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()
    return csv_path, png_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot metrics and confusion matrices.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    curves = subparsers.add_parser("curves", help="Plot learning curves from a metrics JSON file.")
    curves.add_argument("--metrics", type=Path, required=True)
    curves.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "figures")

    confusion = subparsers.add_parser("confusion", help="Plot confusion matrix from a saved checkpoint.")
    confusion.add_argument("--model", type=Path, required=True)
    confusion.add_argument("--split", choices=["train", "val"], default="val")
    confusion.add_argument("--mnist-limit", type=int, default=None)
    confusion.add_argument("--figure-dir", type=Path, default=RESULTS_DIR / "figures")
    confusion.add_argument("--table-dir", type=Path, default=RESULTS_DIR / "tables")

    comparison = subparsers.add_parser("comparison", help="Plot paper-style metric comparison charts.")
    comparison.add_argument("--metrics", type=Path, nargs="+", required=True)
    comparison.add_argument("--dataset", choices=["iris", "mnist"], default=None)
    comparison.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "figures")
    comparison.add_argument("--output-prefix", default="model_comparison")
    comparison.add_argument("--title", default=None)

    report = subparsers.add_parser("report-figures", help="Generate narrative report figures.")
    report.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    report.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "figures")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "curves":
        outputs = plot_learning_curves(args.metrics, args.output_dir)
        for path in outputs:
            print(f"saved {path}")
    elif args.command == "confusion":
        metrics = evaluate_checkpoint(args.model, split=args.split, mnist_limit=args.mnist_limit)
        csv_path, png_path = save_confusion_outputs(
            args.model,
            split=args.split,
            figure_dir=args.figure_dir,
            table_dir=args.table_dir,
            mnist_limit=args.mnist_limit,
        )
        print(
            f"{metrics['dataset']} {metrics['split']}: "
            f"loss={metrics['loss']:.4f}, acc={metrics['accuracy']:.4f}, count={metrics['count']}"
        )
        print(f"saved {csv_path}")
        print(f"saved {png_path}")
    elif args.command == "comparison":
        outputs = plot_metric_comparison(
            metrics_paths=args.metrics,
            output_dir=args.output_dir,
            output_prefix=args.output_prefix,
            dataset=args.dataset,
            title=args.title,
        )
        for path in outputs:
            print(f"saved {path}")
    elif args.command == "report-figures":
        outputs = plot_report_figures(results_dir=args.results_dir, output_dir=args.output_dir)
        for path in outputs:
            print(f"saved {path}")


if __name__ == "__main__":
    main()
