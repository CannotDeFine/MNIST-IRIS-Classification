"""IRIS linear-classifier study: Perceptron vs softmax-linear vs MLP.

A self-contained experiment script for the optimization-theory project. It
implements a from-scratch multiclass Perceptron (error-driven updates), then
compares it against the existing softmax linear classifier (mini-batch SGD on
cross-entropy) and a one-hidden-layer MLP. The script produces three figures:

1. accuracy comparison (linear models vs nonlinear MLP),
2. decision boundaries on the petal feature plane (linear straight edges vs
   the MLP's bent boundary),
3. convergence dynamics (smooth cross-entropy descent vs the Perceptron's
   jumpy mistake count).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Headless: render to files only.
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    from .classification_experiments import SoftmaxRegressionClassifier
    from .classification_nn import MLPClassifier, prepare_iris_split
    from .plot_results import GRAY, GREEN, RED, _apply_paper_style, _style_axes
except ImportError:  # pragma: no cover - direct script execution.
    from classification_experiments import SoftmaxRegressionClassifier
    from classification_nn import MLPClassifier, prepare_iris_split
    from plot_results import GRAY, GREEN, RED, _apply_paper_style, _style_axes


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"
# IRIS feature order: sepal_length, sepal_width, petal_length, petal_width.
PETAL_FEATURES = (2, 3)
PETAL_LABELS = ("petal length (std.)", "petal width (std.)")


class MulticlassPerceptron:
    """Multiclass perceptron with error-driven online updates (zero init)."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_rate: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)
        self.W = np.zeros((output_dim, input_dim), dtype=np.float64)
        self.b = np.zeros(output_dim, dtype=np.float64)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X, dtype=np.float64) @ self.W.T + self.b

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.decision_function(X).argmax(axis=1)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def _mistakes(self, X: np.ndarray, y: np.ndarray) -> int:
        return int(np.sum(self.predict(X) != y))

    def _update_sample(self, x: np.ndarray, yi: int) -> bool:
        """Apply one perceptron update. Returns True if the sample was wrong."""
        scores = self.W @ x + self.b
        pred = int(np.argmax(scores))
        if pred == yi:
            return False
        self.W[yi] += self.learning_rate * x
        self.b[yi] += self.learning_rate
        self.W[pred] -= self.learning_rate * x
        self.b[pred] -= self.learning_rate
        return True

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        val_data: tuple[np.ndarray, np.ndarray] | None = None,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        history: dict[str, list[float]] = {"errors": [], "accuracy": []}
        if val_data is not None:
            history["val_accuracy"] = []

        def record() -> None:
            history["errors"].append(self._mistakes(X, y))
            history["accuracy"].append(self.accuracy(X, y))
            if val_data is not None:
                X_val, y_val = val_data
                history["val_accuracy"].append(self.accuracy(X_val, y_val))

        record()
        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(X.shape[0])
            for idx in order:
                self._update_sample(X[idx], int(y[idx]))
            record()
            if verbose:
                print(f"epoch {epoch:03d} errors={history['errors'][-1]} acc={history['accuracy'][-1]:.4f}")
        return history


def load_iris_features(
    feature_indices: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return standardized (X_train, y_train, X_val, y_val), optionally sliced."""
    X_train, y_train, meta = prepare_iris_split("train")
    X_val, y_val, _ = prepare_iris_split("val", meta=meta)
    if feature_indices is not None:
        cols = list(feature_indices)
        X_train = X_train[:, cols]
        X_val = X_val[:, cols]
    return X_train, y_train, X_val, y_val


def decision_boundary_grid(
    model: Any,
    X2d: np.ndarray,
    resolution: int = 200,
    margin: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict over a mesh spanning the 2D data, for contour plotting."""
    x_min, x_max = X2d[:, 0].min() - margin, X2d[:, 0].max() + margin
    y_min, y_max = X2d[:, 1].min() - margin, X2d[:, 1].max() + margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, resolution),
        np.linspace(y_min, y_max, resolution),
    )
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    Z = model.predict(grid).reshape(xx.shape)
    return xx, yy, Z


def _final_accuracies(model: Any, X_tr, y_tr, X_va, y_va) -> tuple[float, float]:
    return model.accuracy(X_tr, y_tr), model.accuracy(X_va, y_va)


def _build_models(seed: int, hidden_dim: int):
    perceptron = MulticlassPerceptron(input_dim=4, output_dim=3, learning_rate=1.0, seed=seed)
    softmax = SoftmaxRegressionClassifier(input_dim=4, output_dim=3, learning_rate=0.05, seed=seed)
    mlp = MLPClassifier(input_dim=4, hidden_dim=hidden_dim, output_dim=3, learning_rate=0.05, seed=seed)
    return perceptron, softmax, mlp


def plot_accuracy_comparison(results: list[dict[str, Any]], figure_dir: Path) -> Path:
    _apply_paper_style()
    labels = ["Perceptron", "Softmax\n(linear)", "MLP\n(hidden)"]
    train_acc = [r["final_train_accuracy"] for r in results]
    val_acc = [r["final_val_accuracy"] for r in results]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(3.6, 2.1))
    ax.bar(x - width / 2, train_acc, width, color=GRAY, edgecolor="black", linewidth=0.8, label="train")
    ax.bar(x + width / 2, val_acc, width, color=RED, edgecolor="black", linewidth=0.8, label="validation")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Accuracy", fontsize=8, fontweight="bold")
    ax.set_title("IRIS: Linear vs MLP", fontsize=8, fontweight="bold")
    for xi, (ta, va) in enumerate(zip(train_acc, val_acc, strict=True)):
        ax.text(xi - width / 2, ta + 0.01, f"{ta:.3f}", ha="center", va="bottom", fontsize=6, fontweight="bold")
        ax.text(xi + width / 2, va + 0.01, f"{va:.3f}", ha="center", va="bottom", fontsize=6, fontweight="bold")
    ax.legend(loc="lower right", fontsize=6.5, handlelength=1.1)
    _style_axes(ax)
    fig.tight_layout()
    path = figure_dir / "iris_linear_vs_mlp_accuracy.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_decision_boundaries(
    models_2d: list[tuple[str, Any]],
    X2d: np.ndarray,
    y: np.ndarray,
    figure_dir: Path,
) -> Path:
    _apply_paper_style()
    cmap_regions = plt.cm.Pastel1
    point_colors = np.array([RED, GREEN, GRAY])
    fig, axes = plt.subplots(1, len(models_2d), figsize=(2.7 * len(models_2d), 2.6), sharex=True, sharey=True)
    if len(models_2d) == 1:
        axes = [axes]
    for ax, (title, model) in zip(axes, models_2d, strict=True):
        xx, yy, Z = decision_boundary_grid(model, X2d, resolution=200)
        ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_regions, levels=np.arange(-0.5, 3.5, 1))
        for cls in range(3):
            mask = y == cls
            ax.scatter(
                X2d[mask, 0], X2d[mask, 1],
                s=16, c=point_colors[cls], edgecolors="black", linewidths=0.5, label=f"class {cls}",
            )
        ax.set_title(title, fontsize=8, fontweight="bold")
        ax.set_xlabel(PETAL_LABELS[0], fontsize=7, fontweight="bold")
        _style_axes(ax)
    axes[0].set_ylabel(PETAL_LABELS[1], fontsize=7, fontweight="bold")
    axes[-1].legend(loc="lower right", fontsize=5.8, handlelength=0.8, handletextpad=0.3)
    fig.suptitle("IRIS decision boundaries (petal plane)", fontsize=8, fontweight="bold")
    fig.tight_layout()
    path = figure_dir / "iris_linear_decision_boundary.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_convergence(
    softmax_history: dict[str, list[float]],
    perceptron_history: dict[str, list[float]],
    figure_dir: Path,
) -> Path:
    _apply_paper_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.6, 2.2))

    ax1.plot(softmax_history["loss"], color=RED, linewidth=1.5)
    ax1.set_title("Softmax: cross-entropy", fontsize=8, fontweight="bold")
    ax1.set_xlabel("epoch", fontsize=7, fontweight="bold")
    ax1.set_ylabel("train loss", fontsize=7, fontweight="bold")
    _style_axes(ax1)

    ax2.plot(perceptron_history["errors"], color=GREEN, linewidth=1.5)
    ax2.set_title("Perceptron: mistakes", fontsize=8, fontweight="bold")
    ax2.set_xlabel("epoch", fontsize=7, fontweight="bold")
    ax2.set_ylabel("misclassified (train)", fontsize=7, fontweight="bold")
    _style_axes(ax2)

    fig.suptitle("Optimization dynamics: gradient descent vs error-driven", fontsize=8, fontweight="bold")
    fig.tight_layout()
    path = figure_dir / "iris_linear_convergence.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def run_study(
    output: Path = RESULTS_DIR / "iris_linear_study.json",
    figure_dir: Path = RESULTS_DIR / "figures",
    seed: int = 42,
    epochs_perceptron: int = 50,
    epochs_linear: int = 100,
    epochs_mlp: int = 100,
    hidden_dim: int = 16,
    quiet: bool = False,
) -> list[Path]:
    """Run the full IRIS linear study and save the metrics JSON and figures."""
    output = Path(output)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    verbose = not quiet

    # --- 4-feature comparison ---
    X_tr, y_tr, X_va, y_va = load_iris_features()
    perceptron, softmax, mlp = _build_models(seed, hidden_dim)
    perc_hist = perceptron.fit(X_tr, y_tr, epochs=epochs_perceptron, val_data=(X_va, y_va), verbose=verbose)
    soft_hist = softmax.fit(X_tr, y_tr, epochs=epochs_linear, batch_size=16, val_data=(X_va, y_va), verbose=verbose)
    mlp_hist = mlp.fit(X_tr, y_tr, epochs=epochs_mlp, batch_size=16, val_data=(X_va, y_va), verbose=verbose)

    results = [
        _result_entry("iris_perceptron", "multiclass perceptron (error-driven)", perceptron, perc_hist,
                      X_tr, y_tr, X_va, y_va, epochs_perceptron,
                      {"lr": perceptron.learning_rate, "rule": "error-driven", "features": 4}),
        _result_entry("iris_linear_softmax", "linear softmax (SGD on cross-entropy)", softmax, soft_hist,
                      X_tr, y_tr, X_va, y_va, epochs_linear,
                      {"lr": 0.05, "rule": "sgd-cross-entropy", "batch_size": 16, "features": 4}),
        _result_entry("iris_mlp", "one-hidden-layer ReLU MLP", mlp, mlp_hist,
                      X_tr, y_tr, X_va, y_va, epochs_mlp,
                      {"lr": 0.05, "hidden_dim": hidden_dim, "batch_size": 16, "features": 4}),
    ]

    # --- 2-feature models for honest decision boundaries ---
    Xp_tr, yp_tr, _, _ = load_iris_features(feature_indices=PETAL_FEATURES)
    perc2, soft2, mlp2 = (
        MulticlassPerceptron(input_dim=2, output_dim=3, seed=seed),
        SoftmaxRegressionClassifier(input_dim=2, output_dim=3, learning_rate=0.05, seed=seed),
        MLPClassifier(input_dim=2, hidden_dim=hidden_dim, output_dim=3, learning_rate=0.05, seed=seed),
    )
    perc2.fit(Xp_tr, yp_tr, epochs=epochs_perceptron)
    soft2.fit(Xp_tr, yp_tr, epochs=epochs_linear, batch_size=16)
    mlp2.fit(Xp_tr, yp_tr, epochs=epochs_mlp, batch_size=16)

    # --- figures + JSON ---
    outputs = [
        plot_accuracy_comparison(results, figure_dir),
        plot_decision_boundaries([("Perceptron", perc2), ("Softmax", soft2), ("MLP", mlp2)], Xp_tr, yp_tr, figure_dir),
        plot_convergence(soft_hist, perc_hist, figure_dir),
    ]
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    if not quiet:
        for r in results:
            print(f"{r['name']}: train={r['final_train_accuracy']:.4f} val={r['final_val_accuracy']:.4f}")
        for path in outputs:
            print(f"saved {path}")
        print(f"saved {output}")
    return outputs + [output]


def _result_entry(name, description, model, history, X_tr, y_tr, X_va, y_va, epochs, config):
    train_acc, val_acc = _final_accuracies(model, X_tr, y_tr, X_va, y_va)
    return {
        "name": name,
        "dataset": "iris",
        "description": description,
        "config": {"epochs": epochs, **config},
        "history": history,
        "final_train_accuracy": train_acc,
        "final_val_accuracy": val_acc,
        "epochs_trained": epochs,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IRIS linear-classifier study (perceptron vs softmax vs MLP).")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "iris_linear_study.json")
    parser.add_argument("--figure-dir", type=Path, default=RESULTS_DIR / "figures")
    parser.add_argument("--epochs-perceptron", type=int, default=50)
    parser.add_argument("--epochs-linear", type=int, default=100)
    parser.add_argument("--epochs-mlp", type=int, default=100)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    run_study(
        output=args.output,
        figure_dir=args.figure_dir,
        seed=args.seed,
        epochs_perceptron=args.epochs_perceptron,
        epochs_linear=args.epochs_linear,
        epochs_mlp=args.epochs_mlp,
        hidden_dim=args.hidden_dim,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    main()
