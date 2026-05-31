"""Simple neural-network classifiers for iris and MNIST.

The implementation intentionally uses only NumPy so the training logic stays
visible for an optimization-theory project.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:  # Support both `python src/classification_nn.py` and package imports.
    from .data_split import load_iris_split, load_mnist_split, load_mnist_test
except ImportError:  # pragma: no cover - exercised by direct script execution.
    from data_split import load_iris_split, load_mnist_split, load_mnist_test


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


@dataclass
class SplitMeta:
    """Preprocessing metadata shared by train and validation splits."""

    mean: np.ndarray | None
    std: np.ndarray | None
    label_to_id: dict[str, int]


class MLPClassifier:
    """A one-hidden-layer MLP classifier trained with mini-batch SGD."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        learning_rate: float = 0.05,
        l2: float = 0.0,
        optimizer: str = "sgd",
        momentum: float = 0.9,
        lr_decay: float = 0.0,
        early_stopping_patience: int | None = None,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.l2 = l2
        self.optimizer = optimizer
        self.momentum = momentum
        self.lr_decay = lr_decay
        self.early_stopping_patience = early_stopping_patience
        self.rng = np.random.default_rng(seed)

        self.W1 = self.rng.normal(0.0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W2 = self.rng.normal(0.0, np.sqrt(2.0 / hidden_dim), (hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim, dtype=np.float64)
        self._velocity = {
            "W1": np.zeros_like(self.W1),
            "b1": np.zeros_like(self.b1),
            "W2": np.zeros_like(self.W2),
            "b2": np.zeros_like(self.b2),
        }

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = X @ self.W1 + self.b1
        hidden = np.maximum(z1, 0.0)
        logits = hidden @ self.W2 + self.b2
        return z1, hidden, logits

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, _, logits = self._forward(np.asarray(X, dtype=np.float64))
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        probs = self.predict_proba(X)
        n = y.shape[0]
        ce = -np.log(probs[np.arange(n), y] + 1e-12).mean()
        penalty = 0.5 * self.l2 * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2))
        return float(ce + penalty)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def _train_batch(self, X: np.ndarray, y: np.ndarray, learning_rate: float) -> None:
        n = X.shape[0]
        z1, hidden, logits = self._forward(X)
        probs = self._softmax(logits)
        probs[np.arange(n), y] -= 1.0
        probs /= n

        dW2 = hidden.T @ probs + self.l2 * self.W2
        db2 = probs.sum(axis=0)
        dhidden = probs @ self.W2.T
        dz1 = dhidden * (z1 > 0.0)
        dW1 = X.T @ dz1 + self.l2 * self.W1
        db1 = dz1.sum(axis=0)

        gradients = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}
        if self.optimizer == "sgd":
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2
        elif self.optimizer == "momentum":
            for name, gradient in gradients.items():
                self._velocity[name] = self.momentum * self._velocity[name] + gradient
            self.W1 -= learning_rate * self._velocity["W1"]
            self.b1 -= learning_rate * self._velocity["b1"]
            self.W2 -= learning_rate * self._velocity["W2"]
            self.b2 -= learning_rate * self._velocity["b2"]
        else:
            raise ValueError("optimizer must be 'sgd' or 'momentum'")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        batch_size: int = 32,
        val_data: tuple[np.ndarray, np.ndarray] | None = None,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        history: dict[str, list[float]] = {"loss": [], "accuracy": [], "learning_rate": []}
        if val_data is not None:
            history["val_loss"] = []
            history["val_accuracy"] = []

        def record_metrics(current_lr: float) -> None:
            history["loss"].append(self.loss(X, y))
            history["accuracy"].append(self.accuracy(X, y))
            history["learning_rate"].append(current_lr)
            if val_data is not None:
                X_val, y_val = val_data
                history["val_loss"].append(self.loss(X_val, y_val))
                history["val_accuracy"].append(self.accuracy(X_val, y_val))

        def print_latest_metrics(epoch: int) -> None:
            message = (
                f"epoch {epoch:03d} "
                f"loss={history['loss'][-1]:.4f} "
                f"acc={history['accuracy'][-1]:.4f}"
            )
            if val_data is not None:
                message += (
                    f" val_loss={history['val_loss'][-1]:.4f} "
                    f"val_acc={history['val_accuracy'][-1]:.4f}"
                )
            print(message)

        record_metrics(current_lr=self.learning_rate)
        if verbose:
            print_latest_metrics(epoch=0)

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        for epoch in range(1, epochs + 1):
            current_lr = self.learning_rate / (1.0 + self.lr_decay * (epoch - 1))
            order = self.rng.permutation(X.shape[0])
            for start in range(0, X.shape[0], batch_size):
                idx = order[start : start + batch_size]
                self._train_batch(X[idx], y[idx], learning_rate=current_lr)

            record_metrics(current_lr=current_lr)
            if verbose:
                print_latest_metrics(epoch=epoch)
            if val_data is not None and self.early_stopping_patience is not None:
                val_loss = history["val_loss"][-1]
                if val_loss < best_val_loss - 1e-12:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                if epochs_without_improvement >= self.early_stopping_patience:
                    break

        return history


def _meta_to_json(meta: SplitMeta) -> dict[str, Any]:
    return {
        "mean": None if meta.mean is None else meta.mean.tolist(),
        "std": None if meta.std is None else meta.std.tolist(),
        "label_to_id": meta.label_to_id,
    }


def _meta_from_json(payload: dict[str, Any]) -> SplitMeta:
    mean = None if payload["mean"] is None else np.asarray(payload["mean"], dtype=np.float64)
    std = None if payload["std"] is None else np.asarray(payload["std"], dtype=np.float64)
    label_to_id = {str(label): int(idx) for label, idx in payload["label_to_id"].items()}
    return SplitMeta(mean=mean, std=std, label_to_id=label_to_id)


def _encode_labels(labels: np.ndarray, label_to_id: dict[str, int]) -> np.ndarray:
    return np.array([label_to_id[str(label)] for label in labels], dtype=np.int64)


def prepare_iris_split(which: str, meta: SplitMeta | None = None) -> tuple[np.ndarray, np.ndarray, SplitMeta]:
    """Load an iris split, standardize features, and encode labels as integers."""
    X, labels = load_iris_split(which)
    X = X.astype(np.float64)

    if meta is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0.0] = 1.0
        label_to_id = {label: i for i, label in enumerate(sorted(set(labels)))}
        meta = SplitMeta(mean=mean, std=std, label_to_id=label_to_id)

    assert meta.mean is not None and meta.std is not None
    X = (X - meta.mean) / meta.std
    y = _encode_labels(labels, meta.label_to_id)
    return X, y, meta


def prepare_mnist_split(
    which: str,
    limit: int | None = None,
    meta: SplitMeta | None = None,
) -> tuple[np.ndarray, np.ndarray, SplitMeta]:
    """Load an MNIST split, flatten images, scale pixels, and encode labels."""
    if which == "test":
        images, labels = load_mnist_test()
    else:
        images, labels = load_mnist_split(which)
    if limit is not None:
        images = images[:limit]
        labels = labels[:limit]

    X = images.reshape(images.shape[0], -1).astype(np.float64) / 255.0
    if meta is None:
        label_to_id = {str(i): i for i in range(10)}
        meta = SplitMeta(mean=None, std=None, label_to_id=label_to_id)
    y = _encode_labels(labels, meta.label_to_id)
    return X, y, meta


def train_dataset(
    dataset: str,
    epochs: int | None = None,
    hidden_dim: int | None = None,
    learning_rate: float | None = None,
    batch_size: int | None = None,
    l2: float = 0.0,
    optimizer: str = "sgd",
    momentum: float = 0.9,
    lr_decay: float = 0.0,
    early_stopping_patience: int | None = None,
    seed: int = 42,
    mnist_limit: int | None = None,
    verbose: bool = True,
    return_artifacts: bool = False,
) -> dict[str, Any]:
    """Train a classifier on one dataset and return metrics plus configuration."""
    if dataset == "iris":
        X_train, y_train, meta = prepare_iris_split("train")
        X_val, y_val, _ = prepare_iris_split("val", meta=meta)
        config = {
            "epochs": 300 if epochs is None else epochs,
            "hidden_dim": 16 if hidden_dim is None else hidden_dim,
            "learning_rate": 0.05 if learning_rate is None else learning_rate,
            "batch_size": 16 if batch_size is None else batch_size,
            "input_dim": 4,
            "output_dim": 3,
            "l2": l2,
            "optimizer": optimizer,
            "momentum": momentum,
            "lr_decay": lr_decay,
            "early_stopping_patience": early_stopping_patience,
        }
    elif dataset == "mnist":
        X_train, y_train, meta = prepare_mnist_split("train", limit=mnist_limit)
        val_limit = mnist_limit if mnist_limit is not None and mnist_limit < 12000 else None
        X_val, y_val, _ = prepare_mnist_split("val", limit=val_limit, meta=meta)
        config = {
            "epochs": 5 if epochs is None else epochs,
            "hidden_dim": 128 if hidden_dim is None else hidden_dim,
            "learning_rate": 0.1 if learning_rate is None else learning_rate,
            "batch_size": 128 if batch_size is None else batch_size,
            "input_dim": 784,
            "output_dim": 10,
            "l2": l2,
            "optimizer": optimizer,
            "momentum": momentum,
            "lr_decay": lr_decay,
            "early_stopping_patience": early_stopping_patience,
        }
    else:
        raise ValueError("dataset must be 'iris' or 'mnist'")

    model = MLPClassifier(
        input_dim=config["input_dim"],
        hidden_dim=config["hidden_dim"],
        output_dim=config["output_dim"],
        learning_rate=config["learning_rate"],
        l2=config["l2"],
        optimizer=config["optimizer"],
        momentum=config["momentum"],
        lr_decay=config["lr_decay"],
        early_stopping_patience=config["early_stopping_patience"],
        seed=seed,
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=config["epochs"],
        batch_size=config["batch_size"],
        val_data=(X_val, y_val),
        verbose=verbose,
    )
    epochs_trained = len(history["loss"]) - 1
    metrics: dict[str, Any] = {
        "dataset": dataset,
        "config": config | {"seed": seed, "mnist_limit": mnist_limit},
        "epochs_trained": epochs_trained,
        "stopped_early": epochs_trained < config["epochs"],
        "final_train_accuracy": history["accuracy"][-1],
        "final_val_accuracy": history["val_accuracy"][-1],
        "final_train_loss": history["loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "history": history,
    }
    if return_artifacts:
        metrics["model"] = model
        metrics["meta"] = meta
    return metrics


def save_checkpoint(
    model: MLPClassifier,
    dataset: str,
    meta: SplitMeta,
    config: dict[str, Any],
    path: Path,
) -> None:
    """Save model parameters plus preprocessing metadata to a NumPy checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_class": "MLPClassifier",
        "dataset": dataset,
        "config": config,
        "meta": _meta_to_json(meta),
    }
    np.savez_compressed(
        path,
        W1=model.W1,
        b1=model.b1,
        W2=model.W2,
        b2=model.b2,
        metadata=np.asarray(json.dumps(metadata, ensure_ascii=False)),
    )


def load_checkpoint(path: Path) -> tuple[MLPClassifier, str, SplitMeta, dict[str, Any]]:
    """Load a checkpoint saved by save_checkpoint()."""
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        dataset = metadata["dataset"]
        config = metadata["config"]
        model = MLPClassifier(
            input_dim=int(config["input_dim"]),
            hidden_dim=int(config["hidden_dim"]),
            output_dim=int(config["output_dim"]),
            learning_rate=float(config["learning_rate"]),
            l2=float(config.get("l2", 0.0)),
            optimizer=str(config.get("optimizer", "sgd")),
            momentum=float(config.get("momentum", 0.9)),
            lr_decay=float(config.get("lr_decay", 0.0)),
            early_stopping_patience=config.get("early_stopping_patience"),
            seed=int(config.get("seed", 42)),
        )
        model.W1 = data["W1"].copy()
        model.b1 = data["b1"].copy()
        model.W2 = data["W2"].copy()
        model.b2 = data["b2"].copy()
        meta = _meta_from_json(metadata["meta"])
    return model, dataset, meta, config


def evaluate_checkpoint(
    path: Path,
    split: str = "val",
    mnist_limit: int | None = None,
) -> dict[str, Any]:
    """Load a saved model and evaluate it on a train, validation, or test split."""
    model, dataset, meta, config = load_checkpoint(path)
    if dataset == "iris":
        if split == "test":
            raise ValueError("iris has no separate official test split; use train or val")
        X, y, _ = prepare_iris_split(split, meta=meta)
    elif dataset == "mnist":
        limit = mnist_limit if mnist_limit is not None else config.get("mnist_limit")
        X, y, _ = prepare_mnist_split(split, limit=limit, meta=meta)
    else:
        raise ValueError("checkpoint dataset must be 'iris' or 'mnist'")

    return {
        "dataset": dataset,
        "split": split,
        "model_path": str(path),
        "count": int(y.shape[0]),
        "loss": model.loss(X, y),
        "accuracy": model.accuracy(X, y),
        "config": config,
    }


def save_metrics(metrics: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train simple NumPy MLP classifiers.")
    parser.add_argument("--dataset", choices=["iris", "mnist", "all"], default="all")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--l2",
        type=float,
        default=0.0,
        help="L2 regularization strength for MLP weights.",
    )
    parser.add_argument("--optimizer", choices=["sgd", "momentum"], default="sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--lr-decay", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mnist-limit",
        type=int,
        default=None,
        help="Optional sample cap for quick MNIST smoke runs.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR / "classification_nn_metrics.json",
        help="Path for the metrics JSON file.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=None,
        help="Save a single trained dataset model checkpoint to this .npz path.",
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=None,
        help="Save trained checkpoints as <dataset>_mlp.npz in this directory.",
    )
    parser.add_argument(
        "--load-model",
        type=Path,
        default=None,
        help="Load a saved checkpoint and evaluate it instead of training.",
    )
    parser.add_argument(
        "--eval-split",
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split used with --load-model.",
    )
    parser.add_argument("--quiet", action="store_true", help="Hide epoch logs.")
    return parser


def _metrics_for_json(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in {"model", "meta"}}


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.load_model is not None:
        metrics = [evaluate_checkpoint(args.load_model, split=args.eval_split, mnist_limit=args.mnist_limit)]
        save_metrics(metrics, args.output)
        item = metrics[0]
        print(
            f"{item['dataset']} {item['split']}: "
            f"loss={item['loss']:.4f}, acc={item['accuracy']:.4f}, "
            f"count={item['count']}"
        )
        print(f"metrics saved to {args.output}")
        return

    datasets = ["iris", "mnist"] if args.dataset == "all" else [args.dataset]
    if args.model_output is not None and len(datasets) != 1:
        parser.error("--model-output can only be used when --dataset is iris or mnist")

    should_save_model = args.model_output is not None or args.models_dir is not None
    metrics = []
    for dataset in datasets:
        item = train_dataset(
            dataset=dataset,
            epochs=args.epochs,
            hidden_dim=args.hidden_dim,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            l2=args.l2,
            optimizer=args.optimizer,
            momentum=args.momentum,
            lr_decay=args.lr_decay,
            early_stopping_patience=args.early_stopping_patience,
            seed=args.seed,
            mnist_limit=args.mnist_limit,
            verbose=not args.quiet,
            return_artifacts=should_save_model,
        )
        if should_save_model:
            model_path = args.model_output or args.models_dir / f"{dataset}_mlp.npz"
            save_checkpoint(item["model"], dataset=dataset, meta=item["meta"], config=item["config"], path=model_path)
            item["model_path"] = str(model_path)
        metrics.append(_metrics_for_json(item))
    save_metrics(metrics, args.output)

    for item in metrics:
        print(
            f"{item['dataset']}: "
            f"train_acc={item['final_train_accuracy']:.4f}, "
            f"val_acc={item['final_val_accuracy']:.4f}"
        )
        if "model_path" in item:
            print(f"model saved to {item['model_path']}")
    print(f"metrics saved to {args.output}")


if __name__ == "__main__":
    main()
