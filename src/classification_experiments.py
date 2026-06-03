"""Ablation experiments for the iris and MNIST classifiers."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .classification_nn import MLPClassifier, SplitMeta, _encode_labels, save_metrics
    from .data_split import load_iris_split, load_mnist_split
except ImportError:  # pragma: no cover - direct script execution.
    from classification_nn import MLPClassifier, SplitMeta, _encode_labels, save_metrics
    from data_split import load_iris_split, load_mnist_split


ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for one reproducible ablation run."""

    name: str
    dataset: str
    model_type: str
    description: str
    ablates: str
    epochs: int
    learning_rate: float
    batch_size: int
    seed: int = 42
    hidden_dim: int | None = None
    train_limit: int | None = None
    val_limit: int | None = None
    scale_inputs: bool = True
    init_strategy: str = "he"
    l2: float = 0.0
    optimizer: str = "sgd"
    momentum: float = 0.9
    lr_decay: float = 0.0
    early_stopping_patience: int | None = None


class SoftmaxRegressionClassifier:
    """Single linear layer plus softmax, trained with mini-batch SGD."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        learning_rate: float = 0.1,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(0.0, 0.01, (input_dim, output_dim))
        self.b = np.zeros(output_dim, dtype=np.float64)

    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        return exp / exp.sum(axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = np.asarray(X, dtype=np.float64) @ self.W + self.b
        return self._softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        probs = self.predict_proba(X)
        n = y.shape[0]
        return float(-np.log(probs[np.arange(n), y] + 1e-12).mean())

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def _train_batch(self, X: np.ndarray, y: np.ndarray) -> None:
        n = X.shape[0]
        probs = self.predict_proba(X)
        probs[np.arange(n), y] -= 1.0
        probs /= n
        self.W -= self.learning_rate * (X.T @ probs)
        self.b -= self.learning_rate * probs.sum(axis=0)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        val_data: tuple[np.ndarray, np.ndarray] | None = None,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        history: dict[str, list[float]] = {"loss": [], "accuracy": []}
        if val_data is not None:
            history["val_loss"] = []
            history["val_accuracy"] = []

        def record_metrics() -> None:
            history["loss"].append(self.loss(X, y))
            history["accuracy"].append(self.accuracy(X, y))
            if val_data is not None:
                X_val, y_val = val_data
                history["val_loss"].append(self.loss(X_val, y_val))
                history["val_accuracy"].append(self.accuracy(X_val, y_val))

        record_metrics()
        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(X.shape[0])
            for start in range(0, X.shape[0], batch_size):
                idx = order[start : start + batch_size]
                self._train_batch(X[idx], y[idx])
            record_metrics()
            if verbose:
                print(
                    f"epoch {epoch:03d} "
                    f"loss={history['loss'][-1]:.4f} "
                    f"acc={history['accuracy'][-1]:.4f}"
                )
        return history


class SigmoidMSEMLPClassifier:
    """One-hidden-layer MLP with sigmoid outputs and MSE loss."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        learning_rate: float = 0.05,
        seed: int = 42,
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

        self.W1 = self.rng.normal(0.0, np.sqrt(2.0 / input_dim), (input_dim, hidden_dim))
        self.b1 = np.zeros(hidden_dim, dtype=np.float64)
        self.W2 = self.rng.normal(0.0, np.sqrt(2.0 / hidden_dim), (hidden_dim, output_dim))
        self.b2 = np.zeros(output_dim, dtype=np.float64)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        clipped = np.clip(z, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z1 = X @ self.W1 + self.b1
        hidden = np.maximum(z1, 0.0)
        z2 = hidden @ self.W2 + self.b2
        outputs = self._sigmoid(z2)
        return z1, hidden, z2, outputs

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        _, _, _, outputs = self._forward(np.asarray(X, dtype=np.float64))
        sums = outputs.sum(axis=1, keepdims=True)
        return np.divide(outputs, sums, out=np.full_like(outputs, 1.0 / self.output_dim), where=sums != 0.0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)

    def _one_hot(self, y: np.ndarray) -> np.ndarray:
        encoded = np.zeros((y.shape[0], self.output_dim), dtype=np.float64)
        encoded[np.arange(y.shape[0]), y] = 1.0
        return encoded

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        _, _, _, outputs = self._forward(np.asarray(X, dtype=np.float64))
        target = self._one_hot(np.asarray(y, dtype=np.int64))
        return float(0.5 * np.mean(np.sum((outputs - target) ** 2, axis=1)))

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))

    def _train_batch(self, X: np.ndarray, y: np.ndarray) -> None:
        n = X.shape[0]
        z1, hidden, _, outputs = self._forward(X)
        target = self._one_hot(y)

        doutputs = (outputs - target) / n
        dz2 = doutputs * outputs * (1.0 - outputs)
        dW2 = hidden.T @ dz2
        db2 = dz2.sum(axis=0)
        dhidden = dz2 @ self.W2.T
        dz1 = dhidden * (z1 > 0.0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(axis=0)

        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int,
        batch_size: int,
        val_data: tuple[np.ndarray, np.ndarray] | None = None,
        verbose: bool = False,
    ) -> dict[str, list[float]]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        history: dict[str, list[float]] = {"loss": [], "accuracy": []}
        if val_data is not None:
            history["val_loss"] = []
            history["val_accuracy"] = []

        def record_metrics() -> None:
            history["loss"].append(self.loss(X, y))
            history["accuracy"].append(self.accuracy(X, y))
            if val_data is not None:
                X_val, y_val = val_data
                history["val_loss"].append(self.loss(X_val, y_val))
                history["val_accuracy"].append(self.accuracy(X_val, y_val))

        record_metrics()
        for epoch in range(1, epochs + 1):
            order = self.rng.permutation(X.shape[0])
            for start in range(0, X.shape[0], batch_size):
                idx = order[start : start + batch_size]
                self._train_batch(X[idx], y[idx])
            record_metrics()
            if verbose:
                print(
                    f"epoch {epoch:03d} "
                    f"loss={history['loss'][-1]:.4f} "
                    f"acc={history['accuracy'][-1]:.4f}"
                )
        return history


def _label_meta(labels: np.ndarray) -> SplitMeta:
    label_to_id = {str(label): i for i, label in enumerate(sorted(set(labels)))}
    return SplitMeta(mean=None, std=None, label_to_id=label_to_id)


def load_experiment_split(
    dataset: str,
    which: str,
    config: ExperimentConfig,
    meta: SplitMeta | None = None,
) -> tuple[np.ndarray, np.ndarray, SplitMeta]:
    """Load one split with the preprocessing requested by an experiment."""
    limit = config.train_limit if which == "train" else config.val_limit
    if dataset == "iris":
        X, labels = load_iris_split(which)
        if limit is not None:
            X = X[:limit]
            labels = labels[:limit]
        X = X.astype(np.float64)
        if meta is None:
            meta = _label_meta(labels)
            if config.scale_inputs:
                mean = X.mean(axis=0)
                std = X.std(axis=0)
                std[std == 0.0] = 1.0
                meta = SplitMeta(mean=mean, std=std, label_to_id=meta.label_to_id)
        if config.scale_inputs:
            assert meta.mean is not None and meta.std is not None
            X = (X - meta.mean) / meta.std
        return X, _encode_labels(labels, meta.label_to_id), meta

    if dataset == "mnist":
        images, labels = load_mnist_split(which)
        if limit is not None:
            images = images[:limit]
            labels = labels[:limit]
        X = images.reshape(images.shape[0], -1).astype(np.float64)
        if config.scale_inputs:
            X /= 255.0
        if meta is None:
            meta = SplitMeta(mean=None, std=None, label_to_id={str(i): i for i in range(10)})
        return X, _encode_labels(labels, meta.label_to_id), meta

    raise ValueError("dataset must be 'iris' or 'mnist'")


def _make_model(config: ExperimentConfig, input_dim: int, output_dim: int):
    if config.model_type == "linear":
        return SoftmaxRegressionClassifier(
            input_dim=input_dim,
            output_dim=output_dim,
            learning_rate=config.learning_rate,
            seed=config.seed,
        )
    if config.model_type == "mlp":
        if config.hidden_dim is None:
            raise ValueError("hidden_dim is required for MLP experiments")
        model = MLPClassifier(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=output_dim,
            learning_rate=config.learning_rate,
            l2=config.l2,
            optimizer=config.optimizer,
            momentum=config.momentum,
            lr_decay=config.lr_decay,
            early_stopping_patience=config.early_stopping_patience,
            seed=config.seed,
        )
        if config.init_strategy == "small_normal":
            rng = np.random.default_rng(config.seed)
            model.W1 = rng.normal(0.0, 0.01, (input_dim, config.hidden_dim))
            model.W2 = rng.normal(0.0, 0.01, (config.hidden_dim, output_dim))
        elif config.init_strategy != "he":
            raise ValueError("init_strategy must be 'he' or 'small_normal'")
        return model
    if config.model_type == "mlp_mse":
        if config.hidden_dim is None:
            raise ValueError("hidden_dim is required for MLP MSE experiments")
        return SigmoidMSEMLPClassifier(
            input_dim=input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=output_dim,
            learning_rate=config.learning_rate,
            seed=config.seed,
        )
    raise ValueError("model_type must be 'linear', 'mlp', or 'mlp_mse'")


def run_experiment(config: ExperimentConfig, verbose: bool = False) -> dict[str, Any]:
    X_train, y_train, meta = load_experiment_split(config.dataset, "train", config)
    X_val, y_val, _ = load_experiment_split(config.dataset, "val", config, meta=meta)
    batch_size = X_train.shape[0] if config.batch_size <= 0 else config.batch_size
    output_dim = int(np.max(y_train)) + 1
    model = _make_model(config, input_dim=X_train.shape[1], output_dim=output_dim)
    history = model.fit(
        X_train,
        y_train,
        epochs=config.epochs,
        batch_size=batch_size,
        val_data=(X_val, y_val),
        verbose=verbose,
    )
    epochs_trained = len(history["loss"]) - 1
    return {
        "name": config.name,
        "dataset": config.dataset,
        "description": config.description,
        "ablates": config.ablates,
        "config": asdict(config) | {"effective_batch_size": batch_size},
        "epochs_trained": epochs_trained,
        "stopped_early": epochs_trained < config.epochs,
        "initial_train_accuracy": history["accuracy"][0],
        "initial_val_accuracy": history["val_accuracy"][0],
        "final_train_accuracy": history["accuracy"][-1],
        "final_val_accuracy": history["val_accuracy"][-1],
        "final_train_loss": history["loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "history": history,
    }


def build_experiment_configs(dataset: str, profile: str = "quick") -> list[ExperimentConfig]:
    if dataset == "mnist":
        train_limit = 1000 if profile == "quick" else None
        val_limit = 1000 if profile == "quick" else None
        epochs = 5
        return [
            ExperimentConfig(
                name="mnist_untrained_mlp",
                dataset="mnist",
                model_type="mlp",
                description="Current MLP architecture before any gradient update.",
                ablates="all optimization steps after initialization",
                epochs=0,
                hidden_dim=128,
                learning_rate=0.1,
                batch_size=128,
                train_limit=train_limit,
                val_limit=val_limit,
            ),
            ExperimentConfig(
                name="mnist_linear_softmax",
                dataset="mnist",
                model_type="linear",
                description="Simplest trainable neural classifier: xW+b followed by softmax.",
                ablates="hidden ReLU layer and hidden width",
                epochs=epochs,
                learning_rate=0.1,
                batch_size=128,
                train_limit=train_limit,
                val_limit=val_limit,
            ),
            ExperimentConfig(
                name="mnist_small_hidden",
                dataset="mnist",
                model_type="mlp",
                description="One hidden ReLU layer, but only 16 hidden units.",
                ablates="wide 128-unit hidden representation",
                epochs=epochs,
                hidden_dim=16,
                learning_rate=0.1,
                batch_size=128,
                train_limit=train_limit,
                val_limit=val_limit,
            ),
            ExperimentConfig(
                name="mnist_slow_learning_rate",
                dataset="mnist",
                model_type="mlp",
                description="Current width and mini-batches with a conservative learning rate.",
                ablates="aggressive learning rate 0.1",
                epochs=epochs,
                hidden_dim=128,
                learning_rate=0.01,
                batch_size=128,
                train_limit=train_limit,
                val_limit=val_limit,
            ),
            ExperimentConfig(
                name="mnist_full_batch",
                dataset="mnist",
                model_type="mlp",
                description="Current MLP trained with one full-batch update per epoch.",
                ablates="mini-batch SGD update frequency",
                epochs=epochs,
                hidden_dim=128,
                learning_rate=0.1,
                batch_size=0,
                train_limit=train_limit,
                val_limit=val_limit,
            ),
            ExperimentConfig(
                name="mnist_small_normal_init",
                dataset="mnist",
                model_type="mlp",
                description="Current MLP, but ReLU weights use tiny N(0, 0.01) initialization.",
                ablates="He initialization for ReLU layers",
                epochs=epochs,
                hidden_dim=128,
                learning_rate=0.1,
                batch_size=128,
                train_limit=train_limit,
                val_limit=val_limit,
                init_strategy="small_normal",
            ),
            ExperimentConfig(
                name="mnist_no_pixel_scaling",
                dataset="mnist",
                model_type="mlp",
                description="Current MLP, but raw 0-255 pixels are fed to SGD.",
                ablates="pixel scaling to [0, 1]",
                epochs=1,
                hidden_dim=128,
                learning_rate=0.1,
                batch_size=128,
                train_limit=train_limit,
                val_limit=val_limit,
                scale_inputs=False,
            ),
            ExperimentConfig(
                name="mnist_l2_regularization",
                dataset="mnist",
                model_type="mlp",
                description="Current MLP with L2 weight penalty added to the cross-entropy objective.",
                ablates="unregularized objective",
                epochs=epochs,
                hidden_dim=128,
                learning_rate=0.1,
                batch_size=128,
                train_limit=train_limit,
                val_limit=val_limit,
                l2=1e-4,
            ),
            ExperimentConfig(
                name="mnist_current_mlp",
                dataset="mnist",
                model_type="mlp",
                description="Current implementation: scaled pixels, 128 ReLU units, mini-batch SGD.",
                ablates="none",
                epochs=epochs,
                hidden_dim=128,
                learning_rate=0.1,
                batch_size=128,
                train_limit=train_limit,
                val_limit=val_limit,
            ),
        ]

    if dataset == "iris":
        epochs = 100 if profile == "quick" else 300
        return [
            ExperimentConfig(
                name="iris_untrained_mlp",
                dataset="iris",
                model_type="mlp",
                description="Current iris MLP before any gradient update.",
                ablates="all optimization steps after initialization",
                epochs=0,
                hidden_dim=16,
                learning_rate=0.05,
                batch_size=16,
            ),
            ExperimentConfig(
                name="iris_linear_softmax",
                dataset="iris",
                model_type="linear",
                description="Simplest trainable neural classifier: xW+b followed by softmax.",
                ablates="hidden ReLU layer",
                epochs=epochs,
                learning_rate=0.05,
                batch_size=16,
            ),
            ExperimentConfig(
                name="iris_small_hidden",
                dataset="iris",
                model_type="mlp",
                description="One hidden ReLU layer with 4 hidden units.",
                ablates="16-unit hidden representation",
                epochs=epochs,
                hidden_dim=4,
                learning_rate=0.05,
                batch_size=16,
            ),
            ExperimentConfig(
                name="iris_no_standardization",
                dataset="iris",
                model_type="mlp",
                description="Current iris MLP, but raw measurements are not standardized.",
                ablates="feature standardization",
                epochs=epochs,
                hidden_dim=16,
                learning_rate=0.05,
                batch_size=16,
                scale_inputs=False,
            ),
            ExperimentConfig(
                name="iris_small_normal_init",
                dataset="iris",
                model_type="mlp",
                description="Current iris MLP, but ReLU weights use tiny N(0, 0.01) initialization.",
                ablates="He initialization for ReLU layers",
                epochs=epochs,
                hidden_dim=16,
                learning_rate=0.05,
                batch_size=16,
                init_strategy="small_normal",
            ),
            ExperimentConfig(
                name="iris_l2_regularization",
                dataset="iris",
                model_type="mlp",
                description="Current iris MLP with L2 weight penalty added to the cross-entropy objective.",
                ablates="unregularized objective",
                epochs=epochs,
                hidden_dim=16,
                learning_rate=0.05,
                batch_size=16,
                l2=1e-3,
            ),
            ExperimentConfig(
                name="iris_current_mlp",
                dataset="iris",
                model_type="mlp",
                description="Current implementation: standardized features, 16 ReLU units.",
                ablates="none",
                epochs=epochs,
                hidden_dim=16,
                learning_rate=0.05,
                batch_size=16,
            ),
        ]

    raise ValueError("dataset must be 'iris' or 'mnist'")


def build_l2_sweep_configs(dataset: str, profile: str = "quick") -> list[ExperimentConfig]:
    """Build a focused sweep over L2 regularization strengths."""
    if dataset == "mnist":
        train_limit = 1000 if profile == "quick" else None
        val_limit = 1000 if profile == "quick" else None
        return [
            ExperimentConfig(
                name=f"mnist_l2_{l2:g}",
                dataset="mnist",
                model_type="mlp",
                description=f"Current MNIST MLP with L2 regularization strength {l2:g}.",
                ablates="L2 regularization strength",
                epochs=5,
                hidden_dim=128,
                learning_rate=0.1,
                batch_size=128,
                train_limit=train_limit,
                val_limit=val_limit,
                l2=l2,
            )
            for l2 in (0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
        ]

    if dataset == "iris":
        epochs = 100 if profile == "quick" else 300
        return [
            ExperimentConfig(
                name=f"iris_l2_{l2:g}",
                dataset="iris",
                model_type="mlp",
                description=f"Current iris MLP with L2 regularization strength {l2:g}.",
                ablates="L2 regularization strength",
                epochs=epochs,
                hidden_dim=16,
                learning_rate=0.05,
                batch_size=16,
                l2=l2,
            )
            for l2 in (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
        ]

    raise ValueError("dataset must be 'iris' or 'mnist'")


def build_learning_rate_sweep_configs(dataset: str, profile: str = "quick") -> list[ExperimentConfig]:
    """Build a focused sweep over learning rates."""
    if dataset == "mnist":
        train_limit = 1000 if profile == "quick" else None
        val_limit = 1000 if profile == "quick" else None
        return [
            ExperimentConfig(
                name=f"mnist_lr_{learning_rate:g}",
                dataset="mnist",
                model_type="mlp",
                description=f"Current MNIST MLP with learning rate {learning_rate:g}.",
                ablates="learning rate",
                epochs=5,
                hidden_dim=128,
                learning_rate=learning_rate,
                batch_size=128,
                train_limit=train_limit,
                val_limit=val_limit,
            )
            for learning_rate in (0.001, 0.01, 0.1, 0.2)
        ]

    if dataset == "iris":
        epochs = 100 if profile == "quick" else 300
        return [
            ExperimentConfig(
                name=f"iris_lr_{learning_rate:g}",
                dataset="iris",
                model_type="mlp",
                description=f"Current iris MLP with learning rate {learning_rate:g}.",
                ablates="learning rate",
                epochs=epochs,
                hidden_dim=16,
                learning_rate=learning_rate,
                batch_size=16,
            )
            for learning_rate in (0.001, 0.01, 0.05, 0.1)
        ]

    raise ValueError("dataset must be 'iris' or 'mnist'")


def build_width_sweep_configs(dataset: str, profile: str = "quick") -> list[ExperimentConfig]:
    """Build a focused sweep over one-hidden-layer width."""
    if dataset == "mnist":
        train_limit = 1000 if profile == "quick" else None
        val_limit = 1000 if profile == "quick" else None
        return [
            ExperimentConfig(
                name=f"mnist_width_{hidden_dim}",
                dataset="mnist",
                model_type="mlp",
                description=f"MNIST MLP with {hidden_dim} hidden ReLU units.",
                ablates="hidden layer width",
                epochs=5,
                hidden_dim=hidden_dim,
                learning_rate=0.1,
                batch_size=128,
                train_limit=train_limit,
                val_limit=val_limit,
            )
            for hidden_dim in (32, 64, 128, 256)
        ]

    if dataset == "iris":
        epochs = 100 if profile == "quick" else 300
        return [
            ExperimentConfig(
                name=f"iris_width_{hidden_dim}",
                dataset="iris",
                model_type="mlp",
                description=f"Iris MLP with {hidden_dim} hidden ReLU units.",
                ablates="hidden layer width",
                epochs=epochs,
                hidden_dim=hidden_dim,
                learning_rate=0.05,
                batch_size=16,
            )
            for hidden_dim in (4, 8, 16, 32)
        ]

    raise ValueError("dataset must be 'iris' or 'mnist'")


def build_training_strategy_configs(dataset: str, profile: str = "quick") -> list[ExperimentConfig]:
    """Build a compact comparison of training strategies."""
    if dataset == "mnist":
        train_limit = 1000 if profile == "quick" else None
        val_limit = 1000 if profile == "quick" else None
        epochs = 5 if profile == "quick" else 100
        return [
            ExperimentConfig(
                name="mnist_strategy_sgd",
                dataset="mnist",
                model_type="mlp",
                description="Current MNIST MLP with plain mini-batch SGD.",
                ablates="training strategy",
                epochs=epochs,
                hidden_dim=128,
                learning_rate=0.1,
                batch_size=128,
                train_limit=train_limit,
                val_limit=val_limit,
            ),
            ExperimentConfig(
                name="mnist_strategy_momentum",
                dataset="mnist",
                model_type="mlp",
                description="Current MNIST MLP trained with momentum SGD.",
                ablates="plain SGD",
                epochs=epochs,
                hidden_dim=128,
                learning_rate=0.05,
                batch_size=128,
                train_limit=train_limit,
                val_limit=val_limit,
                optimizer="momentum",
                momentum=0.9,
            ),
            ExperimentConfig(
                name="mnist_strategy_lr_decay",
                dataset="mnist",
                model_type="mlp",
                description="Current MNIST MLP with inverse learning-rate decay.",
                ablates="constant learning rate",
                epochs=epochs,
                hidden_dim=128,
                learning_rate=0.1,
                batch_size=128,
                train_limit=train_limit,
                val_limit=val_limit,
                lr_decay=0.2,
            ),
            ExperimentConfig(
                name="mnist_strategy_early_stopping",
                dataset="mnist",
                model_type="mlp",
                description="Current MNIST MLP with validation-loss early stopping.",
                ablates="fixed epoch count",
                epochs=20 if profile == "quick" else 100,
                hidden_dim=128,
                learning_rate=0.1,
                batch_size=128,
                train_limit=train_limit,
                val_limit=val_limit,
                early_stopping_patience=3,
            ),
        ]

    if dataset == "iris":
        return [
            ExperimentConfig(
                name="iris_strategy_sgd",
                dataset="iris",
                model_type="mlp",
                description="Current iris MLP with plain mini-batch SGD.",
                ablates="training strategy",
                epochs=100,
                hidden_dim=16,
                learning_rate=0.05,
                batch_size=16,
            ),
            ExperimentConfig(
                name="iris_strategy_momentum",
                dataset="iris",
                model_type="mlp",
                description="Current iris MLP trained with momentum SGD.",
                ablates="plain SGD",
                epochs=100,
                hidden_dim=16,
                learning_rate=0.02,
                batch_size=16,
                optimizer="momentum",
                momentum=0.9,
            ),
            ExperimentConfig(
                name="iris_strategy_lr_decay",
                dataset="iris",
                model_type="mlp",
                description="Current iris MLP with inverse learning-rate decay.",
                ablates="constant learning rate",
                epochs=100,
                hidden_dim=16,
                learning_rate=0.05,
                batch_size=16,
                lr_decay=0.05,
            ),
            ExperimentConfig(
                name="iris_strategy_early_stopping",
                dataset="iris",
                model_type="mlp",
                description="Current iris MLP with validation-loss early stopping.",
                ablates="fixed epoch count",
                epochs=300,
                hidden_dim=16,
                learning_rate=0.05,
                batch_size=16,
                early_stopping_patience=10,
            ),
        ]

    raise ValueError("dataset must be 'iris' or 'mnist'")


def build_loss_comparison_configs(dataset: str, profile: str = "quick") -> list[ExperimentConfig]:
    """Build a comparison between sigmoid+MSE and softmax+cross-entropy."""
    if dataset == "mnist":
        train_limit = 1000 if profile == "quick" else None
        val_limit = 1000 if profile == "quick" else None
        return [
            ExperimentConfig(
                name="mnist_sigmoid_mse",
                dataset="mnist",
                model_type="mlp_mse",
                description="MNIST MLP with sigmoid output units and MSE loss.",
                ablates="softmax cross-entropy loss",
                epochs=5,
                hidden_dim=128,
                learning_rate=0.1,
                batch_size=128,
                train_limit=train_limit,
                val_limit=val_limit,
            ),
            ExperimentConfig(
                name="mnist_softmax_cross_entropy",
                dataset="mnist",
                model_type="mlp",
                description="MNIST MLP with softmax output and cross-entropy loss.",
                ablates="sigmoid MSE loss",
                epochs=5,
                hidden_dim=128,
                learning_rate=0.1,
                batch_size=128,
                train_limit=train_limit,
                val_limit=val_limit,
            ),
        ]

    if dataset == "iris":
        epochs = 100 if profile == "quick" else 300
        return [
            ExperimentConfig(
                name="iris_sigmoid_mse",
                dataset="iris",
                model_type="mlp_mse",
                description="Iris MLP with sigmoid output units and MSE loss.",
                ablates="softmax cross-entropy loss",
                epochs=epochs,
                hidden_dim=16,
                learning_rate=0.05,
                batch_size=16,
            ),
            ExperimentConfig(
                name="iris_softmax_cross_entropy",
                dataset="iris",
                model_type="mlp",
                description="Iris MLP with softmax output and cross-entropy loss.",
                ablates="sigmoid MSE loss",
                epochs=epochs,
                hidden_dim=16,
                learning_rate=0.05,
                batch_size=16,
            ),
        ]

    raise ValueError("dataset must be 'iris' or 'mnist'")


def format_markdown_table(results: list[dict[str, Any]]) -> str:
    header = (
        "| Experiment | Ablation | Max epochs | Trained epochs | LR | L2 | Optimizer | Batch | "
        "Initial val acc | Final val acc | Final train acc |\n"
        "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|"
    )
    rows = []
    for result in results:
        config = result["config"]
        rows.append(
            "| {name} | {ablates} | {epochs} | {trained} | {lr:g} | {l2:g} | {optimizer} | {batch} | "
            "{iva:.4f} | {fva:.4f} | {fta:.4f} |".format(
                name=result["name"],
                ablates=result["ablates"],
                epochs=config["epochs"],
                trained=result["epochs_trained"],
                lr=config["learning_rate"],
                l2=config["l2"],
                optimizer=config["optimizer"],
                batch=config["effective_batch_size"],
                iva=result["initial_val_accuracy"],
                fva=result["final_val_accuracy"],
                fta=result["final_train_accuracy"],
            )
        )
    return header + "\n" + "\n".join(rows)


def run_suite(
    dataset: str,
    profile: str,
    suite: str = "ablation",
    verbose: bool = False,
) -> list[dict[str, Any]]:
    if suite == "ablation":
        configs = build_experiment_configs(dataset, profile=profile)
    elif suite == "l2-sweep":
        configs = build_l2_sweep_configs(dataset, profile=profile)
    elif suite == "lr-sweep":
        configs = build_learning_rate_sweep_configs(dataset, profile=profile)
    elif suite == "width-sweep":
        configs = build_width_sweep_configs(dataset, profile=profile)
    elif suite == "training-strategies":
        configs = build_training_strategy_configs(dataset, profile=profile)
    elif suite == "loss-comparison":
        configs = build_loss_comparison_configs(dataset, profile=profile)
    else:
        raise ValueError("unknown experiment suite")

    return [
        run_experiment(config, verbose=verbose)
        for config in configs
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run classifier ablation experiments.")
    parser.add_argument("--dataset", choices=["iris", "mnist", "all"], default="all")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument(
        "--suite",
        choices=[
            "ablation",
            "l2-sweep",
            "lr-sweep",
            "width-sweep",
            "training-strategies",
            "loss-comparison",
        ],
        default="ablation",
        help="Experiment suite to run.",
    )
    parser.add_argument("--output", type=Path, default=RESULTS_DIR / "classification_ablation.json")
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Optional Markdown table output path.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    datasets = ["iris", "mnist"] if args.dataset == "all" else [args.dataset]
    results: list[dict[str, Any]] = []
    for dataset in datasets:
        results.extend(run_suite(dataset, profile=args.profile, suite=args.suite, verbose=args.verbose))

    save_metrics(results, args.output)
    print(f"results saved to {args.output}")

    table = format_markdown_table(results)
    print(table)
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(table + "\n", encoding="utf-8")
        print(f"markdown table saved to {args.markdown_output}")


if __name__ == "__main__":
    main()
