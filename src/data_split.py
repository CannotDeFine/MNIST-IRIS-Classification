"""数据集划分脚本：对 iris 与 mnist 进行分层(stratified)训练/验证划分。

划分策略
--------
- 分层抽样：每个类别独立按比例抽取，保证训练集 / 验证集中各类别比例
  与原始数据一致。
- 固定随机种子 SEED，保证全队成员、多次运行得到完全一致的划分。
- 比例：训练集 80% / 验证集 20%。
- MNIST 仅划分原始的 60000 训练样本；10000 的 t10k 保留作测试集，不参与划分。

用法
----
    uv run python src/data_split.py

输出（写入 splits/ 目录）
------------------------
    iris_train.csv / iris_val.csv          —— iris 划分后的样本（含表头）
    mnist_train_indices.npy / mnist_val_indices.npy
                                           —— MNIST 划分索引（指向原始 60000 训练集）
    split_summary.json                     —— 划分结果统计
"""
from __future__ import annotations

import csv
import gzip
import json
import struct
from pathlib import Path

import numpy as np

# ---------------------------- 配置 ----------------------------
SEED = 42          # 固定随机种子，保证划分可复现、全队一致
VAL_RATIO = 0.20   # 验证集比例（训练集 = 1 - VAL_RATIO）

ROOT = Path(__file__).resolve().parent.parent
IRIS_FILE = ROOT / "iris" / "iris.data"
MNIST_DIR = ROOT / "mnist"
SPLITS_DIR = ROOT / "splits"

IRIS_COLUMNS = ["sepal_length", "sepal_width",
                "petal_length", "petal_width", "species"]


# ------------------------- 通用工具 -------------------------
def stratified_split(labels, val_ratio, rng):
    """按类别分层划分。

    每个类别独立打乱后按 val_ratio 抽取验证样本，最终训练 / 验证索引
    再整体打乱，得到可直接使用的混合顺序。返回 (train_idx, val_idx)。
    """
    labels = np.asarray(labels)
    train_parts, val_parts = [], []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)
        n_val = int(round(len(cls_idx) * val_ratio))
        val_parts.append(cls_idx[:n_val])
        train_parts.append(cls_idx[n_val:])
    train_idx = np.concatenate(train_parts)
    val_idx = np.concatenate(val_parts)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def class_counts(labels, idx):
    """统计给定索引子集中各类别的样本数，返回 {类别: 数量} 字典。"""
    values, counts = np.unique(np.asarray(labels)[idx], return_counts=True)
    return {str(v): int(c) for v, c in zip(values, counts)}


def read_idx_images(path):
    """读取 IDX3 格式的 MNIST 图像文件，返回 (N, 28, 28) 的 uint8 数组。"""
    with gzip.open(path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"非法的图像文件 magic: {magic}"
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows, cols)


def read_idx_labels(path):
    """读取 IDX1 格式的 MNIST 标签文件，返回 (N,) 的 uint8 数组。"""
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"非法的标签文件 magic: {magic}"
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


# --------------------------- iris ---------------------------
def split_iris():
    """对 iris 数据集做分层 80/20 划分，写出 csv。"""
    rows, labels = [], []
    with open(IRIS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:                       # 跳过末尾空行
                continue
            parts = line.split(",")
            rows.append(parts)
            labels.append(parts[-1])

    rng = np.random.default_rng(SEED)
    train_idx, val_idx = stratified_split(labels, VAL_RATIO, rng)

    for name, idx in (("train", train_idx), ("val", val_idx)):
        out = SPLITS_DIR / f"iris_{name}.csv"
        with open(out, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(IRIS_COLUMNS)
            for i in idx:
                writer.writerow(rows[int(i)])

    return {
        "total": len(rows),
        "train": {"count": int(len(train_idx)),
                  "per_class": class_counts(labels, train_idx)},
        "val": {"count": int(len(val_idx)),
                "per_class": class_counts(labels, val_idx)},
    }


# --------------------------- mnist --------------------------
def split_mnist():
    """对 MNIST 的 60000 训练样本做分层 80/20 划分，写出索引文件。"""
    labels = read_idx_labels(MNIST_DIR / "train-labels-idx1-ubyte.gz")

    rng = np.random.default_rng(SEED)
    train_idx, val_idx = stratified_split(labels, VAL_RATIO, rng)

    np.save(SPLITS_DIR / "mnist_train_indices.npy", train_idx.astype(np.int32))
    np.save(SPLITS_DIR / "mnist_val_indices.npy", val_idx.astype(np.int32))

    return {
        "total": int(len(labels)),
        "train": {"count": int(len(train_idx)),
                  "per_class": class_counts(labels, train_idx)},
        "val": {"count": int(len(val_idx)),
                "per_class": class_counts(labels, val_idx)},
    }


# ----------------------- 加载辅助函数 -----------------------
def load_iris_split(which="train"):
    """加载 iris 划分子集，返回 (X, y)。which ∈ {'train', 'val'}。"""
    path = SPLITS_DIR / f"iris_{which}.csv"
    data = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=str)
    X = data[:, :4].astype(np.float64)
    y = data[:, 4]
    return X, y


def load_mnist_split(which="train"):
    """加载 mnist 划分子集，返回 (images, labels)。images 形状 (N, 28, 28)。"""
    images = read_idx_images(MNIST_DIR / "train-images-idx3-ubyte.gz")
    labels = read_idx_labels(MNIST_DIR / "train-labels-idx1-ubyte.gz")
    idx = np.load(SPLITS_DIR / f"mnist_{which}_indices.npy")
    return images[idx], labels[idx]


def load_mnist_test():
    """加载 MNIST 官方 t10k 测试集，返回 (images, labels)。"""
    images = read_idx_images(MNIST_DIR / "t10k-images-idx3-ubyte.gz")
    labels = read_idx_labels(MNIST_DIR / "t10k-labels-idx1-ubyte.gz")
    return images, labels


# ---------------------------- 主流程 ----------------------------
def main():
    SPLITS_DIR.mkdir(exist_ok=True)

    iris_stats = split_iris()
    mnist_stats = split_mnist()

    summary = {
        "seed": SEED,
        "val_ratio": VAL_RATIO,
        "method": "stratified (per-class)",
        "datasets": {"iris": iris_stats, "mnist": mnist_stats},
    }
    with open(SPLITS_DIR / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    for name, stats in summary["datasets"].items():
        tr, va, total = stats["train"], stats["val"], stats["total"]
        print(f"[{name}] 总计 {total} -> 训练 {tr['count']} / 验证 {va['count']}")
        print(f"    训练各类别: {tr['per_class']}")
        print(f"    验证各类别: {va['per_class']}")
    print(f"\n划分完成，随机种子={SEED}，结果已写入 {SPLITS_DIR}")


if __name__ == "__main__":
    main()
