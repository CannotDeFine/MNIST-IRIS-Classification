# Experiment Roadmap

本文档说明本项目从最简单模型到最终模型的演进路径。每个版本都包含：做了什么、
结果如何、观察到什么问题，以及为什么进入下一版。实验路线遵循“先手搓核心训练流程，
再逐步加入组件，最后统一调参”的原则。详细实现位于
[src/classification_nn.py](/home/cdf/optimization/MNIST-IRIS-Classification/src/classification_nn.py:1)
和 [src/classification_experiments.py](/home/cdf/optimization/MNIST-IRIS-Classification/src/classification_experiments.py:1)。

## 结果文件索引

| 内容 | 文件 |
|---|---|
| 标准消融与基础版本对比 | [classification_ablation_l2.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/classification_ablation_l2.md) |
| baseline 进化路线重跑结果 | [baseline_progression.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/baseline_progression.md) |
| baseline 进化路线 full 结果 | [baseline_progression_full.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/baseline_progression_full.md) |
| 当前主模型完整训练结果 | [classification_nn_metrics.json](/home/cdf/optimization/MNIST-IRIS-Classification/results/classification_nn_metrics.json) |
| L2 参数扫描 | [l2_sweep.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/l2_sweep.md) |
| 学习率扫描 | [lr_sweep.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/lr_sweep.md) |
| 隐藏层宽度扫描 | [width_sweep.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/width_sweep.md) |
| 训练策略对比 | [training_strategies.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/training_strategies.md) |
| 损失函数对比 | [loss_comparison.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/loss_comparison.md) |
| IRIS 混淆矩阵图片 | [iris_mlp_iris_val_confusion.png](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/iris_mlp_iris_val_confusion.png) |
| IRIS 混淆矩阵表格 | [iris_mlp_iris_val_confusion.csv](/home/cdf/optimization/MNIST-IRIS-Classification/results/tables/iris_mlp_iris_val_confusion.csv) |
| MNIST quick 混淆矩阵图片 | [mnist_mlp_mnist_val_confusion.png](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/mnist_mlp_mnist_val_confusion.png) |
| MNIST quick 混淆矩阵表格 | [mnist_mlp_mnist_val_confusion.csv](/home/cdf/optimization/MNIST-IRIS-Classification/results/tables/mnist_mlp_mnist_val_confusion.csv) |
| MNIST CNN 扩展 | [cnn_extension.md](/home/cdf/optimization/MNIST-IRIS-Classification/docs/cnn_extension.md) |
| MNIST CNN quick 结果 | [cnn_mnist_metrics.json](/home/cdf/optimization/MNIST-IRIS-Classification/results/cnn_mnist_metrics.json) |
| MNIST full 候选模型汇总 | [mnist_full_candidates_summary.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/mnist_full_candidates_summary.md) |
| MNIST 最终模型汇总 | [final_mnist_summary.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/final_mnist_summary.md) |
| MNIST 最终训练/验证指标 | [final_mnist_train_metrics.json](/home/cdf/optimization/MNIST-IRIS-Classification/results/final_mnist_train_metrics.json) |
| MNIST 官方测试集指标 | [final_mnist_test_metrics.json](/home/cdf/optimization/MNIST-IRIS-Classification/results/final_mnist_test_metrics.json) |
| MNIST 最终 checkpoint | [final_mnist_mlp.npz](/home/cdf/optimization/MNIST-IRIS-Classification/results/models/final_mnist_mlp.npz) |
| MNIST early-stopping 最终模型汇总 | [final_mnist_es100_summary.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/final_mnist_es100_summary.md) |
| MNIST early-stopping 官方测试指标 | [final_mnist_es100_test_metrics.json](/home/cdf/optimization/MNIST-IRIS-Classification/results/final_mnist_es100_test_metrics.json) |
| 报告图：MNIST 方法进化 | [report_mnist_method_progression.png](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/report_mnist_method_progression.png) |
| 报告图：MNIST 累积消融 | [report_mnist_cumulative_ablation.png](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/report_mnist_cumulative_ablation.png) |
| 报告图：MNIST 损失函数检查 | [report_mnist_loss_function_check.png](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/report_mnist_loss_function_check.png) |
| 报告图：MNIST L2 扫描 | [report_mnist_l2_sweep.png](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/report_mnist_l2_sweep.png) |
| 报告图：MNIST 学习率扫描 | [report_mnist_learning_rate_sweep.png](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/report_mnist_learning_rate_sweep.png) |
| 报告图：MNIST 隐藏层宽度扫描 | [report_mnist_hidden_width_sweep.png](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/report_mnist_hidden_width_sweep.png) |
| 报告图：MNIST 训练策略 | [report_mnist_training_strategies.png](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/report_mnist_training_strategies.png) |
| 报告图：MNIST 最终权衡 | [report_mnist_final_tradeoff.png](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/report_mnist_final_tradeoff.png) |

注意：`results/` 目录默认被 `.gitignore` 忽略。上表链接指向本地已生成结果；如果换机器，
需要按 README 或本文档命令重新生成。

## Version Overview

本文档的主线按真实实验路线组织。所有核心训练逻辑均由 NumPy 手搓实现，包括前向传播、
softmax、loss、反向传播、梯度下降、momentum、L2 和 early stopping；没有使用
scikit-learn、PyTorch、TensorFlow 等第三方机器学习训练框架。

| Version | 内容 | 作用 |
|---|---|---|
| V0 | 未训练随机网络 | 确认训练前基线 |
| V1 | Linear Softmax baseline | 从无隐藏层的最简单可训练多分类模型开始 |
| V2 | Small MLP | 只加入小隐藏层，验证非线性表达能力 |
| V3 | Stable MLP | 逐步加入标准化/归一化、ReLU、He 初始化和 mini-batch |
| V4 | Component Checks | 分别检查损失函数、L2、学习率、宽度和训练策略 |
| Final | Unified Tuning and Final Model | 统一比较参数组合，再确定最终模型 |

quick 实验用于快速观察趋势；full 实验用于正式选择模型；test set 只在最终模型确定后
使用一次。候选模型之间应使用相同的数据划分、batch size、random seed、最大 epoch
预算和停止规则，避免把训练时长或数据差异误当成结构改进。

这里的 baseline 不是“故意很差的模型”。V0 是真正未训练的随机基线；V1 是最简单的
可训练分类器。由于 V1 已经使用归一化输入、softmax cross-entropy 和 mini-batch
gradient descent，它在 MNIST 上会比较强。后续版本的意义是说明我们如何在强基线之上
逐步增加表达能力、训练稳定性和收敛效率，而不是只和随机猜测比较。

## Reproduce All Experiments

按下面顺序可以重新生成本文档引用的核心结果。quick 实验用于探索趋势，最终测试只在
最终模型确定后执行。当前 full 候选使用相同 train/validation split、batch size、
random seed 和 5 epoch 训练预算。更严格的正式实验可以把 `--epochs` 作为最大 epoch，
并统一加入 `--early-stopping-patience`，根据 validation loss 停止训练。

```bash
uv run python src/classification_experiments.py \
  --dataset all \
  --profile quick \
  --output results/classification_ablation_l2.json \
  --markdown-output results/classification_ablation_l2.md

uv run python src/classification_experiments.py \
  --dataset all \
  --profile quick \
  --output results/baseline_progression.json \
  --markdown-output results/baseline_progression.md

uv run python src/classification_experiments.py \
  --dataset all \
  --profile full \
  --output results/baseline_progression_full.json \
  --markdown-output results/baseline_progression_full.md

uv run python src/classification_experiments.py --suite loss-comparison \
  --dataset all --profile quick \
  --output results/loss_comparison.json \
  --markdown-output results/loss_comparison.md

uv run python src/classification_experiments.py --suite l2-sweep \
  --dataset all --profile quick \
  --output results/l2_sweep.json \
  --markdown-output results/l2_sweep.md

uv run python src/classification_experiments.py --suite lr-sweep \
  --dataset all --profile quick \
  --output results/lr_sweep.json \
  --markdown-output results/lr_sweep.md

uv run python src/classification_experiments.py --suite width-sweep \
  --dataset all --profile quick \
  --output results/width_sweep.json \
  --markdown-output results/width_sweep.md

uv run python src/classification_experiments.py --suite training-strategies \
  --dataset all --profile quick \
  --output results/training_strategies.json \
  --markdown-output results/training_strategies.md

uv run python src/cnn_mnist.py \
  --train-limit 1000 \
  --val-limit 1000 \
  --epochs 3 \
  --batch-size 32 \
  --filters 8 \
  --learning-rate 0.05 \
  --output results/cnn_mnist_metrics.json \
  --quiet

uv run python main.py --dataset mnist --epochs 5 \
  --hidden-dim 128 --learning-rate 0.1 --batch-size 128 \
  --optimizer sgd --l2 0 \
  --output results/mnist_full_current_mlp.json --quiet

uv run python main.py --dataset mnist --epochs 5 \
  --hidden-dim 256 --learning-rate 0.1 --batch-size 128 \
  --optimizer sgd --l2 0 \
  --output results/mnist_full_width_256.json --quiet

uv run python main.py --dataset mnist --epochs 5 \
  --hidden-dim 128 --learning-rate 0.05 --batch-size 128 \
  --optimizer momentum --momentum 0.9 --l2 0 \
  --output results/mnist_full_momentum.json --quiet

uv run python main.py --dataset mnist --epochs 5 \
  --hidden-dim 128 --learning-rate 0.1 --batch-size 128 \
  --optimizer sgd --l2 0.001 \
  --output results/mnist_full_l2_1e-3.json --quiet

uv run python main.py --dataset mnist --epochs 5 \
  --hidden-dim 256 --learning-rate 0.05 --batch-size 128 \
  --optimizer momentum --momentum 0.9 --l2 0.001 \
  --output results/mnist_full_combined_candidate.json --quiet

uv run python main.py --dataset mnist --epochs 5 \
  --hidden-dim 256 --learning-rate 0.05 --batch-size 128 \
  --optimizer momentum --momentum 0.9 --l2 0 \
  --output results/mnist_full_width256_momentum_no_l2.json --quiet

uv run python main.py \
  --dataset iris \
  --epochs 300 \
  --hidden-dim 16 \
  --learning-rate 0.05 \
  --batch-size 16 \
  --optimizer sgd \
  --l2 0 \
  --model-output results/models/iris_mlp.npz \
  --output results/iris_model_metrics.json \
  --quiet

uv run python src/plot_results.py confusion \
  --model results/models/iris_mlp.npz \
  --split val \
  --figure-dir results/figures \
  --table-dir results/tables

uv run python main.py \
  --dataset mnist \
  --epochs 100 \
  --hidden-dim 256 \
  --learning-rate 0.05 \
  --batch-size 128 \
  --optimizer momentum \
  --momentum 0.9 \
  --l2 0 \
  --early-stopping-patience 3 \
  --model-output results/models/final_mnist_mlp_es100.npz \
  --output results/mnist_full_es100_width256_momentum_no_l2.json \
  --quiet

uv run python main.py \
  --load-model results/models/final_mnist_mlp_es100.npz \
  --eval-split test \
  --output results/final_mnist_es100_test_metrics.json \
  --quiet

uv run python src/plot_results.py report-figures \
  --results-dir results \
  --output-dir results/figures
```

## Version 0: Untrained Random Network

做法：初始化 MLP，但不进行梯度更新，只记录 `epoch 000` 的指标。

对应实验：

- `iris_untrained_mlp`
- `mnist_untrained_mlp`

结果：

| 数据集 | val accuracy |
|---|---:|
| IRIS | 0.4667 |
| MNIST quick | 0.1400 |

结果来源：[classification_ablation_l2.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/classification_ablation_l2.md)

复现命令：

```bash
uv run python src/classification_experiments.py \
  --dataset all \
  --profile quick \
  --output results/classification_ablation_l2.json \
  --markdown-output results/classification_ablation_l2.md
```

分析：MNIST 接近 10 类随机猜测水平；IRIS 高于 1/3 是随机初始化带来的类别偏置，
不代表模型已经学习。下一步需要通过反向传播和梯度下降训练参数。

## Version 1: Linear Softmax Baseline

做法：使用最简单的可训练分类器，无隐藏层。

```text
IRIS: 4 -> softmax(3)
MNIST: 784 -> softmax(10)
```

对应实验：

- `iris_linear_softmax`
- `mnist_linear_softmax`

结果：

| 数据集 | val accuracy | train accuracy |
|---|---:|---:|
| IRIS | 0.9667 | 0.9417 |
| MNIST quick | 0.8340 | 0.8500 |

结果来源：[classification_ablation_l2.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/classification_ablation_l2.md)

复现命令：

```bash
uv run python src/classification_experiments.py \
  --dataset all \
  --profile quick \
  --output results/classification_ablation_l2.json \
  --markdown-output results/classification_ablation_l2.md
```

分析：这是“最简单可训练 baseline”，不是最弱模型。它已经完成了从零实现的 softmax、
cross-entropy、反向传播和 mini-batch 梯度更新，因此 MNIST quick 上可以达到较高准确率。
但线性模型表达能力有限，无法表达更复杂的非线性边界，因此下一步只加入小隐藏层和非线性激活。

## Version 2: Small One-hidden-layer MLP

做法：加入一个 ReLU 隐藏层，但先使用较小隐藏层，观察非线性模型是否能学习。

```text
IRIS: 4 -> ReLU(4) -> softmax(3)
MNIST: 784 -> ReLU(16) -> softmax(10)
```

对应实验：

- `iris_small_hidden`
- `mnist_small_hidden`

结果：

| 数据集 | val accuracy | train accuracy |
|---|---:|---:|
| IRIS | 0.9333 | 0.9583 |
| MNIST quick | 0.7380 | 0.7690 |

结果来源：[classification_ablation_l2.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/classification_ablation_l2.md)

复现命令：

```bash
uv run python src/classification_experiments.py \
  --dataset all \
  --profile quick \
  --output results/classification_ablation_l2.json \
  --markdown-output results/classification_ablation_l2.md
```

分析：小隐藏层可以训练，但容量有限，尤其 MNIST 上 16 个隐藏单元不足。下一步需要增加
隐藏层宽度，并改善训练稳定性。

## Version 3: Stable Training MLP

做法：在一层 MLP 基础上加入稳定训练设置：

- IRIS 使用训练集均值和标准差做标准化。
- MNIST 像素缩放到 `[0, 1]`。
- 隐藏层使用 ReLU。
- 权重使用适合 ReLU 的 He 初始化。
- 使用 mini-batch SGD。
- 使用较合适的学习率。

当前主模型：

```text
IRIS: 4 -> ReLU(16) -> softmax(3)
MNIST: 784 -> ReLU(128) -> softmax(10)
```

结果：

| 数据集 | 设置 | val accuracy | train accuracy |
|---|---|---:|---:|
| IRIS | quick/current | 0.9667 | 0.9667 |
| MNIST | quick/current | 0.8180 | 0.8480 |
| MNIST | full current | 0.9467 | 0.9564 |

结果来源：

- quick/current：[classification_ablation_l2.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/classification_ablation_l2.md)
- full current：[classification_nn_metrics.json](/home/cdf/optimization/MNIST-IRIS-Classification/results/classification_nn_metrics.json)

复现命令：

```bash
uv run python src/classification_experiments.py \
  --dataset all \
  --profile quick \
  --output results/classification_ablation_l2.json \
  --markdown-output results/classification_ablation_l2.md

uv run python main.py \
  --dataset mnist \
  --epochs 5 \
  --hidden-dim 128 \
  --learning-rate 0.1 \
  --batch-size 128 \
  --optimizer sgd \
  --l2 0 \
  --output results/mnist_full_current_mlp.json \
  --quiet
```

支撑对比：

| 对比项 | 观察 |
|---|---|
| `iris_no_standardization` | IRIS 验证准确率降到 0.9000 |
| `mnist_no_pixel_scaling` | MNIST quick 验证准确率降到 0.1090 |
| `mnist_full_batch` | MNIST quick 验证准确率为 0.4950，低于 mini-batch |
| `mnist_slow_learning_rate` | MNIST quick 验证准确率为 0.4340 |
| `mnist_small_normal_init` | MNIST quick 验证准确率为 0.4110 |

分析：输入缩放、mini-batch 更新频率、He 初始化和合适学习率对 MNIST 训练非常关键。
这一版不是最终调参结果，而是后续组件检查的稳定基础。

### Loss Function Check

在 stable MLP 设置下，我们额外对比了两种输出层和损失函数组合：

```text
Sigmoid + MSE
Softmax + Cross-Entropy
```

结果：

| 数据集 | 损失设置 | val accuracy | train accuracy |
|---|---|---:|---:|
| IRIS | Sigmoid + MSE | 0.9333 | 0.9000 |
| IRIS | Softmax + Cross-Entropy | 0.9667 | 0.9667 |
| MNIST quick | Sigmoid + MSE | 0.4230 | 0.5010 |
| MNIST quick | Softmax + Cross-Entropy | 0.8180 | 0.8480 |

完整结果：[loss_comparison.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/loss_comparison.md)

复现命令：

```bash
uv run python src/classification_experiments.py --suite loss-comparison \
  --dataset all \
  --profile quick \
  --output results/loss_comparison.json \
  --markdown-output results/loss_comparison.md
```

分析：`Softmax + Cross-Entropy` 在两个数据集上都更好，MNIST 差距尤其明显。因此后续
版本继续使用 softmax 交叉熵作为多分类训练目标。

## Version 4: Component Checks

做法：在 stable MLP 基础上逐项检查会影响训练效果的因素，包括损失函数、L2 正则化、
学习率、隐藏层宽度和训练策略。这里不急着确定最终模型，而是观察每个组件的作用。
L2 是验证项，不是最终模型的提升点。

### L2 Regularization Check

在 stable MLP 上加入 L2 正则化，优化目标变为：

```text
loss = cross_entropy + λ/2 * (||W1||² + ||W2||²)
```

对应实验：

- `iris_l2_regularization`
- `mnist_l2_regularization`
- `--suite l2-sweep`

L2 扫描结果摘要：

| 数据集 | λ | val accuracy | train accuracy |
|---|---:|---:|---:|
| MNIST quick | 0 | 0.8180 | 0.8480 |
| MNIST quick | 1e-4 | 0.8180 | 0.8480 |
| MNIST quick | 1e-3 | 0.8190 | 0.8470 |
| MNIST quick | 1e-2 | 0.8160 | 0.8470 |
| MNIST quick | 1e-1 | 0.8010 | 0.8200 |

完整结果：[l2_sweep.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/l2_sweep.md)

复现命令：

```bash
uv run python src/classification_experiments.py --suite l2-sweep \
  --dataset all \
  --profile quick \
  --output results/l2_sweep.json \
  --markdown-output results/l2_sweep.md
```

分析：小的 L2 对结果影响不大，`λ=1e-3` 在 MNIST quick 上只高 0.001，不能作为显著
提升结论；`λ=0.1` 时训练和验证准确率都下降，说明正则化过强会欠拟合。full 候选中
`hidden_dim=256 + momentum + L2=1e-3` 的验证准确率低于 no-L2 版本，因此最终模型
没有使用 L2。

### 学习率

| 数据集 | 最佳/代表设置 | 观察 |
|---|---|---|
| IRIS | `lr=0.05` 或 `0.1` | 验证准确率达到 0.9667 |
| MNIST quick | `lr=0.1` | 验证准确率 0.8180，明显优于 0.001 和 0.01 |

完整结果：[lr_sweep.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/lr_sweep.md)

复现命令：

```bash
uv run python src/classification_experiments.py --suite lr-sweep \
  --dataset all \
  --profile quick \
  --output results/lr_sweep.json \
  --markdown-output results/lr_sweep.md
```

### 隐藏层宽度

| 数据集 | 最佳/代表设置 | 观察 |
|---|---|---|
| IRIS | `hidden_dim=8` | 本次验证准确率 1.0000，但验证集小，需谨慎解释 |
| MNIST quick | `hidden_dim=256` | 验证准确率 0.8400 |

完整结果：[width_sweep.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/width_sweep.md)

复现命令：

```bash
uv run python src/classification_experiments.py --suite width-sweep \
  --dataset all \
  --profile quick \
  --output results/width_sweep.json \
  --markdown-output results/width_sweep.md
```

### 训练策略

| 数据集 | 策略 | val accuracy | 说明 |
|---|---|---:|---|
| MNIST quick | SGD | 0.8180 | 当前 quick 基线 |
| MNIST quick | momentum | 0.8580 | 历史梯度方向提升收敛 |
| MNIST quick | LR decay | 0.8130 | 当前 decay 设置未提升 |
| MNIST quick | early stopping | 0.8710 | 上限 20 epoch，提升也来自更长训练 |

完整结果：[training_strategies.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/training_strategies.md)

复现命令：

```bash
uv run python src/classification_experiments.py --suite training-strategies \
  --dataset all \
  --profile quick \
  --output results/training_strategies.json \
  --markdown-output results/training_strategies.md
```

分析：MNIST quick 中 momentum 优于普通 5-epoch SGD。early stopping 的结果需要谨慎
解释，因为该实验的最大 epoch 是 20，提升不一定来自“提前停止”本身，也可能来自允许
模型训练更久。更合理的报告表述是：early stopping 主要用于根据 validation loss 控制
训练停止，避免固定 epoch 过少或过多。

本阶段只说明各个组件的趋势，不直接宣布最终模型。最终模型选择放到下一阶段，在 full
训练设置下统一比较候选参数组合。

## Final Selected IRIS MLP

IRIS 没有单独官方 test set，因此使用固定训练/验证划分报告最终验证集结果。

最终 IRIS MLP：

```text
4 -> ReLU(16) -> softmax(3)
preprocessing = train-set standardization
loss = Softmax + Cross-Entropy
optimizer = mini-batch SGD
initialization = He initialization
```

结果：

| Split | Accuracy | Loss | Count |
|---|---:|---:|---:|
| Train | 0.9750 | 0.0527 | 120 |
| Validation | 0.9667 | 0.0606 | 30 |

结果来源：

- [classification_nn_metrics.json](/home/cdf/optimization/MNIST-IRIS-Classification/results/classification_nn_metrics.json)
- [iris_mlp_iris_val_confusion.png](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/iris_mlp_iris_val_confusion.png)
- [iris_mlp_iris_val_confusion.csv](/home/cdf/optimization/MNIST-IRIS-Classification/results/tables/iris_mlp_iris_val_confusion.csv)

复现命令：

```bash
uv run python main.py \
  --dataset iris \
  --epochs 300 \
  --hidden-dim 16 \
  --learning-rate 0.05 \
  --batch-size 16 \
  --optimizer sgd \
  --l2 0 \
  --model-output results/models/iris_mlp.npz \
  --output results/iris_model_metrics.json \
  --quiet

uv run python src/plot_results.py confusion \
  --model results/models/iris_mlp.npz \
  --split val \
  --figure-dir results/figures \
  --table-dir results/tables
```

## Final: Unified Tuning and Selected MNIST MLP

最后一步不再单独讨论某一个组件，而是在完整 MNIST 训练/验证划分上统一比较候选参数组合。
所有候选都保持相同数据划分、batch size、random seed、最大 epoch 和 early-stopping
规则，然后根据 validation loss / validation accuracy 选择最终模型。

当前选择：

```text
784 -> ReLU(256) -> softmax(10)
initialization = He initialization
loss = Softmax + Cross-Entropy
training = mini-batch gradient descent
optimizer = momentum
learning_rate = 0.05
momentum = 0.9
batch_size = 128
l2 = 0
max_epoch = 100
early_stopping_patience = 3
```

候选模型汇总见
[mnist_full_candidates_summary.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/mnist_full_candidates_summary.md)。
在该组实验里，`hidden_dim=256 + momentum + l2=0` 的验证集准确率最高。最终模型主要
依靠 ReLU、He 初始化、Softmax+Cross-Entropy、mini-batch、momentum 和
`hidden_dim=256`；没有使用 L2，也没有使用 learning rate decay。

早期固定 5 epoch full 候选结果：

| Candidate | Hidden | Optimizer | L2 | Val accuracy |
|---|---:|---|---:|---:|
| Current MLP | 128 | sgd | 0 | 0.9467 |
| Width 256 | 256 | sgd | 0 | 0.9519 |
| Momentum | 128 | momentum | 0 | 0.9706 |
| L2 1e-3 | 128 | sgd | 0.001 | 0.9439 |
| Width 256 + momentum + L2 | 256 | momentum | 0.001 | 0.9719 |
| Width 256 + momentum, no L2 | 256 | momentum | 0 | 0.9742 |

这些结果说明 momentum 和更宽隐藏层都有效，但固定 5 epoch 仍然可能限制部分模型。
因此正式比较进一步使用统一 `max_epoch=100` 和 `early_stopping_patience=3`。

统一 early-stopping full 候选结果：

| Candidate | Hidden | Optimizer | L2 | Epochs trained | Val accuracy | Val loss |
|---|---:|---|---:|---:|---:|---:|
| Current MLP | 128 | sgd | 0 | 48 | 0.9754 | 0.0826 |
| Width 256 | 256 | sgd | 0 | 45 | 0.9762 | 0.0798 |
| Momentum | 128 | momentum | 0 | 12 | 0.9767 | 0.0825 |
| L2 1e-3 | 128 | sgd | 0.001 | 57 | 0.9731 | 0.1831 |
| Width 256 + momentum, no L2 | 256 | momentum | 0 | 18 | 0.9784 | 0.0781 |

对应图表：[report_mnist_final_tradeoff.png](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/report_mnist_final_tradeoff.png)

统一调参后的最终结果：

| Split | Accuracy | Loss | Count |
|---|---:|---:|---:|
| Train | 0.9998 | 0.0047 | 48000 |
| Validation | 0.9784 | 0.0781 | 12000 |
| Test | 0.9800 | 0.0673 | 10000 |

结果来源：

- [final_mnist_es100_summary.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/final_mnist_es100_summary.md)
- [mnist_full_es100_width256_momentum_no_l2.json](/home/cdf/optimization/MNIST-IRIS-Classification/results/mnist_full_es100_width256_momentum_no_l2.json)
- [final_mnist_es100_test_metrics.json](/home/cdf/optimization/MNIST-IRIS-Classification/results/final_mnist_es100_test_metrics.json)

复现命令：

```bash
uv run python main.py --dataset mnist --epochs 100 \
  --hidden-dim 128 --learning-rate 0.1 --batch-size 128 \
  --optimizer sgd --l2 0 \
  --early-stopping-patience 3 \
  --output results/mnist_full_es100_current_mlp.json --quiet

uv run python main.py --dataset mnist --epochs 100 \
  --hidden-dim 256 --learning-rate 0.1 --batch-size 128 \
  --optimizer sgd --l2 0 \
  --early-stopping-patience 3 \
  --output results/mnist_full_es100_width_256.json --quiet

uv run python main.py --dataset mnist --epochs 100 \
  --hidden-dim 128 --learning-rate 0.05 --batch-size 128 \
  --optimizer momentum --momentum 0.9 --l2 0 \
  --early-stopping-patience 3 \
  --output results/mnist_full_es100_momentum.json --quiet

uv run python main.py --dataset mnist --epochs 100 \
  --hidden-dim 128 --learning-rate 0.1 --batch-size 128 \
  --optimizer sgd --l2 0.001 \
  --early-stopping-patience 3 \
  --output results/mnist_full_es100_l2_1e-3.json --quiet

uv run python main.py --dataset mnist --epochs 100 \
  --hidden-dim 256 --learning-rate 0.05 --batch-size 128 \
  --optimizer momentum --momentum 0.9 --l2 0.001 \
  --early-stopping-patience 3 \
  --output results/mnist_full_es100_combined_candidate.json --quiet

uv run python main.py --dataset mnist --epochs 100 \
  --hidden-dim 256 --learning-rate 0.05 --batch-size 128 \
  --optimizer momentum --momentum 0.9 --l2 0 \
  --early-stopping-patience 3 \
  --output results/mnist_full_es100_width256_momentum_no_l2.json --quiet

uv run python main.py \
  --dataset mnist \
  --epochs 100 \
  --hidden-dim 256 \
  --learning-rate 0.05 \
  --batch-size 128 \
  --optimizer momentum \
  --momentum 0.9 \
  --l2 0 \
  --early-stopping-patience 3 \
  --model-output results/models/final_mnist_mlp_es100.npz \
  --output results/mnist_full_es100_width256_momentum_no_l2.json \
  --quiet

uv run python main.py \
  --load-model results/models/final_mnist_mlp_es100.npz \
  --eval-split test \
  --output results/final_mnist_es100_test_metrics.json \
  --quiet
```

分析：验证集用于模型选择，官方 t10k 测试集只在最终模型确定后使用一次。增大最大 epoch
后，各模型验证准确率更接近；momentum 的优势主要体现在用更少 epoch 达到接近或更好的
结果。最终模型训练 18 个 epoch 后停止，测试准确率为 `0.9800`。

## 分析图表

这些图表不属于训练演进版本本身，而是用于结果分析。

报告叙事图：

- [Method progression](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/report_mnist_method_progression.png)
- [Cumulative ablation](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/report_mnist_cumulative_ablation.png)
- [Loss function check](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/report_mnist_loss_function_check.png)
- [L2 sweep](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/report_mnist_l2_sweep.png)
- [Learning-rate sweep](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/report_mnist_learning_rate_sweep.png)
- [Hidden-width sweep](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/report_mnist_hidden_width_sweep.png)
- [Training strategies](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/report_mnist_training_strategies.png)
- [Final tuning trade-off](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/report_mnist_final_tradeoff.png)

混淆矩阵：

- [IRIS confusion matrix](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/iris_mlp_iris_val_confusion.png)
- [IRIS confusion table](/home/cdf/optimization/MNIST-IRIS-Classification/results/tables/iris_mlp_iris_val_confusion.csv)
- [MNIST quick confusion matrix](/home/cdf/optimization/MNIST-IRIS-Classification/results/figures/mnist_mlp_mnist_val_confusion.png)
- [MNIST quick confusion table](/home/cdf/optimization/MNIST-IRIS-Classification/results/tables/mnist_mlp_mnist_val_confusion.csv)

## CNN Extension

CNN 是独立扩展，不属于 MLP 训练演进主线。实现和说明见
[docs/cnn_extension.md](/home/cdf/optimization/MNIST-IRIS-Classification/docs/cnn_extension.md)。

当前 quick 设置结果：

| Model | Train samples | Val samples | Epochs | Val accuracy | Train accuracy |
|---|---:|---:|---:|---:|---:|
| Simple CNN | 1000 | 1000 | 3 | 0.7970 | 0.8240 |

结果来源：[cnn_mnist_metrics.json](/home/cdf/optimization/MNIST-IRIS-Classification/results/cnn_mnist_metrics.json)

复现命令：

```bash
uv run python src/cnn_mnist.py \
  --train-limit 1000 \
  --val-limit 1000 \
  --epochs 3 \
  --batch-size 32 \
  --filters 8 \
  --learning-rate 0.05 \
  --output results/cnn_mnist_metrics.json \
  --quiet
```

这个结果可以作为“尝试更适合图像的局部特征提取结构”的扩展示例。由于当前 CNN 只训练
3 epoch 且结构较小，不建议用它替代 MLP 主线模型。
