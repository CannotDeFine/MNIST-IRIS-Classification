# 简易神经网络分类模型与消融实验说明

本文档说明当前 `src/classification_nn.py` 中的 iris / MNIST 分类模型从最简单神经网络
到当前实现经历了哪些优化，并给出可复现的消融实验实现、运行命令和参考结果。

## 1. 当前模型范围

当前实现只使用 NumPy，没有使用 PyTorch、TensorFlow、Adam、BatchNorm 或数据增强。
主线模型是单隐藏层 MLP，训练算法以 mini-batch SGD 为基础，可选 momentum、
学习率衰减和 early stopping。基础损失函数是 softmax 交叉熵，主模型可选加入
L2 正则化，使优化目标变为交叉熵加权重平方惩罚。CNN 作为独立扩展实现，不混入
MLP 主线。

当前主入口：

```bash
uv run python main.py --dataset all
```

对应默认配置如下：

| 数据集 | 输入                      | 预处理                          | 模型                                | 默认 epoch | 学习率 | batch size | 默认 L2 |
| ------ | ------------------------- | ------------------------------- | ----------------------------------- | ---------: | -----: | ---------: | ------: |
| iris   | 4 个数值特征              | 用训练集均值/标准差标准化       | `4 -> ReLU(16) -> softmax(3)`     |        300 |   0.05 |         16 |       0 |
| MNIST  | 28x28 灰度图展平为 784 维 | 像素除以 255，缩放到 `[0, 1]` | `784 -> ReLU(128) -> softmax(10)` |          5 |    0.1 |        128 |       0 |

`fit()` 会先记录 `epoch 000`，表示初始化后、尚未做任何梯度更新时的准确率。
因此 `epoch 001` 已经不是“初始准确率”，而是完整遍历训练集一遍后的结果。

## 2. 从最简单模型到当前模型

### 2.1 最简单可训练模型：线性 softmax

最简单的神经网络分类器可以写成：

```text
logits = XW + b
probabilities = softmax(logits)
loss = cross_entropy(probabilities, y)
```

它没有隐藏层，也没有非线性表达能力，本质上是 softmax regression。本文的消融脚本
用 `SoftmaxRegressionClassifier` 实现这个基线。

### 2.2 输入缩放/标准化

当前实现对 iris 使用训练集统计量标准化，对 MNIST 使用 `pixel / 255.0`。这不是
改变模型表达能力，而是让梯度尺度合理。MNIST 如果直接使用 0-255 原始像素，同时
保持当前学习率 0.1，第一轮训练后几乎不能学习。

### 2.3 隐藏层与 ReLU

当前模型加入一个隐藏层：

```text
hidden = ReLU(XW1 + b1)
logits = hidden W2 + b2
```

隐藏层让模型可以表达非线性决策边界。iris 很小，线性模型已经很强；MNIST 中隐藏层
通常在更多数据和足够训练轮数下更有优势。

### 2.4 隐藏层宽度

MNIST 默认使用 128 个隐藏单元，iris 默认使用 16 个隐藏单元。宽度更大时，模型容量
更强，但也更依赖合适的初始化、学习率和样本量。消融中将 MNIST 宽度降到 16 后，
验证准确率明显下降。

### 2.5 He 初始化

当前 MLP 使用 He 初始化：

```text
W1 ~ N(0, sqrt(2 / input_dim))
W2 ~ N(0, sqrt(2 / hidden_dim))
```

这是为 ReLU 网络准备的初始化，使前向激活和反向梯度在训练初期更稳定。消融实验中
将其替换为很小的 `N(0, 0.01)` 初始化后，MNIST 收敛明显变慢。

### 2.6 mini-batch SGD

当前 MNIST 使用 batch size 128。这里的一个 epoch 包含多次参数更新。例如 MNIST
训练划分有 48000 张图，batch size 128 时一个 epoch 大约有 375 次更新。若改成
full-batch，每个 epoch 只有 1 次更新，在相同 epoch 数下学习会慢很多。

### 2.7 学习率

MNIST 默认学习率 0.1 是一个偏快的教学配置。将学习率降为 0.01 后，同样 5 个 epoch
的验证准确率会明显降低。这也是为什么你会看到 MNIST 精度上升很快：它不是高级优化器，
而是合理缩放输入后，使用了较大的学习率和很多 mini-batch 更新。

### 2.8 L2 正则化

当前 MLP 可以通过 `--l2` 或实验配置中的 `l2` 参数加入 L2 权重惩罚：

```text
loss = cross_entropy + λ/2 * (||W1||² + ||W2||²)
dW1 = dW1_data + λW1
dW2 = dW2_data + λW2
```

这里的 `λ` 控制正则化强度。L2 不改变网络结构，而是改变优化目标，限制权重无限增大。
实验中主要观察训练准确率、验证准确率以及二者差距是否变化。若 `λ` 太大，模型会欠拟合；
若 `λ` 太小，效果可能不明显。

### 2.9 高级训练策略

当前 `MLPClassifier` 默认仍使用 mini-batch SGD。为了做优化策略对比，模型还支持：

- `optimizer="momentum"`：使用 momentum SGD，累积历史梯度方向。
- `lr_decay`：使用 `α_t = α_0 / (1 + decay * t)` 的反比例学习率衰减。
- `early_stopping_patience`：验证集 loss 连续若干 epoch 未改善时提前停止。

这些策略不改变网络结构，只改变参数更新路径，适合放在最优化课程报告中讨论。

## 3. 消融实验实现

消融脚本位于：

```text
src/classification_experiments.py
```

它实现了：

- `SoftmaxRegressionClassifier`：线性 softmax 基线。
- `ExperimentConfig`：每个实验的模型、学习率、batch size、预处理、初始化和 L2 配置。
- `build_experiment_configs()`：生成 iris / MNIST 的标准消融配置。
- `run_experiment()`：加载固定训练/验证划分，训练并返回初始/最终指标。
- Markdown 与 JSON 输出，便于写报告或复现实验。

主训练脚本 `src/classification_nn.py` 另外实现了模型保存和加载：

- `save_checkpoint()`：保存 `W1/b1/W2/b2` 参数、数据集名称、训练配置和预处理元数据。
- `load_checkpoint()`：从 `.npz` 文件恢复模型参数和预处理元数据。
- `evaluate_checkpoint()`：加载 checkpoint 后，重新读取训练集或验证集并计算 loss / accuracy。

checkpoint 中包含 iris 的标准化均值/标准差和标签映射；MNIST checkpoint 中包含标签映射
和训练时的 `mnist_limit`。因此加载模型后评估验证集时，不需要重新训练，也不会复用
训练过程中的内存状态。

默认 `quick` profile 规则：

- iris：使用完整训练/验证划分，训练 100 epoch。
- MNIST：使用训练集前 1000 张和验证集前 1000 张，训练 5 epoch。
- `mnist_no_pixel_scaling` 只训练 1 epoch，因为使用原始 0-255 像素和当前学习率
  已经足以展示该消融会破坏训练尺度。
- L2 消融默认使用 `iris_l2_regularization: λ=1e-3` 与
  `mnist_l2_regularization: λ=1e-4`，作为温和权重惩罚的对比项。
- `--suite l2-sweep` 会专门扫描多组 L2 系数，用于观察 `λ` 从很小到过大时的趋势。
- `--suite lr-sweep` 扫描学习率，`--suite width-sweep` 扫描隐藏层宽度，
  `--suite training-strategies` 对比普通 SGD、momentum、学习率衰减和 early stopping。
- `--suite loss-comparison` 对比 `Sigmoid + MSE` 与 `Softmax + Cross-Entropy`。

这个 profile 用于快速复现实验趋势，不是最终性能排行榜。最终性能应使用完整 MNIST
划分重新运行。

## 4. 复现命令

创建/同步环境：

```bash
uv sync
```

运行快速消融，并输出 JSON 与 Markdown 表格：

```bash
uv run python src/classification_experiments.py \
  --dataset all \
  --profile quick \
  --output results/classification_ablation_quick.json \
  --markdown-output results/classification_ablation_quick.md
```

只运行 MNIST 消融：

```bash
uv run python src/classification_experiments.py \
  --dataset mnist \
  --profile quick \
  --output results/mnist_ablation_quick.json
```

运行当前主模型：

```bash
uv run python main.py --dataset all
```

只查看训练前基线：

```bash
uv run python main.py --dataset all --mnist-limit 1000 --epochs 0
```

使用 L2 正则化训练，并将结果单独保存：

```bash
uv run python main.py \
  --dataset all \
  --l2 0.001 \
  --output results/classification_nn_l2_metrics.json
```

扫描多组 L2 正则化系数：

```bash
uv run python src/classification_experiments.py \
  --suite l2-sweep \
  --dataset all \
  --profile quick \
  --output results/l2_sweep.json \
  --markdown-output results/l2_sweep.md
```

扫描学习率、隐藏层宽度和训练策略：

```bash
uv run python src/classification_experiments.py \
  --suite lr-sweep \
  --dataset all \
  --profile quick \
  --output results/lr_sweep.json \
  --markdown-output results/lr_sweep.md

uv run python src/classification_experiments.py \
  --suite width-sweep \
  --dataset all \
  --profile quick \
  --output results/width_sweep.json \
  --markdown-output results/width_sweep.md

uv run python src/classification_experiments.py \
  --suite training-strategies \
  --dataset all \
  --profile quick \
  --output results/training_strategies.json \
  --markdown-output results/training_strategies.md

uv run python src/classification_experiments.py \
  --suite loss-comparison \
  --dataset all \
  --profile quick \
  --output results/loss_comparison.json \
  --markdown-output results/loss_comparison.md
```

绘制学习曲线和混淆矩阵：

```bash
uv run python src/plot_results.py curves \
  --metrics results/classification_nn_metrics.json \
  --output-dir results/figures

uv run python src/plot_results.py confusion \
  --model results/models/iris_mlp.npz \
  --split val \
  --figure-dir results/figures \
  --table-dir results/tables
```

训练并保存参数：

```bash
uv run python main.py \
  --dataset mnist \
  --mnist-limit 1000 \
  --epochs 1 \
  --model-output results/models/mnist_mlp.npz \
  --output results/mnist_train_metrics.json \
  --quiet
```

重新加载参数，在验证集上独立评估：

```bash
uv run python main.py \
  --load-model results/models/mnist_mlp.npz \
  --eval-split val \
  --output results/mnist_loaded_val_metrics.json \
  --quiet
```

如果训练时使用了 `--mnist-limit 1000`，checkpoint 会记录这个设置；加载评估时默认
使用相同的验证集前 1000 张样本。训练完整 MNIST 时不传 `--mnist-limit`，加载评估
会使用完整验证划分。MNIST 还支持 `--eval-split test`，用于在官方 t10k 测试集上
做最终评估；iris 没有单独官方测试集，因此只支持 `train` 和 `val`。

注意：`results/` 默认被 `.gitignore` 忽略；实验结果文件用于本地复现和报告整理。

## 5. 快速消融参考结果

以下结果由命令生成：

```bash
uv run python src/classification_experiments.py \
  --dataset all \
  --profile quick \
  --output /private/tmp/classification_ablation_quick.json \
  --markdown-output /private/tmp/classification_ablation_quick.md
```

运行环境：Python 3.12.13，uv 管理环境，随机种子 `42`，固定使用 `splits/` 中已有
训练/验证划分。

| Experiment               | Ablation                                    | Epochs |   LR |     L2 | Batch | Initial val acc | Final val acc | Final train acc |
| ------------------------ | ------------------------------------------- | -----: | ---: | -----: | ----: | --------------: | ------------: | --------------: |
| iris_untrained_mlp       | all optimization steps after initialization |      0 | 0.05 |      0 |    16 |          0.4667 |        0.4667 |          0.4583 |
| iris_linear_softmax      | hidden ReLU layer                           |    100 | 0.05 |      0 |    16 |          0.7333 |        0.9667 |          0.9417 |
| iris_small_hidden        | 16-unit hidden representation               |    100 | 0.05 |      0 |    16 |          0.3333 |        0.9333 |          0.9583 |
| iris_no_standardization  | feature standardization                     |    100 | 0.05 |      0 |    16 |          0.3333 |        0.9000 |          0.9500 |
| iris_small_normal_init   | He initialization for ReLU layers           |    100 | 0.05 |      0 |    16 |          0.4667 |        1.0000 |          0.9667 |
| iris_l2_regularization   | unregularized objective                     |    100 | 0.05 |  0.001 |    16 |          0.4667 |        0.9667 |          0.9667 |
| iris_current_mlp         | none                                        |    100 | 0.05 |      0 |    16 |          0.4667 |        0.9667 |          0.9667 |
| mnist_untrained_mlp      | all optimization steps after initialization |      0 |  0.1 |      0 |   128 |          0.1400 |        0.1400 |          0.1020 |
| mnist_linear_softmax     | hidden ReLU layer and hidden width          |      5 |  0.1 |      0 |   128 |          0.1120 |        0.8340 |          0.8500 |
| mnist_small_hidden       | wide 128-unit hidden representation         |      5 |  0.1 |      0 |   128 |          0.1130 |        0.7380 |          0.7690 |
| mnist_slow_learning_rate | aggressive learning rate 0.1                |      5 | 0.01 |      0 |   128 |          0.1400 |        0.4340 |          0.4520 |
| mnist_full_batch         | mini-batch SGD update frequency             |      5 |  0.1 |      0 |  1000 |          0.1400 |        0.4950 |          0.5190 |
| mnist_small_normal_init  | He initialization for ReLU layers           |      5 |  0.1 |      0 |   128 |          0.1400 |        0.4110 |          0.4830 |
| mnist_no_pixel_scaling   | pixel scaling to [0, 1]                     |      1 |  0.1 |      0 |   128 |          0.1400 |        0.1090 |          0.0960 |
| mnist_l2_regularization  | unregularized objective                     |      5 |  0.1 | 0.0001 |   128 |          0.1400 |        0.8180 |          0.8480 |
| mnist_current_mlp        | none                                        |      5 |  0.1 |      0 |   128 |          0.1400 |        0.8180 |          0.8480 |

## 6. 结果解读

### 6.1 初始准确率

MNIST 的 `mnist_untrained_mlp` 验证准确率是 0.1400，接近 10 类随机猜测的量级。
iris 的初始准确率是 0.4667，不严格等于 1/3，是因为随机初始化网络不是均匀随机
猜标签；它会对某些类别有初始偏置。这个现象不表示模型已经训练过。

### 6.2 输入缩放是必要优化

MNIST 移除 `pixel / 255.0` 后，使用同样学习率 0.1 训练 1 epoch，验证准确率从
初始 0.1400 降到 0.1090，训练准确率也只有 0.0960。原因是输入尺度变为 0-255 后，
梯度更新尺度过大，当前学习率不再合适。

iris 移除标准化后，验证准确率从当前 0.9667 降到 0.9000。iris 的特征维度少，
数值尺度差异没有 MNIST 那么极端，所以影响较小。

### 6.3 学习率解释了 MNIST 上升很快

在 MNIST quick profile 中，当前模型 5 epoch 验证准确率为 0.8180；只把学习率从
0.1 降到 0.01，验证准确率降到 0.4340。说明当前实现的快速上升主要来自输入缩放后
允许使用较大的学习率，而不是高级优化器。

### 6.4 mini-batch 比 full-batch 快很多

MNIST quick profile 中，batch size 128 的当前模型达到 0.8180；full-batch 版本
只有 0.4950。二者 epoch 数相同，但 mini-batch 在一个 epoch 内做了多次参数更新，
full-batch 每个 epoch 只更新一次。

### 6.5 He 初始化对 MNIST 收敛很关键

保持 128 隐藏单元、学习率 0.1、batch size 128 不变，只把 He 初始化换成很小的
`N(0, 0.01)`，MNIST 验证准确率从 0.8180 降到 0.4110。小初始化会让 ReLU 隐藏层
早期激活和梯度信号偏弱，收敛明显变慢。

iris 的同一消融得到 1.0000 验证准确率，不能说明小初始化更好。iris 验证集只有
30 个样本，1 个样本的变化就是 3.33 个百分点，单次 seed 的波动很大；对 iris 应
更多关注趋势和多 seed 平均，而不是单次最高值。

### 6.6 隐藏层不是所有设置下都立刻赢

MNIST quick profile 中，线性 softmax 达到 0.8340，略高于当前 MLP 的 0.8180。
这是 1000 样本、5 epoch 的快速实验，不代表 MLP 表达能力更差。线性模型参数更少，
短训练和小样本下可能更快收敛。完整 MNIST 数据上，当前主模型 5 epoch 的一次运行为：

```bash
uv run python main.py --dataset mnist --epochs 5 --quiet --output /private/tmp/current_mnist_5epochs.json
```

输出：

```text
mnist: train_acc=0.9564, val_acc=0.9467
```

### 6.7 L2 正则化是目标函数层面的扩展

quick profile 中，`iris_l2_regularization` 与当前 iris MLP 的验证准确率同为 0.9667；
`mnist_l2_regularization` 与当前 MNIST MLP 的验证准确率同为 0.8180。这个结果说明
当前选取的 L2 系数较温和，在小样本 quick 设置下不会明显改变准确率。

报告中不应把单次 quick 结果解读为“L2 一定无效”。更合理的说法是：L2 正则化改变了
优化目标，理论上会抑制过大的权重；它的实际效果依赖数据规模、训练轮数、模型容量和
`λ` 的选择。后续如果要更严谨比较，可以在完整 MNIST 或多 seed 设置下扫描不同
`λ`，例如 `1e-5`、`1e-4`、`1e-3`。

### 6.8 L2 参数扫描结果

下面结果由命令生成：

```bash
uv run python src/classification_experiments.py \
  --suite l2-sweep \
  --dataset all \
  --profile quick \
  --output results/l2_sweep.json \
  --markdown-output results/l2_sweep.md
```

| Experiment | Ablation | Epochs | LR | L2 | Batch | Initial val acc | Final val acc | Final train acc |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| iris_l2_0 | L2 regularization strength | 100 | 0.05 | 0 | 16 | 0.4667 | 0.9667 | 0.9667 |
| iris_l2_0.0001 | L2 regularization strength | 100 | 0.05 | 0.0001 | 16 | 0.4667 | 0.9667 | 0.9667 |
| iris_l2_0.001 | L2 regularization strength | 100 | 0.05 | 0.001 | 16 | 0.4667 | 0.9667 | 0.9667 |
| iris_l2_0.01 | L2 regularization strength | 100 | 0.05 | 0.01 | 16 | 0.4667 | 0.9667 | 0.9583 |
| iris_l2_0.1 | L2 regularization strength | 100 | 0.05 | 0.1 | 16 | 0.4667 | 0.9667 | 0.9500 |
| mnist_l2_0 | L2 regularization strength | 5 | 0.1 | 0 | 128 | 0.1400 | 0.8180 | 0.8480 |
| mnist_l2_1e-05 | L2 regularization strength | 5 | 0.1 | 1e-05 | 128 | 0.1400 | 0.8180 | 0.8480 |
| mnist_l2_0.0001 | L2 regularization strength | 5 | 0.1 | 0.0001 | 128 | 0.1400 | 0.8180 | 0.8480 |
| mnist_l2_0.001 | L2 regularization strength | 5 | 0.1 | 0.001 | 128 | 0.1400 | 0.8190 | 0.8470 |
| mnist_l2_0.01 | L2 regularization strength | 5 | 0.1 | 0.01 | 128 | 0.1400 | 0.8160 | 0.8470 |
| mnist_l2_0.1 | L2 regularization strength | 5 | 0.1 | 0.1 | 128 | 0.1400 | 0.8010 | 0.8200 |

在这组 quick 实验中，iris 验证准确率对 `λ` 不敏感，主要原因是验证集只有 30 个样本，
单个样本就对应 3.33 个百分点。MNIST 上 `λ=1e-3` 的验证准确率略高于未正则化版本，
但差距只有 0.001，不能作为显著提升结论；`λ=0.1` 时训练准确率和验证准确率都下降，
说明正则化过强会限制模型拟合能力。报告中可以把这组结果解释为：温和 L2 基本保持
性能，过大的 L2 会导致欠拟合。

### 6.9 学习率扫描结果

结果文件：

```text
results/lr_sweep.json
results/lr_sweep.md
```

| Experiment | LR | Final val acc | Final train acc |
|---|---:|---:|---:|
| iris_lr_0.001 | 0.001 | 0.7333 | 0.8250 |
| iris_lr_0.01 | 0.01 | 0.9333 | 0.9083 |
| iris_lr_0.05 | 0.05 | 0.9667 | 0.9667 |
| iris_lr_0.1 | 0.1 | 0.9667 | 0.9583 |
| mnist_lr_0.001 | 0.001 | 0.1750 | 0.1340 |
| mnist_lr_0.01 | 0.01 | 0.4340 | 0.4520 |
| mnist_lr_0.1 | 0.1 | 0.8180 | 0.8480 |
| mnist_lr_0.2 | 0.2 | 0.8080 | 0.8720 |

学习率过小时收敛明显变慢，尤其是 MNIST quick 中 `0.001` 和 `0.01` 都远低于
`0.1`。`0.2` 的训练准确率更高，但验证准确率略低于 `0.1`，说明步长继续增大不一定
带来更好的泛化表现。

### 6.10 隐藏层宽度扫描结果

结果文件：

```text
results/width_sweep.json
results/width_sweep.md
```

| Experiment | Hidden units | Final val acc | Final train acc |
|---|---:|---:|---:|
| iris_width_4 | 4 | 0.9333 | 0.9583 |
| iris_width_8 | 8 | 1.0000 | 0.9667 |
| iris_width_16 | 16 | 0.9667 | 0.9667 |
| iris_width_32 | 32 | 0.9333 | 0.9583 |
| mnist_width_32 | 32 | 0.8330 | 0.8660 |
| mnist_width_64 | 64 | 0.8220 | 0.8530 |
| mnist_width_128 | 128 | 0.8180 | 0.8480 |
| mnist_width_256 | 256 | 0.8400 | 0.8720 |

IRIS 数据量很小，宽度带来的单次结果波动较明显，`8` 个隐藏单元在本次划分上达到
1.0000 验证准确率，但不能单独说明它一定优于 `16`。MNIST quick 中 `256` 个隐藏单元
验证准确率最高，说明更宽模型在该设置下有一定容量优势。

### 6.11 训练策略对比结果

结果文件：

```text
results/training_strategies.json
results/training_strategies.md
```

| Experiment | Strategy | Final val acc | Final train acc |
|---|---|---:|---:|
| iris_strategy_sgd | SGD | 0.9667 | 0.9667 |
| iris_strategy_momentum | momentum | 0.9667 | 0.9833 |
| iris_strategy_lr_decay | LR decay | 0.9667 | 0.9583 |
| iris_strategy_early_stopping | early stopping | 0.9667 | 0.9750 |
| mnist_strategy_sgd | SGD | 0.8180 | 0.8480 |
| mnist_strategy_momentum | momentum | 0.8580 | 0.9070 |
| mnist_strategy_lr_decay | LR decay | 0.8130 | 0.8300 |
| mnist_strategy_early_stopping | early stopping | 0.8710 | 0.9390 |

MNIST quick 中，momentum 和 early stopping 设置的验证准确率高于普通 5-epoch SGD。
这里 early stopping 的实验上限是 20 epoch，因此提升部分也来自更长训练；报告中应把它
解释为“带验证监控的更长训练流程”，而不是只归因于提前停止机制本身。

### 6.12 曲线与混淆矩阵输出

报告图表由 `src/plot_results.py report-figures` 生成，当前保留在 `results/figures/`
中的主要图包括方法进化、累积消融、损失函数、L2、学习率、隐藏层宽度、训练策略和
最终 tuning trade-off。

混淆矩阵输出已生成到：

```text
results/figures/iris_mlp_iris_val_confusion.png
results/tables/iris_mlp_iris_val_confusion.csv
results/figures/mnist_mlp_mnist_val_confusion.png
results/tables/mnist_mlp_mnist_val_confusion.csv
```

当前混淆矩阵对应的 checkpoint 评估结果为：IRIS 验证集 `acc=0.9667`，MNIST quick
验证集前 1000 张 `acc=0.8180`。这些图表适合放入报告的 Results and Analysis 部分。

### 6.13 损失函数对比结果

结果文件：

```text
results/loss_comparison.json
results/loss_comparison.md
```

对比设置保持一层隐藏层 MLP 和 mini-batch SGD 不变，只改变输出层与损失函数：

- `Sigmoid + MSE`：每个输出节点独立 sigmoid，使用 one-hot 标签的均方误差。
- `Softmax + Cross-Entropy`：输出类别概率分布，使用多分类交叉熵。

| Experiment | Loss setup | Final val acc | Final train acc |
|---|---|---:|---:|
| iris_sigmoid_mse | Sigmoid + MSE | 0.9333 | 0.9000 |
| iris_softmax_cross_entropy | Softmax + Cross-Entropy | 0.9667 | 0.9667 |
| mnist_sigmoid_mse | Sigmoid + MSE | 0.4230 | 0.5010 |
| mnist_softmax_cross_entropy | Softmax + Cross-Entropy | 0.8180 | 0.8480 |

结果显示，`Softmax + Cross-Entropy` 在两个数据集上都优于 `Sigmoid + MSE`，MNIST
差距尤其明显。原因是 softmax 将多分类输出建模为一个归一化概率分布，交叉熵直接
优化正确类别概率；而 sigmoid+MSE 更像把多分类拆成多个独立回归目标，梯度信号通常
不如交叉熵适合分类任务。

## 7. 加载参数后的验证集评估示例

为了避免只看训练过程中打印出来的曲线，下面给出一个“训练保存 -> 重新加载 -> 验证集
评估”的完整例子。这个流程会启动两个独立进程，第二个进程只从磁盘 checkpoint 恢复
参数，然后重新读取 `splits/` 中定义的验证集。

训练 MNIST quick 模型并保存：

```bash
uv run python main.py \
  --dataset mnist \
  --mnist-limit 1000 \
  --epochs 100 \
  --model-output /private/tmp/mnist_model.npz \
  --output /private/tmp/mnist_train_save_metrics.json \
  --quiet
```

输出：

```text
mnist: train_acc=0.6210, val_acc=0.6070
model saved to /private/tmp/mnist_model.npz
metrics saved to /private/tmp/mnist_train_save_metrics.json
```

重新加载 checkpoint 并在验证集上评估：

```bash
uv run python main.py \
  --load-model /private/tmp/mnist_model.npz \
  --eval-split val \
  --output /private/tmp/mnist_loaded_val_metrics.json \
  --quiet
```

输出：

```text
mnist val: loss=1.6757, acc=0.6070, count=1000
metrics saved to /private/tmp/mnist_loaded_val_metrics.json
```

加载评估的 JSON 结果：

```json
[
  {
    "dataset": "mnist",
    "split": "val",
    "model_path": "/private/tmp/mnist_model.npz",
    "count": 1000,
    "loss": 1.6756697342535385,
    "accuracy": 0.607,
    "config": {
      "epochs": 1,
      "hidden_dim": 128,
      "learning_rate": 0.1,
      "batch_size": 128,
      "input_dim": 784,
      "output_dim": 10,
      "l2": 0.0,
      "seed": 42,
      "mnist_limit": 1000
    }
  }
]
```

这个验证集准确率和训练阶段最后记录的 `final_val_accuracy=0.6070` 一致，说明保存的
参数、预处理元数据和验证集加载逻辑可以复现实验结果。训练 JSON 中同时包含
`history["loss"]` / `history["accuracy"]` 和 `history["val_loss"]` /
`history["val_accuracy"]`，因此报告中应同时画训练曲线与验证曲线，而不是只看训练
loss 或训练准确率。

## 8. MNIST full 最终测试结果

quick profile 只用于快速比较趋势。最终模型选择后，需要重新用完整 MNIST 训练集划分
训练，并在官方 t10k 测试集上评估一次。当前选择的 MLP 配置为：

```text
784 -> ReLU(256) -> softmax(10)
optimizer = momentum
learning_rate = 0.05
momentum = 0.9
l2 = 0
epochs = 5
batch_size = 128
```

训练命令：

```bash
uv run python main.py \
  --dataset mnist \
  --epochs 5 \
  --hidden-dim 256 \
  --learning-rate 0.05 \
  --batch-size 128 \
  --optimizer momentum \
  --momentum 0.9 \
  --l2 0 \
  --model-output results/models/final_mnist_mlp.npz \
  --output results/final_mnist_train_metrics.json \
  --quiet
```

测试命令：

```bash
uv run python main.py \
  --load-model results/models/final_mnist_mlp.npz \
  --eval-split test \
  --output results/final_mnist_test_metrics.json \
  --quiet
```

当前结果：

| Split | Accuracy | Loss | Count |
|---|---:|---:|---:|
| Train | 0.9885 | 0.0432 | 48000 |
| Validation | 0.9742 | 0.0889 | 12000 |
| Test | 0.9761 | 0.0783 | 10000 |

结果文件：

- [final_mnist_summary.md](/home/cdf/optimization/MNIST-IRIS-Classification/results/final_mnist_summary.md)
- [final_mnist_train_metrics.json](/home/cdf/optimization/MNIST-IRIS-Classification/results/final_mnist_train_metrics.json)
- [final_mnist_test_metrics.json](/home/cdf/optimization/MNIST-IRIS-Classification/results/final_mnist_test_metrics.json)
- [final_mnist_mlp.npz](/home/cdf/optimization/MNIST-IRIS-Classification/results/models/final_mnist_mlp.npz)

## 9. 建议报告写法

报告中可以把当前模型描述为：

> 本项目从最简单的线性 softmax 分类器出发，逐步加入输入缩放/标准化、单隐藏层
> ReLU、适合 ReLU 的 He 初始化、mini-batch SGD 和较大的学习率。消融实验显示，
> MNIST 上快速收敛主要来自输入缩放、mini-batch 更新频率、He 初始化和学习率设置；
> iris 数据集较小，线性模型已能取得较高准确率，因此隐藏层带来的提升不如 MNIST
> 明显。

如果需要做更严谨的最终实验，建议把 `classification_experiments.py` 扩展为多 seed
运行，报告平均值和标准差，而不是只报告单次 seed=42。
