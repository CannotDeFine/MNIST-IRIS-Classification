# 简易神经网络分类模型与消融实验说明

本文档说明当前 `src/classification_nn.py` 中的 iris / MNIST 分类模型从最简单神经网络
到当前实现经历了哪些优化，并给出可复现的消融实验实现、运行命令和参考结果。

## 1. 当前模型范围

当前实现只使用 NumPy，没有使用 PyTorch、TensorFlow、Adam、Momentum、CNN、
BatchNorm 或数据增强。训练算法是 mini-batch SGD，损失函数是 softmax 交叉熵。

当前主入口：

```bash
uv run python main.py --dataset all
```

对应默认配置如下：

| 数据集 | 输入                      | 预处理                          | 模型                                | 默认 epoch | 学习率 | batch size |
| ------ | ------------------------- | ------------------------------- | ----------------------------------- | ---------: | -----: | ---------: |
| iris   | 4 个数值特征              | 用训练集均值/标准差标准化       | `4 -> ReLU(16) -> softmax(3)`     |        300 |   0.05 |         16 |
| MNIST  | 28x28 灰度图展平为 784 维 | 像素除以 255，缩放到 `[0, 1]` | `784 -> ReLU(128) -> softmax(10)` |          5 |    0.1 |        128 |

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

## 3. 消融实验实现

消融脚本位于：

```text
src/classification_experiments.py
```

它实现了：

- `SoftmaxRegressionClassifier`：线性 softmax 基线。
- `ExperimentConfig`：每个实验的模型、学习率、batch size、预处理和初始化配置。
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
会使用完整验证划分。

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

| Experiment               | Ablation                                    | Epochs |   LR | Batch | Initial val acc | Final val acc | Final train acc |
| ------------------------ | ------------------------------------------- | -----: | ---: | ----: | --------------: | ------------: | --------------: |
| iris_untrained_mlp       | all optimization steps after initialization |      0 | 0.05 |    16 |          0.4667 |        0.4667 |          0.4583 |
| iris_linear_softmax      | hidden ReLU layer                           |    100 | 0.05 |    16 |          0.7333 |        0.9667 |          0.9417 |
| iris_small_hidden        | 16-unit hidden representation               |    100 | 0.05 |    16 |          0.3333 |        0.9333 |          0.9583 |
| iris_no_standardization  | feature standardization                     |    100 | 0.05 |    16 |          0.3333 |        0.9000 |          0.9500 |
| iris_small_normal_init   | He initialization for ReLU layers           |    100 | 0.05 |    16 |          0.4667 |        1.0000 |          0.9667 |
| iris_current_mlp         | none                                        |    100 | 0.05 |    16 |          0.4667 |        0.9667 |          0.9667 |
| mnist_untrained_mlp      | all optimization steps after initialization |      0 |  0.1 |   128 |          0.1400 |        0.1400 |          0.1020 |
| mnist_linear_softmax     | hidden ReLU layer and hidden width          |      5 |  0.1 |   128 |          0.1120 |        0.8340 |          0.8500 |
| mnist_small_hidden       | wide 128-unit hidden representation         |      5 |  0.1 |   128 |          0.1130 |        0.7380 |          0.7690 |
| mnist_slow_learning_rate | aggressive learning rate 0.1                |      5 | 0.01 |   128 |          0.1400 |        0.4340 |          0.4520 |
| mnist_full_batch         | mini-batch SGD update frequency             |      5 |  0.1 |  1000 |          0.1400 |        0.4950 |          0.5190 |
| mnist_small_normal_init  | He initialization for ReLU layers           |      5 |  0.1 |   128 |          0.1400 |        0.4110 |          0.4830 |
| mnist_no_pixel_scaling   | pixel scaling to [0, 1]                     |      1 |  0.1 |   128 |          0.1400 |        0.1090 |          0.0960 |
| mnist_current_mlp        | none                                        |      5 |  0.1 |   128 |          0.1400 |        0.8180 |          0.8480 |

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

## 8. 建议报告写法

报告中可以把当前模型描述为：

> 本项目从最简单的线性 softmax 分类器出发，逐步加入输入缩放/标准化、单隐藏层
> ReLU、适合 ReLU 的 He 初始化、mini-batch SGD 和较大的学习率。消融实验显示，
> MNIST 上快速收敛主要来自输入缩放、mini-batch 更新频率、He 初始化和学习率设置；
> iris 数据集较小，线性模型已能取得较高准确率，因此隐藏层带来的提升不如 MNIST
> 明显。

如果需要做更严谨的最终实验，建议把 `classification_experiments.py` 扩展为多 seed
运行，报告平均值和标准差，而不是只报告单次 seed=42。
