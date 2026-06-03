# CNN Extension for MNIST

本文档记录 MNIST 上的卷积神经网络扩展。CNN 不是基础主线模型，主线仍然是
[classification_nn.py](/home/cdf/optimization/MNIST-IRIS-Classification/src/classification_nn.py:1)
中的 MLP；CNN 作为 PPT 中 “Convolutional neural network (CNN)” 扩展方向的一个
from-scratch 实现示例。

## 模型结构

实现文件：

[src/cnn_mnist.py](/home/cdf/optimization/MNIST-IRIS-Classification/src/cnn_mnist.py:1)

当前 CNN 结构：

```text
Input: 28x28x1
-> Conv 3x3, 8 filters
-> ReLU
-> MaxPool 2x2
-> Flatten
-> Dense 10
-> Softmax
-> Cross-Entropy
```

训练仍然使用 mini-batch SGD，所有前向传播、反向传播、卷积、池化和参数更新均使用
NumPy 手写，没有使用 PyTorch、TensorFlow 或 sklearn。

## 运行命令

quick 设置使用 MNIST 训练集前 1000 张和验证集前 1000 张：

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

## 当前结果

结果文件：

[cnn_mnist_metrics.json](/home/cdf/optimization/MNIST-IRIS-Classification/results/cnn_mnist_metrics.json)

quick 训练结果：

| Model | Train samples | Val samples | Epochs | Filters | Val accuracy | Train accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Simple CNN | 1000 | 1000 | 3 | 8 | 0.7970 | 0.8240 |

## 结果解读

这个 CNN 版本证明我们已经从零实现了图像任务中更自然的局部特征提取结构。当前 quick
结果略低于同样 quick 设置下的 MLP 主模型 `0.8180`，原因包括：

- CNN 只训练 3 epoch，而 MLP quick 通常训练 5 epoch。
- 当前 CNN 只有一个卷积层和 8 个 filters，容量较小。
- 纯 NumPy 卷积实现较慢，因此这里优先保持结构简单和可解释。

报告中建议把 CNN 写作扩展探索，而不是最终主模型。最终主线仍然以 MLP 的系统优化实验
为主，CNN 用于展示我们尝试了更适合图像的网络结构。

## 后续可扩展方向

如果时间允许，可以继续尝试：

- 增加 filters，例如 16 或 32。
- 训练更多 epoch。
- 加入第二个卷积层。
- 对完整 MNIST 训练/验证划分运行 CNN。

这些都应保持在 `src/cnn_mnist.py` 里，不要混入 MLP 主线代码。
