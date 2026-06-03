# 最优化理论 Project 3 (2026)

厦门大学《最优化理论》课程第三次大作业。基于 iris 与 MNIST 数据集实现并比较最优化算法。

## 环境配置

本项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python 环境与依赖。Python 版本固定为 **3.12**（见 `.python-version`）。

首次克隆仓库后，每位成员在**自己的电脑上**执行一次：

```bash
# 1. 安装 uv（如未安装）
#    macOS / Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh
#    Windows (PowerShell):
#    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 创建虚拟环境并安装全部依赖（依据 uv.lock 精确还原）
uv sync
```

`uv sync` 会自动下载 Python 3.12、创建 `.venv/` 并安装锁定版本的依赖。`.venv/` 是本机专属、已被 `.gitignore` 忽略，**请勿提交**。

运行代码：

```bash
uv run python main.py          # 直接运行脚本
uv run python src/xxx.py       # 运行 src 下的实现
```

训练简易神经网络分类器：

```bash
# 训练 iris 与 MNIST（默认写出 results/classification_nn_metrics.json）
uv run python main.py --dataset all

# 快速检查 MNIST 流程，可只取一部分样本
uv run python main.py --dataset mnist --mnist-limit 1000 --epochs 2

# 只训练 iris
uv run python main.py --dataset iris

# 只查看训练前基线准确率（epoch 000）
uv run python main.py --dataset all --mnist-limit 1000 --epochs 0

# 使用 L2 正则化训练，并将结果单独保存
uv run python main.py --dataset all --l2 0.001 \
  --output results/classification_nn_l2_metrics.json

# 扫描多组 L2 正则化系数，输出 JSON
uv run python src/classification_experiments.py \
  --suite l2-sweep \
  --dataset all \
  --profile quick \
  --output results/l2_sweep.json

# 学习率、隐藏层宽度和训练策略扫描
uv run python src/classification_experiments.py --suite lr-sweep \
  --dataset all --profile quick --output results/lr_sweep.json
uv run python src/classification_experiments.py --suite width-sweep \
  --dataset all --profile quick --output results/width_sweep.json
uv run python src/classification_experiments.py --suite training-strategies \
  --dataset all --profile quick --output results/training_strategies.json
uv run python src/classification_experiments.py --suite loss-comparison \
  --dataset all --profile quick --output results/loss_comparison.json

# 从训练 JSON 画 loss / accuracy 曲线
uv run python src/plot_results.py curves \
  --metrics results/classification_nn_metrics.json \
  --output-dir results/learning_curves

# 训练并保存参数、预处理信息
uv run python main.py --dataset mnist --mnist-limit 1000 --epochs 1 \
  --model-output results/models/mnist_mlp.npz

# 重新加载参数，在验证集上独立评估
uv run python main.py --load-model results/models/mnist_mlp.npz --eval-split val

# 训练当前选择的 MNIST full 模型，并保存 checkpoint
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

# 加载 checkpoint，在 MNIST 官方 t10k 测试集上做最终评估
uv run python main.py \
  --load-model results/models/final_mnist_mlp.npz \
  --eval-split test \
  --output results/final_mnist_test_metrics.json \
  --quiet

# 用保存的模型生成混淆矩阵 PDF 和 CSV 表格
uv run python src/plot_results.py confusion \
  --model results/models/mnist_mlp.npz \
  --split val \
  --mnist-limit 1000

# MNIST CNN 扩展示例
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

当前模型位于 `src/classification_nn.py`，使用 NumPy 实现两层 MLP
（输入层 -> ReLU 隐藏层 -> softmax 输出层），并复用 `src/data_split.py`
中的 iris / MNIST 训练集和验证集划分；MNIST 官方 t10k 测试集只用于最终测试。
训练日志从 `epoch 000` 开始，
表示参数初始化后、尚未做梯度更新时的基线准确率。
可通过 `--l2` 为 MLP 权重加入 L2 正则化惩罚项，用于比较未正则化目标和
正则化目标的训练/验证表现。实验入口还支持学习率扫描、隐藏层宽度扫描、
momentum、学习率衰减、early stopping 和损失函数对比。
模型优化与消融实验说明见 `docs/classification_nn_experiments.md`。
从 baseline 到优化模型的演进路线见 `docs/experiment_roadmap.md`。
按 roadmap 分组的版本化实验入口位于 `versions/`，每个 version 有独立目录和 `run.py`。
MNIST CNN 扩展说明见 `docs/cnn_extension.md`。
当前选定 MNIST full MLP 结果见 `results/final_mnist_summary.md`：验证集准确率
`0.9742`，官方测试集准确率 `0.9761`。

新增依赖（会同时更新 `pyproject.toml` 和 `uv.lock`，记得提交这两个文件）：

```bash
uv add <package-name>
```

## 目录结构

```
Project_3_2026/
├── pyproject.toml      # 项目元数据与依赖声明
├── uv.lock             # 锁定的依赖版本（保证全员环境一致，需提交）
├── .python-version     # 固定 Python 版本 3.12
├── main.py             # 入口示例
├── src/                # 算法实现与共享代码
├── versions/           # 按 roadmap 拆分的 V0/V1/V2/V3/V4/Final 实验入口
├── notebooks/          # Jupyter 实验记录
├── results/            # 图表与实验输出（默认不纳入版本控制）
├── iris/               # iris 数据集
├── mnist/              # MNIST 数据集（idx 格式 .gz）
└── Porject3_Assignment_2026.pptx  # 课程作业要求
```

按版本运行示例：

```bash
uv run python versions/v0_untrained_random/run.py --dataset all --profile quick
uv run python versions/v1_linear_softmax/run.py --dataset all --profile quick
uv run python versions/v2_small_mlp/run.py --dataset all --profile quick
uv run python versions/v3_stable_mlp/run.py --dataset all --profile quick
uv run python versions/v4_component_checks/run.py --dataset all --profile quick
uv run python versions/final_selected_model/run.py --mnist-limit 1000 --epochs 5 --skip-test
```

## Git 协作流程

为避免冲突，约定如下分支工作流：

- `main` 分支保持随时可运行，不直接在其上提交。
- 每个功能/每人新建独立分支：`git checkout -b feat/<姓名或功能>`。
- 完成后推送分支并在 GitHub 上发起 Pull Request，经至少一位队友 review 后合并到 `main`。
- 开始工作前先同步：`git pull --rebase origin main`。

常用命令：

```bash
git checkout -b feat/my-task     # 新建并切换到功能分支
git add -A && git commit -m "描述本次改动"
git push -u origin feat/my-task  # 首次推送分支
```

提交信息建议简洁说明「做了什么」，例如 `实现梯度下降在 iris 上的分类`。

## 注意事项

- 不要提交 `.venv/`、`__pycache__/`、`.DS_Store` 等（已在 `.gitignore` 中）。
- 修改依赖后务必把 `pyproject.toml` 与 `uv.lock` 一起提交。
- `results/` 默认被忽略；如需共享某些结果，可在 `.gitignore` 中调整或手动 `git add -f`。
