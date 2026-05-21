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
├── notebooks/          # Jupyter 实验记录
├── results/            # 图表与实验输出（默认不纳入版本控制）
├── iris/               # iris 数据集
├── mnist/              # MNIST 数据集（idx 格式 .gz）
└── Porject3_Assignment_2026.pptx  # 课程作业要求
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
