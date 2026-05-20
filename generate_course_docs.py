# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle, Circle


REPO_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = REPO_ROOT / "assets"
DOCS_DIR = REPO_ROOT / "docs"
OUTPUT_DIR = REPO_ROOT / "course_outputs"


def ensure_dirs():
    ASSETS_DIR.mkdir(exist_ok=True)
    DOCS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)


def load_experiment_summary():
    summary_path = OUTPUT_DIR / "experiment_summary.json"
    if summary_path.exists():
        return json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        "baseline": {
            "final_loss": None,
            "best_win_ratio": None,
            "last_win_ratio": None,
            "avg_episode_len": None,
        },
        "reward_shaping": {
            "final_loss": None,
            "best_win_ratio": None,
            "last_win_ratio": None,
            "avg_episode_len": None,
            "avg_heuristic_reward": None,
            "total_open_three": None,
            "total_open_four": None,
            "total_five": None,
        },
        "comparison": {"loss_delta": None, "win_ratio_delta": None},
        "inference_benchmark": [],
    }


def draw_box(ax, xy, width, height, text, facecolor="#16202a", edgecolor="#4ea1ff", fontsize=12):
    rect = Rectangle(xy, width, height, linewidth=1.8, edgecolor=edgecolor, facecolor=facecolor)
    ax.add_patch(rect)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color="white",
        fontsize=fontsize,
        wrap=True,
    )


def draw_arrow(ax, start, end, color="#6cb6ff"):
    arrow = FancyArrowPatch(start, end, arrowstyle="->", mutation_scale=18, linewidth=2, color=color)
    ax.add_patch(arrow)


def build_static_assets():
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    draw_box(ax, (0.05, 0.62), 0.2, 0.2, "Game\nEnvironment")
    draw_box(ax, (0.32, 0.62), 0.2, 0.2, "MCTS\n(Policy Guided)")
    draw_box(ax, (0.59, 0.62), 0.2, 0.2, "Policy-Value\nNetwork")
    draw_box(ax, (0.18, 0.24), 0.22, 0.2, "Self-play\nData Buffer")
    draw_box(ax, (0.52, 0.24), 0.22, 0.2, "Training\nPipeline")
    draw_arrow(ax, (0.25, 0.72), (0.32, 0.72))
    draw_arrow(ax, (0.52, 0.72), (0.59, 0.72))
    draw_arrow(ax, (0.69, 0.62), (0.63, 0.44))
    draw_arrow(ax, (0.41, 0.24), (0.52, 0.24))
    draw_arrow(ax, (0.63, 0.44), (0.63, 0.62))
    draw_arrow(ax, (0.29, 0.44), (0.18, 0.62))
    ax.set_title("AlphaZero Gomoku Architecture", fontsize=20, color="white", pad=18)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "architecture_diagram.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")
    steps = [
        ("Board state", 0.04),
        ("MCTS search", 0.22),
        ("Policy / value", 0.40),
        ("Self-play move", 0.58),
        ("Replay buffer", 0.76),
        ("Network update", 0.90),
    ]
    for label, xpos in steps:
        draw_box(ax, (xpos, 0.38), 0.12, 0.24, label, fontsize=11)
    for i in range(len(steps) - 1):
        draw_arrow(ax, (steps[i][1] + 0.12, 0.5), (steps[i + 1][1], 0.5))
    ax.set_title("AlphaZero Working Flow", fontsize=20, color="white", pad=18)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "alpha_zero_workflow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.axis("off")
    draw_box(ax, (0.42, 0.76), 0.16, 0.12, "train.py", fontsize=15)
    children = [
        ("game.py", (0.08, 0.48)),
        ("mcts_alphaZero.py", (0.31, 0.48)),
        ("policy_value_net_pytorch.py", (0.56, 0.48)),
        ("heuristic_reward.py", (0.80, 0.48)),
    ]
    for label, xy in children:
        draw_box(ax, xy, 0.16, 0.12, label, fontsize=11)
        draw_arrow(ax, (0.50, 0.76), (xy[0] + 0.08, xy[1] + 0.12))
    draw_box(ax, (0.30, 0.18), 0.18, 0.12, "human_play.py", fontsize=11)
    draw_box(ax, (0.56, 0.18), 0.18, 0.12, "run_course_experiments.py", fontsize=11)
    draw_arrow(ax, (0.39, 0.48), (0.39, 0.30))
    draw_arrow(ax, (0.64, 0.48), (0.65, 0.30))
    ax.set_title("Module Relationship", fontsize=20, color="white", pad=18)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "module_relationship.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    positions = {
        "root": (0.5, 0.82),
        "s1": (0.25, 0.52),
        "s2": (0.50, 0.52),
        "s3": (0.75, 0.52),
        "l1": (0.15, 0.22),
        "l2": (0.35, 0.22),
        "l3": (0.50, 0.22),
        "l4": (0.65, 0.22),
        "l5": (0.85, 0.22),
    }
    for name, (x, y) in positions.items():
        circle = Circle((x, y), 0.045, facecolor="#16202a", edgecolor="#4ea1ff", linewidth=2)
        ax.add_patch(circle)
        label = "Root" if name == "root" else name.upper()
        ax.text(x, y, label, color="white", ha="center", va="center", fontsize=11)
    edges = [
        ("root", "s1"),
        ("root", "s2"),
        ("root", "s3"),
        ("s1", "l1"),
        ("s1", "l2"),
        ("s2", "l3"),
        ("s3", "l4"),
        ("s3", "l5"),
    ]
    for start, end in edges:
        draw_arrow(ax, positions[start], positions[end])
    ax.text(0.5, 0.92, "MCTS Search Tree Sketch", color="white", fontsize=20, ha="center")
    ax.text(0.25, 0.62, "Selection", color="#9ad0ff", fontsize=12, ha="center")
    ax.text(0.5, 0.62, "Expansion", color="#9ad0ff", fontsize=12, ha="center")
    ax.text(0.75, 0.62, "Simulation", color="#9ad0ff", fontsize=12, ha="center")
    ax.text(0.5, 0.10, "Backup propagates value back to ancestors", color="#9ad0ff", fontsize=12, ha="center")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "mcts_tree.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_markdown(summary):
    inference_rows = []
    for item in summary["inference_benchmark"]:
        inference_rows.append(
            "| {} | {:.4f} | {:.4f} |".format(
                item["n_playout"], item["avg_time_sec"], item["std_time_sec"]
            )
        )
    inference_table = "\n".join(inference_rows) if inference_rows else "| - | - | - |"

    tech_doc = f"""# AlphaZero_Gomoku 技术文档

## 1. 项目整体架构

项目由 5 个核心层组成：

1. `game.py`：负责棋盘状态、落子规则、胜负判定与自博弈回放。
2. `mcts_alphaZero.py`：负责 AlphaZero 风格的 MCTS 搜索。
3. `policy_value_net_pytorch.py`：负责策略-价值网络前向推理与训练。
4. `train.py`：负责自博弈采样、经验增强、网络更新和周期评估。
5. `heuristic_reward.py`：课程设计新增模块，用于启发式奖励 shaping。

![系统架构图](../assets/architecture_diagram.png)

## 2. AlphaZero 工作流程

AlphaZero 在本项目中的主链路是：

1. 从当前棋盘状态编码出神经网络输入。
2. 用策略-价值网络给出先验概率 `P(s, a)` 和局面价值 `V(s)`。
3. MCTS 用网络结果引导搜索，得到更稳健的访问分布。
4. 按访问分布采样或选择动作，推进自博弈。
5. 收集 `(state, mcts_prob, z)` 训练样本。
6. 对样本做旋转/翻转增强后更新网络。
7. 周期性与纯 MCTS 对弈，保存 `current_policy` 与 `best_policy`。

```mermaid
flowchart LR
    A["棋盘状态 s"] --> B["策略-价值网络"]
    B --> C["策略先验 P(s,a)"]
    B --> D["局面价值 V(s)"]
    C --> E["MCTS 搜索"]
    D --> E
    E --> F["动作分布 pi"]
    F --> G["自博弈落子"]
    G --> H["生成训练数据"]
    H --> I["数据增强"]
    I --> J["网络训练"]
    J --> B
```

![AlphaZero 工作流](../assets/alpha_zero_workflow.png)

## 3. MCTS 的四阶段

### 3.1 Selection

从根节点开始，按照 `Q + U` 最大原则递归选择子节点。其中：

- `Q`：历史平均价值
- `U`：由先验概率和访问次数共同决定的探索项

### 3.2 Expansion

当搜索到叶子节点后，调用策略-价值网络输出可行动作及其概率，并把这些动作扩展成新子节点。

### 3.3 Simulation

在本项目的 `mcts_alphaZero.py` 中，Simulation 并不是随机 rollout，而是直接使用价值网络输出的 `leaf_value` 作为叶子局面的评估结果。

### 3.4 Backup

把叶子节点估值沿着搜索路径逐层回传，并在父子节点之间交替取负号，从而体现双方博弈的零和属性。

```mermaid
flowchart TD
    A["Root"] --> B["Selection"]
    B --> C["Leaf"]
    C --> D["Expansion"]
    D --> E["Policy / Value Evaluation"]
    E --> F["Backup"]
    F --> A
```

![MCTS 搜索树示意图](../assets/mcts_tree.png)

## 4. 神经网络输入输出

### 输入

网络输入维度为 `4 x board_width x board_height`：

1. 当前玩家棋子分布
2. 对手棋子分布
3. 上一步落子位置
4. 当前是否轮到先手的标记平面

### 输出

网络有两个头：

1. `policy head`：输出每个位置的对数概率
2. `value head`：输出当前局面的胜负估计，范围约为 `[-1, 1]`

## 5. self-play 流程

1. 初始化棋盘
2. 用 `MCTSPlayer.get_action()` 获取动作和访问分布
3. 保存当前状态、MCTS 概率、当前执子方
4. 执行动作并检查终局
5. 若终局则生成 `winners_z`
6. 若开启课程创新点，则将启发式奖励加到 `winners_z` 上形成 shaped target
7. 返回训练样本并清空 MCTS 根节点

## 6. train.py 的主循环

`train.py` 的 `run()` 主循环可概括为：

1. 调用 `collect_selfplay_data()` 收集一批自博弈数据
2. 当缓冲区样本数超过 `batch_size` 后，调用 `policy_update()`
3. 每隔 `check_freq` 个 batch 与纯 MCTS 评估一次
4. 保存 `current_policy.model`
5. 若胜率更高，则刷新 `best_policy.model`
6. 将 batch 级指标写入 `training_metrics.jsonl`

## 7. 模块关系图

```mermaid
graph TD
    T["train.py"] --> G["game.py"]
    T --> M["mcts_alphaZero.py"]
    T --> N["policy_value_net_pytorch.py"]
    T --> H["heuristic_reward.py"]
    HP["human_play.py"] --> G
    HP --> M
    HP --> N
    EXP["run_course_experiments.py"] --> T
```

![模块关系图](../assets/module_relationship.png)

## 8. 课程设计创新点

本次优先实现了方案 A：**启发式奖励 shaping**。

### 设计目标

在不改动整体 AlphaZero 训练框架的前提下，为训练目标增加轻量级局面先验，使模型在早期训练阶段更快感知“五子棋局部进攻结构”。

### 实现方式

新增 `heuristic_reward.py`，在 `game.start_self_play()` 中对每一步自博弈落子后的局面做额外评分，并将该分数以小权重叠加到原始终局奖励 `z`：

`shaped_z = clip(z + shaping_weight * heuristic_bonus, -1, 1)`

当前启发式模式采用最小实现：

- 活三：`01110`
- 活四：`011110`
- 连五：`11111`

默认权重：

- 活三：`1.0`
- 活四：`2.0`
- 连五：`4.0`
- shaping 总系数：`0.15`

### 优点

1. 改动小，只新增一个模块并在 self-play 处插入可选逻辑
2. 可开关，便于与 baseline 做公平对比
3. 有稳定日志输出，可直接做课程实验

## 9. 实验设计

### 对比组

- Baseline：原始 PyTorch 训练流程，不开启启发式奖励
- Reward shaping：开启启发式奖励，其他超参数保持一致

### 训练设置

- 棋盘：`6x6`
- 胜利条件：`4` 子连线
- MCTS playout：`32`
- batch size：`32`
- game batch num：`8`
- 评估频率：每 `2` 个 batch
- 随机种子：`2026`

### 推理时间基准

对提供的 `8x8` 预训练模型测试不同 `playout` 下的单步思考时间。

## 10. 实验结果

### 10.1 训练曲线

![Loss 对比](../assets/experiment_loss.png)

![Win ratio 对比](../assets/experiment_win_ratio.png)

![Heuristic reward 曲线](../assets/experiment_heuristic_reward.png)

### 10.2 量化对比

| 指标 | Baseline | Reward shaping |
| --- | ---: | ---: |
| Final loss | {summary["baseline"]["final_loss"] if summary["baseline"]["final_loss"] is not None else "-"} | {summary["reward_shaping"]["final_loss"] if summary["reward_shaping"]["final_loss"] is not None else "-"} |
| Best win ratio | {summary["baseline"]["best_win_ratio"] if summary["baseline"]["best_win_ratio"] is not None else "-"} | {summary["reward_shaping"]["best_win_ratio"] if summary["reward_shaping"]["best_win_ratio"] is not None else "-"} |
| Last eval win ratio | {summary["baseline"]["last_win_ratio"] if summary["baseline"]["last_win_ratio"] is not None else "-"} | {summary["reward_shaping"]["last_win_ratio"] if summary["reward_shaping"]["last_win_ratio"] is not None else "-"} |
| Avg episode len | {summary["baseline"]["avg_episode_len"] if summary["baseline"]["avg_episode_len"] is not None else "-"} | {summary["reward_shaping"]["avg_episode_len"] if summary["reward_shaping"]["avg_episode_len"] is not None else "-"} |
| Avg heuristic reward | 0 | {summary["reward_shaping"]["avg_heuristic_reward"] if summary["reward_shaping"]["avg_heuristic_reward"] is not None else "-"} |
| Total open three | 0 | {summary["reward_shaping"]["total_open_three"] if summary["reward_shaping"]["total_open_three"] is not None else "-"} |
| Total open four | 0 | {summary["reward_shaping"]["total_open_four"] if summary["reward_shaping"]["total_open_four"] is not None else "-"} |
| Total five | 0 | {summary["reward_shaping"]["total_five"] if summary["reward_shaping"]["total_five"] is not None else "-"} |

### 10.3 结论

短程实验下，reward shaping 版本相比 baseline：

1. 能稳定输出非零启发式奖励日志；
2. 能显式统计活三/活四/连五等局面模式；
3. 在当前快速实验中，loss 与评估胜率变化幅度不大，但训练目标中已经注入了更丰富的五子棋结构信号；
4. 这类 shaping 更适合作为“前期学习加速器”，后续可继续扩大训练轮次观察长期收益。

## 11. 推理时间测试

![推理时间曲线](../assets/inference_time.png)

| Playout | Avg move time (s) | Std (s) |
| ---: | ---: | ---: |
{inference_table}

## 12. 总结与展望

本课程设计在原始 AlphaZero_Gomoku 项目的基础上，完成了以下工作：

1. 将训练后端切换并稳定到 Windows + Conda + PyTorch 路线；
2. 增加了可自动化执行的人机对战、训练与日志保存流程；
3. 引入了最小改动的启发式奖励 shaping 创新点；
4. 建立了可复现实验、图表生成、技术文档和课程答辩 PPT 的完整交付链路。

后续可以继续扩展：

1. 将活三/活四模式从“连续形态”扩展到“跳三/冲四”等更完整棋形；
2. 引入动态 `c_puct` 做 MCTS 自适应探索；
3. 为 MCTS 增加局面缓存，加速重复局面搜索。
"""

    experiment_doc = f"""# 启发式奖励实验结果

## 实验对象

- Baseline：不启用 reward shaping
- Reward shaping：启用活三/活四/连五奖励

## 关键结论

- Final loss 差值：{summary["comparison"]["loss_delta"] if summary["comparison"]["loss_delta"] is not None else "-"}
- Last win ratio 差值：{summary["comparison"]["win_ratio_delta"] if summary["comparison"]["win_ratio_delta"] is not None else "-"}
- shaping 平均启发式奖励：{summary["reward_shaping"]["avg_heuristic_reward"] if summary["reward_shaping"]["avg_heuristic_reward"] is not None else "-"}

## 图表

![Loss](../assets/experiment_loss.png)

![Win Ratio](../assets/experiment_win_ratio.png)

![Heuristic Reward](../assets/experiment_heuristic_reward.png)
"""

    (DOCS_DIR / "AlphaZero_Gomoku_技术文档.md").write_text(tech_doc, encoding="utf-8")
    (DOCS_DIR / "启发式奖励实验对比.md").write_text(experiment_doc, encoding="utf-8")


def main():
    ensure_dirs()
    build_static_assets()
    build_markdown(load_experiment_summary())
    print(str(DOCS_DIR / "AlphaZero_Gomoku_技术文档.md"))


if __name__ == "__main__":
    main()
