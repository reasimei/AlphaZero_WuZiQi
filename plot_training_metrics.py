# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_records(metrics_path):
    records = []
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_single_curve(x, y, title, xlabel, ylabel, output_path, color):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, color=color, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from training_metrics.jsonl")
    parser.add_argument("--run-dir", required=True, help="Directory containing training_metrics.jsonl")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    metrics_path = run_dir / "training_metrics.jsonl"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    records = load_records(metrics_path)
    if not records:
        raise SystemExit("No records found in {}".format(metrics_path))

    plt.style.use("dark_background")

    batches = [record["batch"] for record in records]
    losses = [record["loss"] for record in records if "loss" in record]
    entropies = [record["entropy"] for record in records if "entropy" in record]
    kls = [record["kl"] for record in records if "kl" in record]
    lr_multipliers = [record["lr_multiplier"] for record in records if "lr_multiplier" in record]
    episode_lens = [record["episode_len"] for record in records]
    batch_times = [record["batch_time_sec"] for record in records]

    save_single_curve(
        batches[:len(losses)],
        losses,
        "Training Loss",
        "Batch",
        "Loss",
        plot_dir / "loss_curve.png",
        "#5dade2",
    )
    save_single_curve(
        batches[:len(entropies)],
        entropies,
        "Policy Entropy",
        "Batch",
        "Entropy",
        plot_dir / "policy_entropy_curve.png",
        "#58d68d",
    )
    save_single_curve(
        batches[:len(kls)],
        kls,
        "KL Divergence",
        "Batch",
        "KL",
        plot_dir / "kl_curve.png",
        "#f5b041",
    )
    save_single_curve(
        batches[:len(lr_multipliers)],
        lr_multipliers,
        "Learning Rate Multiplier",
        "Batch",
        "LR Multiplier",
        plot_dir / "lr_multiplier_curve.png",
        "#af7ac5",
    )
    save_single_curve(
        batches,
        episode_lens,
        "Self-play Episode Length",
        "Batch",
        "Episode Length",
        plot_dir / "episode_length_curve.png",
        "#ec7063",
    )
    save_single_curve(
        batches,
        batch_times,
        "Batch Time",
        "Batch",
        "Time (s)",
        plot_dir / "batch_time_curve.png",
        "#76d7c4",
    )

    eval_records = [record for record in records if record.get("win_ratio") is not None]
    if eval_records:
        save_single_curve(
            [record["batch"] for record in eval_records],
            [record["win_ratio"] for record in eval_records],
            "Evaluation Win Ratio vs Pure MCTS",
            "Batch",
            "Win Ratio",
            plot_dir / "win_ratio_curve.png",
            "#f1948a",
        )

    summary = {
        "run_dir": str(run_dir),
        "num_records": len(records),
        "plot_dir": str(plot_dir),
        "generated_files": sorted([path.name for path in plot_dir.glob("*.png")]),
    }
    with open(plot_dir / "plot_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
