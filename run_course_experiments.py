# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from game import Board
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy


REPO_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = REPO_ROOT / "assets"
OUTPUT_DIR = REPO_ROOT / "course_outputs"
EXPERIMENT_DIR = OUTPUT_DIR / "experiments"


def ensure_dirs():
    ASSETS_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    EXPERIMENT_DIR.mkdir(exist_ok=True)


def load_metrics(path):
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def run_train_variant(name, extra_args, exp_args, force=False):
    save_dir = EXPERIMENT_DIR / name
    metrics_path = save_dir / "training_metrics.jsonl"
    summary_path = save_dir / "training_summary.json"
    if metrics_path.exists() and summary_path.exists() and not force:
        return load_metrics(metrics_path), json.loads(summary_path.read_text(encoding="utf-8"))

    cmd = [
        sys.executable,
        "train.py",
        "--backend",
        "pytorch",
        "--board-width",
        str(exp_args.board_width),
        "--board-height",
        str(exp_args.board_height),
        "--n-in-row",
        str(exp_args.n_in_row),
        "--n-playout",
        str(exp_args.n_playout),
        "--batch-size",
        str(exp_args.batch_size),
        "--buffer-size",
        str(exp_args.buffer_size),
        "--play-batch-size",
        str(exp_args.play_batch_size),
        "--epochs",
        str(exp_args.epochs),
        "--check-freq",
        str(exp_args.check_freq),
        "--game-batch-num",
        str(exp_args.game_batch_num),
        "--pure-mcts-playout-num",
        str(exp_args.pure_mcts_playout_num),
        "--eval-games",
        str(exp_args.eval_games),
        "--seed",
        str(exp_args.seed),
        "--save-dir",
        str(save_dir),
    ]
    cmd.extend(extra_args)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    return load_metrics(metrics_path), json.loads(summary_path.read_text(encoding="utf-8"))


def plot_training_curves(baseline_metrics, shaping_metrics, output_prefix="experiment"):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        [m["batch"] for m in baseline_metrics if "loss" in m],
        [m["loss"] for m in baseline_metrics if "loss" in m],
        marker="o",
        linewidth=2,
        color="#5dade2",
        label="Baseline",
    )
    ax.plot(
        [m["batch"] for m in shaping_metrics if "loss" in m],
        [m["loss"] for m in shaping_metrics if "loss" in m],
        marker="s",
        linewidth=2,
        color="#f5b041",
        label="Reward shaping",
    )
    ax.set_title("Training Loss Comparison")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / f"{output_prefix}_loss.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        [m["batch"] for m in baseline_metrics if m["win_ratio"] is not None],
        [m["win_ratio"] for m in baseline_metrics if m["win_ratio"] is not None],
        marker="o",
        linewidth=2,
        color="#5dade2",
        label="Baseline",
    )
    ax.plot(
        [m["batch"] for m in shaping_metrics if m["win_ratio"] is not None],
        [m["win_ratio"] for m in shaping_metrics if m["win_ratio"] is not None],
        marker="s",
        linewidth=2,
        color="#58d68d",
        label="Reward shaping",
    )
    ax.set_title("Evaluation Win Ratio")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Win ratio vs pure MCTS")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / f"{output_prefix}_win_ratio.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(
        [m["batch"] for m in shaping_metrics],
        [m["avg_heuristic_reward"] for m in shaping_metrics],
        color="#af7ac5",
        width=0.55,
    )
    ax.set_title("Average Heuristic Reward Per Batch")
    ax.set_xlabel("Batch")
    ax.set_ylabel("Average reward bonus")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / f"{output_prefix}_heuristic_reward.png", dpi=200)
    plt.close(fig)


def benchmark_inference(force=False):
    output_path = OUTPUT_DIR / "inference_benchmark.json"
    if output_path.exists() and not force:
        return json.loads(output_path.read_text(encoding="utf-8"))

    model_file = REPO_ROOT / "best_policy_8_8_5.model"
    try:
        params = pickle.load(open(model_file, "rb"))
    except Exception:
        params = pickle.load(open(model_file, "rb"), encoding="bytes")
    best_policy = PolicyValueNetNumpy(8, 8, params)

    results = []
    for n_playout in [100, 200, 400]:
        timings = []
        for _ in range(3):
            board = Board(width=8, height=8, n_in_row=5)
            board.init_board()
            player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=n_playout)
            start = time.perf_counter()
            _ = player.get_action(board)
            timings.append(time.perf_counter() - start)
        results.append(
            {
                "n_playout": n_playout,
                "avg_time_sec": float(np.mean(timings)),
                "std_time_sec": float(np.std(timings)),
            }
        )

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        [item["n_playout"] for item in results],
        [item["avg_time_sec"] for item in results],
        color="#ec7063",
        marker="o",
        linewidth=2.5,
    )
    ax.set_title("Inference Time vs. MCTS Playout")
    ax.set_xlabel("Playout")
    ax.set_ylabel("Average move time (s)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(ASSETS_DIR / "inference_time.png", dpi=200)
    plt.close(fig)
    return results


def summarise_results(
    baseline_metrics,
    baseline_summary,
    shaping_metrics,
    shaping_summary,
    inference_results,
    summary_name="experiment_summary.json",
):
    baseline_loss = baseline_metrics[-1]["loss"]
    shaping_loss = shaping_metrics[-1]["loss"]
    baseline_win = [
        record["win_ratio"] for record in baseline_metrics if record["win_ratio"] is not None
    ]
    shaping_win = [
        record["win_ratio"] for record in shaping_metrics if record["win_ratio"] is not None
    ]
    summary = {
        "baseline": {
            "final_loss": baseline_loss,
            "best_win_ratio": baseline_summary["best_win_ratio"],
            "last_win_ratio": baseline_win[-1] if baseline_win else None,
            "avg_episode_len": float(np.mean([item["episode_len"] for item in baseline_metrics])),
        },
        "reward_shaping": {
            "final_loss": shaping_loss,
            "best_win_ratio": shaping_summary["best_win_ratio"],
            "last_win_ratio": shaping_win[-1] if shaping_win else None,
            "avg_episode_len": float(np.mean([item["episode_len"] for item in shaping_metrics])),
            "avg_heuristic_reward": float(
                np.mean([item["avg_heuristic_reward"] for item in shaping_metrics])
            ),
            "total_open_three": int(np.sum([item["open_three_count"] for item in shaping_metrics])),
            "total_open_four": int(np.sum([item["open_four_count"] for item in shaping_metrics])),
            "total_five": int(np.sum([item["five_count"] for item in shaping_metrics])),
        },
        "inference_benchmark": inference_results,
        "comparison": {
            "loss_delta": float(baseline_loss - shaping_loss),
            "win_ratio_delta": float(
                (shaping_win[-1] if shaping_win else 0.0)
                - (baseline_win[-1] if baseline_win else 0.0)
            ),
        },
    }
    (OUTPUT_DIR / summary_name).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run reward shaping experiments.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--tag", default="experiment")
    parser.add_argument("--board-width", type=int, default=6)
    parser.add_argument("--board-height", type=int, default=6)
    parser.add_argument("--n-in-row", type=int, default=4)
    parser.add_argument("--n-playout", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=5000)
    parser.add_argument("--play-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--check-freq", type=int, default=2)
    parser.add_argument("--game-batch-num", type=int, default=8)
    parser.add_argument("--pure-mcts-playout-num", type=int, default=32)
    parser.add_argument("--eval-games", type=int, default=6)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    ensure_dirs()
    baseline_metrics, baseline_summary = run_train_variant(
        f"{args.tag}_baseline",
        [],
        args,
        force=args.force,
    )
    shaping_metrics, shaping_summary = run_train_variant(
        f"{args.tag}_reward_shaping",
        [
            "--reward-shaping",
            "--shaping-weight",
            "0.15",
            "--shape-open-three",
            "1.0",
            "--shape-open-four",
            "2.0",
            "--shape-five",
            "4.0",
        ],
        args,
        force=args.force,
    )
    plot_training_curves(baseline_metrics, shaping_metrics, output_prefix=args.tag)
    inference_results = benchmark_inference(force=args.force)
    summary = summarise_results(
        baseline_metrics,
        baseline_summary,
        shaping_metrics,
        shaping_summary,
        inference_results,
        summary_name=f"{args.tag}_summary.json",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
