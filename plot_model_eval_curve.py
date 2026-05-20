from __future__ import annotations

import argparse
import importlib
import json
import pickle
import time
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from mcts_pure import MCTSPlayer as MCTS_Pure
from policy_value_net_numpy import PolicyValueNetNumpy


def build_ai_player(args):
    if args.backend == "pytorch":
        policy_module = importlib.import_module("policy_value_net_pytorch")
        PolicyValueNetTorch = policy_module.PolicyValueNet
        policy = PolicyValueNetTorch(
            args.width,
            args.height,
            model_file=args.model_file,
            use_gpu=args.use_gpu,
        )
        return MCTSPlayer(
            policy.policy_value_fn,
            c_puct=args.c_puct,
            n_playout=args.ai_playout,
        )

    try:
        policy_param = pickle.load(open(args.model_file, "rb"))
    except Exception:
        policy_param = pickle.load(open(args.model_file, "rb"), encoding="bytes")
    policy = PolicyValueNetNumpy(args.width, args.height, policy_param)
    return MCTSPlayer(
        policy.policy_value_fn,
        c_puct=args.c_puct,
        n_playout=args.ai_playout,
    )


def evaluate_once(game, ai_player, pure_playout, n_games):
    pure_player = MCTS_Pure(c_puct=5, n_playout=pure_playout)
    win_cnt = defaultdict(int)
    started_at = time.time()
    for i in range(n_games):
        winner = game.start_play(
            ai_player,
            pure_player,
            start_player=i % 2,
            is_shown=0,
        )
        win_cnt[winner] += 1
    elapsed = time.time() - started_at
    win_ratio = (win_cnt[1] + 0.5 * win_cnt[-1]) / float(n_games)
    return {
        "pure_mcts_playout": pure_playout,
        "games": n_games,
        "wins": int(win_cnt[1]),
        "losses": int(win_cnt[2]),
        "ties": int(win_cnt[-1]),
        "win_ratio": float(win_ratio),
        "elapsed_sec": round(elapsed, 4),
    }


def plot_curve(records, output_path):
    xs = [record["pure_mcts_playout"] for record in records]
    ys = [record["win_ratio"] for record in records]
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(xs, ys, marker="o", linewidth=2, color="#5dade2")
    for record in records:
        ax.annotate(
            f'{record["win_ratio"]:.3f}',
            (record["pure_mcts_playout"], record["win_ratio"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            color="#d6eaf8",
        )
    ax.set_title("Model Evaluation Win Ratio vs Pure MCTS")
    ax.set_xlabel("Pure MCTS Playout")
    ax.set_ylabel("Win Ratio")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Gomoku model against Pure MCTS and plot win ratio."
    )
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--backend", choices=["numpy", "pytorch"], default="numpy")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--n-in-row", type=int, default=5)
    parser.add_argument("--ai-playout", type=int, default=400)
    parser.add_argument("--c-puct", type=float, default=5.0)
    parser.add_argument("--eval-games", type=int, default=4)
    parser.add_argument("--pure-playouts", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", default="artifacts/model_eval")
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    board = Board(width=args.width, height=args.height, n_in_row=args.n_in_row)
    game = Game(board)
    ai_player = build_ai_player(args)

    records = []
    for pure_playout in args.pure_playouts:
        result = evaluate_once(game, ai_player, pure_playout, args.eval_games)
        records.append(result)
        print(
            "pure_mcts_playout={pure_mcts_playout}, games={games}, "
            "win={wins}, loss={losses}, tie={ties}, win_ratio={win_ratio:.3f}, elapsed={elapsed_sec}s".format(
                **result
            )
        )

    chart_path = output_dir / "win_ratio_curve.png"
    json_path = output_dir / "evaluation_results.json"
    plot_curve(records, chart_path)

    summary = {
        "model_file": str(Path(args.model_file).resolve()),
        "backend": args.backend,
        "board": {
            "width": args.width,
            "height": args.height,
            "n_in_row": args.n_in_row,
        },
        "ai_playout": args.ai_playout,
        "eval_games": args.eval_games,
        "pure_playouts": args.pure_playouts,
        "records": records,
        "win_ratio_curve": str(chart_path),
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
