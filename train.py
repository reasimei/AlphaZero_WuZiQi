# -*- coding: utf-8 -*-
"""
An implementation of the training pipeline of AlphaZero for Gomoku

@author: Junxiao Song
"""

from __future__ import print_function
import argparse
import importlib
import json
import os
import random
import time
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer


def get_policy_value_net(backend):
    backend_modules = {
        "pytorch": "policy_value_net_pytorch",
        "tensorflow": "policy_value_net_tensorflow",
        "keras": "policy_value_net_keras",
        "theano": "policy_value_net",
    }
    module_name = backend_modules[backend]
    module = importlib.import_module(module_name)
    return module.PolicyValueNet


def create_policy_value_net(policy_cls, args):
    init_kwargs = {}
    if args.backend == "pytorch":
        init_kwargs["use_gpu"] = args.use_gpu
    if args.init_model:
        init_kwargs["model_file"] = args.init_model
    return policy_cls(args.board_width, args.board_height, **init_kwargs)


def set_random_seed(seed, backend):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    if backend == "pytorch":
        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is not None:
            torch = importlib.import_module("torch")
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)


def build_reward_shaping_config(args):
    return {
        "enabled": args.reward_shaping,
        "shaping_weight": args.shaping_weight,
        "open_three_weight": args.shape_open_three,
        "open_four_weight": args.shape_open_four,
        "five_weight": args.shape_five,
        "normalizer": args.shape_normalizer,
    }


class TrainPipeline():
    def __init__(self, args):
        self.args = args
        PolicyValueNet = get_policy_value_net(args.backend)
        # params of the board and the game
        self.board_width = args.board_width
        self.board_height = args.board_height
        self.n_in_row = args.n_in_row
        self.board = Board(width=self.board_width,
                           height=self.board_height,
                           n_in_row=self.n_in_row)
        self.game = Game(self.board)
        # training params
        self.learn_rate = args.learn_rate
        self.lr_multiplier = 1.0  # adaptively adjust the learning rate based on KL
        self.temp = args.temp  # the temperature param
        self.n_playout = args.n_playout  # num of simulations for each move
        self.c_puct = args.c_puct
        self.buffer_size = args.buffer_size
        self.batch_size = args.batch_size  # mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = args.play_batch_size
        self.epochs = args.epochs  # num of train_steps for each update
        self.kl_targ = args.kl_targ
        self.check_freq = args.check_freq
        self.game_batch_num = args.game_batch_num
        self.best_win_ratio = 0.0
        self.save_dir = args.save_dir
        self.eval_games = args.eval_games
        self.reward_shaping = build_reward_shaping_config(args)
        # num of simulations used for the pure mcts, which is used as
        # the opponent to evaluate the trained policy
        self.pure_mcts_playout_num = args.pure_mcts_playout_num
        os.makedirs(self.save_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.save_dir, "training_metrics.jsonl")
        self.summary_path = os.path.join(self.save_dir, "training_summary.json")
        self.config_path = os.path.join(self.save_dir, "train_config.json")
        self.records = []
        self.last_episode_info = {
            "episode_len": 0,
            "avg_heuristic_reward": 0.0,
            "max_heuristic_reward": 0.0,
            "open_three_count": 0,
            "open_four_count": 0,
            "five_count": 0,
            "shaping_enabled": False,
        }
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=2)
        self.policy_value_net = create_policy_value_net(PolicyValueNet, args)
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout,
                                      is_selfplay=1)

    def get_equi_data(self, play_data):
        """augment the data set by rotation and flipping
        play_data: [(state, mcts_prob, winner_z), ..., ...]
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        episode_infos = []
        for i in range(n_games):
            winner, play_data, episode_info = self.game.start_self_play(
                self.mcts_player,
                temp=self.temp,
                reward_shaping=self.reward_shaping,
            )
            play_data = list(play_data)[:]
            self.episode_len = len(play_data)
            episode_infos.append(episode_info)
            # augment the data
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)
        if episode_infos:
            count = float(len(episode_infos))
            self.last_episode_info = {
                "episode_len": int(np.mean([info["episode_len"] for info in episode_infos])),
                "avg_heuristic_reward": float(
                    np.mean([info["avg_heuristic_reward"] for info in episode_infos])
                ),
                "max_heuristic_reward": float(
                    np.max([info["max_heuristic_reward"] for info in episode_infos])
                ),
                "open_three_count": int(
                    np.sum([info["open_three_count"] for info in episode_infos])
                ),
                "open_four_count": int(
                    np.sum([info["open_four_count"] for info in episode_infos])
                ),
                "five_count": int(np.sum([info["five_count"] for info in episode_infos])),
                "shaping_enabled": episode_infos[0]["shaping_enabled"],
            }

    def policy_update(self):
        """update the policy-value net"""
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        winner_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    winner_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = (1 -
                             np.var(np.array(winner_batch) - old_v.flatten()) /
                             np.var(np.array(winner_batch)))
        explained_var_new = (1 -
                             np.var(np.array(winner_batch) - new_v.flatten()) /
                             np.var(np.array(winner_batch)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl,
                        self.lr_multiplier,
                        loss,
                        entropy,
                        explained_var_old,
                        explained_var_new))
        return {
            "loss": float(loss),
            "entropy": float(entropy),
            "kl": float(kl),
            "lr_multiplier": float(self.lr_multiplier),
            "explained_var_old": float(explained_var_old),
            "explained_var_new": float(explained_var_new),
        }

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct,
                                         n_playout=self.n_playout)
        pure_mcts_player = MCTS_Pure(c_puct=5,
                                     n_playout=self.pure_mcts_playout_num)
        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player,
                                          pure_mcts_player,
                                          start_player=i % 2,
                                          is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0*(win_cnt[1] + 0.5*win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
                self.pure_mcts_playout_num,
                win_cnt[1], win_cnt[2], win_cnt[-1]))
        return win_ratio

    def append_metrics(self, record):
        self.records.append(record)
        with open(self.metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_summary(self):
        summary = {
            "backend": self.args.backend,
            "reward_shaping": self.reward_shaping,
            "best_win_ratio": float(self.best_win_ratio),
            "num_records": len(self.records),
            "last_record": self.records[-1] if self.records else None,
            "save_dir": self.save_dir,
        }
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    def run(self):
        """run the training pipeline"""
        try:
            for i in range(self.game_batch_num):
                batch_start = time.time()
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}, shape_avg:{:.4f}, open3:{}, open4:{}, five:{}".format(
                        i+1,
                        self.episode_len,
                        self.last_episode_info["avg_heuristic_reward"],
                        self.last_episode_info["open_three_count"],
                        self.last_episode_info["open_four_count"],
                        self.last_episode_info["five_count"]))
                update_metrics = None
                if len(self.data_buffer) > self.batch_size:
                    update_metrics = self.policy_update()
                # check the performance of the current model,
                # and save the model params
                win_ratio = None
                if (i+1) % self.check_freq == 0:
                    print("current self-play batch: {}".format(i+1))
                    win_ratio = self.policy_evaluate(self.eval_games)
                    self.policy_value_net.save_model(
                        os.path.join(self.save_dir, 'current_policy.model')
                    )
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!!!!!!!!")
                        self.best_win_ratio = win_ratio
                        # update the best_policy
                        self.policy_value_net.save_model(
                            os.path.join(self.save_dir, 'best_policy.model')
                        )
                        if (self.best_win_ratio == 1.0 and
                                self.pure_mcts_playout_num < 5000):
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0
                record = {
                    "batch": i + 1,
                    "episode_len": int(self.episode_len),
                    "buffer_size": len(self.data_buffer),
                    "batch_time_sec": round(time.time() - batch_start, 4),
                    "win_ratio": None if win_ratio is None else float(win_ratio),
                    "best_win_ratio": float(self.best_win_ratio),
                    "reward_shaping_enabled": self.reward_shaping["enabled"],
                    "avg_heuristic_reward": self.last_episode_info["avg_heuristic_reward"],
                    "max_heuristic_reward": self.last_episode_info["max_heuristic_reward"],
                    "open_three_count": self.last_episode_info["open_three_count"],
                    "open_four_count": self.last_episode_info["open_four_count"],
                    "five_count": self.last_episode_info["five_count"],
                }
                if update_metrics:
                    record.update(update_metrics)
                self.append_metrics(record)
        except KeyboardInterrupt:
            print('\n\rquit')
        finally:
            self.write_summary()


def build_parser():
    parser = argparse.ArgumentParser(description="Train AlphaZero Gomoku.")
    parser.add_argument(
        "--backend",
        choices=["pytorch", "tensorflow", "keras", "theano"],
        default="pytorch",
    )
    parser.add_argument("--board-width", type=int, default=6)
    parser.add_argument("--board-height", type=int, default=6)
    parser.add_argument("--n-in-row", type=int, default=4)
    parser.add_argument("--learn-rate", type=float, default=2e-3)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--n-playout", type=int, default=400)
    parser.add_argument("--c-puct", type=float, default=5)
    parser.add_argument("--buffer-size", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--play-batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--kl-targ", type=float, default=0.02)
    parser.add_argument("--check-freq", type=int, default=50)
    parser.add_argument("--game-batch-num", type=int, default=1500)
    parser.add_argument("--pure-mcts-playout-num", type=int, default=1000)
    parser.add_argument("--save-dir", default="training_output")
    parser.add_argument("--init-model", default=None)
    parser.add_argument("--eval-games", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--reward-shaping", action="store_true")
    parser.add_argument("--shaping-weight", type=float, default=0.15)
    parser.add_argument("--shape-open-three", type=float, default=1.0)
    parser.add_argument("--shape-open-four", type=float, default=2.0)
    parser.add_argument("--shape-five", type=float, default=4.0)
    parser.add_argument("--shape-normalizer", type=float, default=4.0)
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    set_random_seed(args.seed, args.backend)
    training_pipeline = TrainPipeline(args)
    training_pipeline.run()
