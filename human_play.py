# -*- coding: utf-8 -*-
"""
human VS AI models
Input your move in the format: 2,3

@author: Junxiao Song
"""

from __future__ import print_function
import argparse
import importlib
import pickle
from datetime import datetime
from pathlib import Path
from game import Board, Game
from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
from PIL import Image, ImageDraw, ImageFont


class Human(object):
    """
    human player
    """

    def __init__(self, moves=None):
        self.player = None
        self._moves = list(moves or [])
        self._move_index = 0

    def set_player_ind(self, p):
        self.player = p

    def _read_location(self):
        if self._move_index < len(self._moves):
            location = self._moves[self._move_index]
            self._move_index += 1
            print("Your move: {}".format(location))
            return location
        return input("Your move: ")

    def get_action(self, board):
        try:
            location = self._read_location()
            if isinstance(location, str):  # for python3
                location = [int(n, 10) for n in location.split(",")]
            move = board.location_to_move(location)
        except EOFError:
            raise SystemExit("No more scripted moves available.")
        except Exception:
            move = -1
        if move == -1 or move not in board.availables:
            print("invalid move")
            move = self.get_action(board)
        return move

    def __str__(self):
        return "Human {}".format(self.player)


class GifRecorder(object):
    def __init__(self, output_path, frame_duration=700, final_duration=1800):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.frame_duration = frame_duration
        self.final_duration = final_duration
        self.frames = []
        self._winner = None
        self._font = self._load_font(22)
        self._small_font = self._load_font(16)

    def _load_font(self, size):
        for font_name in ["C:/Windows/Fonts/msyh.ttc",
                          "C:/Windows/Fonts/arial.ttf",
                          "C:/Windows/Fonts/consola.ttf"]:
            font_path = Path(font_name)
            if font_path.exists():
                return ImageFont.truetype(str(font_path), size)
        return ImageFont.load_default()

    def _draw_piece(self, draw, center, radius, fill_color, outline_color):
        x, y = center
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=fill_color,
            outline=outline_color,
            width=2,
        )

    def _board_to_image(self, board, player1, player2, winner=None):
        cell = 64
        margin = 70
        top_bar = 90
        width = margin * 2 + cell * (board.width - 1) + 1
        height = top_bar + margin + cell * (board.height - 1) + margin
        image = Image.new("RGB", (width, height), color=(16, 24, 32))
        draw = ImageDraw.Draw(image)

        draw.text((24, 18), "Gomoku Human vs AI", font=self._font,
                  fill=(232, 240, 248))
        status = "Player {} = X, Player {} = O".format(player1, player2)
        if winner is not None:
            if winner == -1:
                status += " | Result: Tie"
            else:
                status += " | Winner: Player {}".format(winner)
        draw.text((24, 52), status, font=self._small_font, fill=(156, 181, 205))

        board_left = margin
        board_top = top_bar
        board_right = board_left + cell * (board.width - 1)
        board_bottom = board_top + cell * (board.height - 1)

        for x in range(board.width):
            xpos = board_left + x * cell
            draw.line((xpos, board_top, xpos, board_bottom),
                      fill=(96, 120, 150), width=2)
            draw.text((xpos - 6, board_bottom + 18), str(x),
                      font=self._small_font, fill=(190, 205, 220))
        for y in range(board.height):
            ypos = board_bottom - y * cell
            draw.line((board_left, ypos, board_right, ypos),
                      fill=(96, 120, 150), width=2)
            draw.text((board_left - 32, ypos - 10), str(y),
                      font=self._small_font, fill=(190, 205, 220))

        radius = 22
        for move, player in board.states.items():
            row, col = board.move_to_location(move)
            xpos = board_left + col * cell
            ypos = board_bottom - row * cell
            if player == player1:
                self._draw_piece(draw, (xpos, ypos), radius,
                                 (248, 248, 248), (30, 30, 30))
            else:
                self._draw_piece(draw, (xpos, ypos), radius,
                                 (18, 18, 18), (220, 220, 220))

        if board.last_move != -1:
            row, col = board.move_to_location(board.last_move)
            xpos = board_left + col * cell
            ypos = board_bottom - row * cell
            marker = 12
            draw.rectangle(
                (xpos - marker, ypos - marker, xpos + marker, ypos + marker),
                outline=(255, 99, 71),
                width=3,
            )

        return image

    def capture(self, board, player1, player2):
        self.frames.append(self._board_to_image(board, player1, player2))

    def finish(self, winner):
        self._winner = winner
        if not self.frames:
            return
        final_frame = self.frames[-1].copy()
        board_draw = ImageDraw.Draw(final_frame)
        if winner == -1:
            label = "Game Over: Tie"
        else:
            label = "Game Over: Player {} wins".format(winner)
        board_draw.rounded_rectangle(
            (24, final_frame.height - 62, 320, final_frame.height - 18),
            radius=10,
            fill=(24, 36, 48),
            outline=(78, 161, 255),
            width=2,
        )
        board_draw.text((38, final_frame.height - 52), label,
                        font=self._small_font, fill=(232, 240, 248))
        self.frames[-1] = final_frame
        durations = [self.frame_duration] * len(self.frames)
        durations[-1] = self.final_duration
        self.frames[0].save(
            self.output_path,
            save_all=True,
            append_images=self.frames[1:],
            duration=durations,
            loop=0,
        )
        print("GIF saved to {}".format(self.output_path.resolve()))


def load_scripted_moves(moves_file):
    if not moves_file:
        return []
    with open(moves_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines()]
    return [line for line in lines if line and not line.startswith("#")]


def build_ai_player(args):
    if args.backend == "pure":
        return MCTS_Pure(c_puct=args.c_puct, n_playout=args.n_playout)

    if args.backend == "pytorch":
        policy_module = importlib.import_module("policy_value_net_pytorch")
        PolicyValueNetTorch = policy_module.PolicyValueNet
        best_policy = PolicyValueNetTorch(
            args.width,
            args.height,
            model_file=args.model_file,
            use_gpu=args.use_gpu,
        )
        return MCTSPlayer(
            best_policy.policy_value_fn,
            c_puct=args.c_puct,
            n_playout=args.n_playout,
        )

    try:
        policy_param = pickle.load(open(args.model_file, "rb"))
    except Exception:
        policy_param = pickle.load(open(args.model_file, "rb"), encoding="bytes")
    best_policy = PolicyValueNetNumpy(args.width, args.height, policy_param)
    return MCTSPlayer(
        best_policy.policy_value_fn,
        c_puct=args.c_puct,
        n_playout=args.n_playout,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="Play Gomoku against AlphaZero.")
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=8)
    parser.add_argument("--n-in-row", type=int, default=5)
    parser.add_argument("--model-file", default="best_policy_8_8_5.model")
    parser.add_argument(
        "--backend",
        choices=["numpy", "pytorch", "pure"],
        default="numpy",
        help="Inference backend for the AI player.",
    )
    parser.add_argument("--c-puct", type=float, default=5)
    parser.add_argument("--n-playout", type=int, default=400)
    parser.add_argument(
        "--start-player",
        type=int,
        choices=[0, 1],
        default=1,
        help="0 means human first, 1 means AI first.",
    )
    parser.add_argument(
        "--moves-file",
        default=None,
        help="Optional scripted moves file, one move per line such as 2,3.",
    )
    parser.add_argument(
        "--gif-path",
        default=None,
        help="Optional GIF output path. Default saves to artifacts/gifs/.",
    )
    parser.add_argument(
        "--no-save-gif",
        action="store_true",
        help="Disable automatic GIF export for the whole match.",
    )
    parser.add_argument("--use-gpu", action="store_true")
    return parser


def run(args=None):
    args = build_parser().parse_args(args=args)
    recorder = None
    try:
        board = Board(width=args.width, height=args.height, n_in_row=args.n_in_row)
        game = Game(board)
        mcts_player = build_ai_player(args)
        human = Human(moves=load_scripted_moves(args.moves_file))
        if not args.no_save_gif:
            output_path = args.gif_path
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = Path("artifacts") / "gifs" / (
                    "human_play_{}.gif".format(timestamp)
                )
            recorder = GifRecorder(output_path)
        game.start_play(
            human,
            mcts_player,
            start_player=args.start_player,
            is_shown=1,
            recorder=recorder,
        )
    except KeyboardInterrupt:
        print("\n\rquit")


if __name__ == "__main__":
    run()
