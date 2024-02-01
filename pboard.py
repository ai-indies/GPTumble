"""
switch numbering in rows:
row
0       0       1       2       3       4
1   0       1       2       3       4       5
2       0       1       2       3       4

Transition from odd row is same/+1, from even row -1/same
"""
import numpy as np
from enum import Enum
import logging


DEBUG = 0

logger = logging.Logger(f"Board logger", logging.DEBUG if DEBUG else logging.INFO)


class Side(Enum):
    LEFT = (0, "l")
    RIGHT = (1, "r")


class Board:
    rng = np.random.default_rng()

    def __init__(self, width, height, R_init_chance=False, weights_offbalance=False) -> None:
        self.width = width
        self.height = height
        self.params = {
            "state_L": {
                "ball_from_l": {
                    "ball_to_l": self._get_almost_non_rnd_params(0, weights_offbalance),
                    "to_state_R": self._get_almost_non_rnd_params(1, weights_offbalance),
                },
                "ball_from_r": {
                    "ball_to_l": self._get_almost_non_rnd_params(0, weights_offbalance),
                    "to_state_R": self._get_almost_non_rnd_params(1, weights_offbalance),
                },
            },
            "state_R": {
                "ball_from_l": {
                    "ball_to_l": self._get_almost_non_rnd_params(1, weights_offbalance),
                    "to_state_R": self._get_almost_non_rnd_params(0, weights_offbalance),
                },
                "ball_from_r": {
                    "ball_to_l": self._get_almost_non_rnd_params(1, weights_offbalance),
                    "to_state_R": self._get_almost_non_rnd_params(0, weights_offbalance),
                },
            },
        }

        self.states = [
            [
                "state_" + ("R" if R_init_chance and self.rng.random() > R_init_chance else "L")
                for j in range(width if i % 2 == 1 else width - 1)
            ]
            for i in range(height)
        ]

    def _as_string(self, ball_pos=None) -> str:
        board_as_slash = self._one_char_board
        if ball_pos:
            board_as_slash[ball_pos[0]] = (
                board_as_slash[ball_pos[0]][: ball_pos[1] + ball_pos[0] % 2]
                + ["*"]
                + board_as_slash[ball_pos[0]][ball_pos[1] + ball_pos[0] % 2 :]
            )
        return "\n".join(("" if rn % 2 == 1 else " ") + " ".join(row) for rn, row in enumerate(board_as_slash)).replace(
            " * ", "*"
        )

    @staticmethod
    def _state_to_char(state):
        return "\\" if state == "state_L" else "/"

    @property
    def _one_char_board(self):
        return [[self._state_to_char(pos) for pos in row] for row in self.states]

    def _render(self, ball_pos=None):
        return self._as_string(ball_pos)
        # rendered = self._render_a_board(self._one_char_board + [["="] * (self.width)])
        # if ball_pos is not None:
        #     row, pos = ball_pos

        #     idx = row * (2 * self.width) + pos
        #     if row % 2 == 1:
        #         idx += 1
        #     rendered = rendered[:idx] + "*" + rendered[idx + 1 :]

        # return rendered

    def __repr__(self):
        return self._render()

    def roll_from_w(self, w) -> int:
        in_row_pos = w
        last_move = Side.RIGHT
        if DEBUG:
            print("Input:")
            print(" " + (" |" * w + "*" + "| " * (self.width - w - 1)))  # ball over the board
            print(self._render(), "\n")

        for row in range(self.height):
            move = self._roll_w_h(in_row_pos, row, last_move)

            if row % 2 == 1:  # wide row
                in_row_pos += move.value[0] - 1
            else:
                in_row_pos += move.value[0]

            if DEBUG:
                print(self._render((row, in_row_pos)), "\n")

        return in_row_pos

    def _roll_w_h(self, w, h, last_shift: Side) -> Side:
        if h % 2 == 1:
            if w == 0:
                return Side.RIGHT
            if w == self.width - 1:
                return Side.LEFT

        old_state = state = self.states[h][w]

        param_state = self.params.get(state)

        new_param_pos = param_state.get("ball_from_" + last_shift.value[1])

        to_l_prob = new_param_pos.get("ball_to_l")[h][w]
        to_R_prob = new_param_pos.get("to_state_R")[h][w]

        side_roll = self.rng.random()
        state_roll = self.rng.random()

        new_state = state = "state_L" if state_roll > to_R_prob else "state_R"
        # print(f"- new {h}:{w} state {'<' if state == 'state_L' else '>'}")
        self.states[h][w] = new_state

        move = Side.RIGHT if side_roll > to_l_prob else Side.LEFT

        if DEBUG:
            print(f"{h}:{w} {self._state_to_char(old_state)} => {self._state_to_char(new_state)}, move: {move.name}")

        return move

    def _get_almost_non_rnd_params(self, main_mode, off_main=0):
        return self.rng.random((self.height, self.width)) * off_main + main_mode * (1 - off_main)

    def nd_roll_from_w(self, w) -> int:
        in_row_pos = w
        last_move = Side.RIGHT

        for row in range(self.height):
            move = self._roll_w_h(in_row_pos, row, last_move)

            if row % 2 == 1:  # wide row
                in_row_pos += move.value[0] - 1
            else:
                in_row_pos += move.value[0]

            if DEBUG:
                print(self._render((row, in_row_pos)), "\n")

        return in_row_pos


if __name__ == "__main__":
    from collections import Counter

    stats = Counter()
    trans_stats = Counter()

    b = Board(30, 30, R_init_chance=0.1, weights_offbalance=0.9)
    ball_pos = 0  # b.width // 2

    for i in range(10000):
        prev_pos = ball_pos
        ball_pos = b.roll_from_w(ball_pos)

        stats.update([ball_pos])
        trans_stats.update([(prev_pos, ball_pos)])

    # print(b._render((b.height - 1, ball_pos)))

    max_val = max(stats.values())
    for p in range(b.width):
        print("*" * ((stats.get(p, 0) > 0) + stats.get(p, 0) * 100 // max_val))
    print(max_val / 100)

    max_val = max(trans_stats.values())
    for k, v in sorted(trans_stats.items()):
        print(k, "*" * (v * 100 // max_val))
    print(max_val / 100)
