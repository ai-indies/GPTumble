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

from collections import namedtuple

_side_val_class = namedtuple("side", ["val", "name"])


class Side(Enum):
    LEFT = _side_val_class(0, "l")
    RIGHT = _side_val_class(1, "r")


class Board:
    rng = np.random.default_rng()

    def __init__(self, width, height, R_init_chance=False, weights_offbalance=False) -> None:
        self.width = width
        self.height = height
        self.params = {
            "L": {
                "ball_from_l": {
                    "ball_to_l": self._get_almost_non_rnd_params(0, weights_offbalance),
                    "to_state_R": self._get_almost_non_rnd_params(1, weights_offbalance),
                },
                "ball_from_r": {
                    "ball_to_l": self._get_almost_non_rnd_params(0, weights_offbalance),
                    "to_state_R": self._get_almost_non_rnd_params(1, weights_offbalance),
                },
            },
            "R": {
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

        self._state_to_side = {0: "L", 1: "R"}

        self._side_to_state = {v: k for k, v in self._state_to_side.items()}

        self.states = np.random.choice([0, 1], size=(self.width, self.height), p=[1 - R_init_chance, R_init_chance])

    def _as_string(self, ball_pos=None) -> str:
        board_as_slash = self._one_char_board
        if ball_pos:
            board_as_slash[ball_pos[0]] = (
                board_as_slash[ball_pos[0]][: ball_pos[1] + ball_pos[0] % 2]
                + ["*"]
                + board_as_slash[ball_pos[0]][ball_pos[1] + ball_pos[0] % 2 :]
            )
        return "\n".join(
            ("" if self.is_wide(rn) else " ") + " ".join(row) for rn, row in enumerate(board_as_slash)
        ).replace(" * ", "*")

    @staticmethod
    def _state_to_char(state):
        return "\\" if state == 0 else "/"

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

            if self.is_wide(row):  # wide row
                in_row_pos += move.value[0] - 1
            else:
                in_row_pos += move.value[0]

            if DEBUG:
                print(self._render((row, in_row_pos)), "\n")

        return in_row_pos

    def _roll_w_h(self, w, h, last_shift: Side) -> Side:
        if self.is_wide(h):
            if w == 0:
                return Side.RIGHT
            if w == self.width - 1:
                return Side.LEFT

        old_state = state = self.states[h][w]

        param_state = self.params.get(self._state_to_side[state])

        new_param_pos = param_state.get("ball_from_" + last_shift.value[1])

        to_l_prob = new_param_pos.get("ball_to_l")[h][w]
        to_R_prob = new_param_pos.get("to_state_R")[h][w]

        side_roll = self.rng.random()
        state_roll = self.rng.random()

        new_state = state = "L" if state_roll > to_R_prob else "R"

        # print(f"- new {h}:{w} state {'<' if state == 'L' else '>'}")

        self.states[h][w] = self._side_to_state[new_state]

        move = Side.RIGHT if side_roll > to_l_prob else Side.LEFT

        if DEBUG:
            print(f"{h}:{w} {self._state_to_char(old_state)} => {self._state_to_char(new_state)}, move: {move.name}")

        return move

    def _get_almost_non_rnd_params(self, main_mode, off_main=0):
        return self.rng.random((self.height, self.width)) * off_main + main_mode * (1 - off_main)

    @staticmethod
    def is_wide(row_n) -> bool:
        return row_n % 2 == 1

    def row_w(self, row_n) -> bool:
        return self.width if self.is_wide(row_n) else (self.width - 1)

    def _row_n_zero_distr(row_n):
        """returns zeros array size: 2 * row_width"""
        return np.zeros((2, self.row_w(row_n=0)))

    def nd_roll_from_pos(self, pos):  # returns end bin distribution and state update distribution
        # starting with prob = [0, 0, ..., 1 (pos), 0, ..., 0]
        in_distr = self._row_n_zero_distr(0)
        in_distr[Side.RIGHT.val, pos] = 1.0

        for row_n in range(self.height):
            next_r_prob = self._nd_roll_one_row(in_distr, row_n)

            if self.is_wide(row_n):  # wide row
                in_row_pos += move.value[0] - 1
            else:
                in_row_pos += move.value[0]

            if DEBUG:
                print(self._render((row, in_row_pos)), "\n")

        return in_row_pos

    def _nd_roll_one_row(self, in_distr, row_n):
        """
        in_probs - float array of size row_w * 2 (from left / from right)

        returns: float array of size next_row_w * 2 (left / right)
        """

        out_dist = self._row_n_zero_distr(row_n)
        out_state_distr = self._row_n_zero_distr(row_n)

        for sided_in_distr, transition_probs, state_change_probs in zip(in_distr, self._get_row_probs(row_n)):
            sided_out_dist = sided_in_distr * transition_probs  # needs board borders adjustment or does it not?
            out_dist += sided_out_dist

            sided_out_state = sided_in_distr * state_change_probs
            out_state_distr += sided_out_state

        return out_distr, out_state_distr


if __name__ == "__main__":
    from collections import Counter

    stats = Counter()
    trans_stats = Counter()

    b = Board(30, 30, R_init_chance=0.1, weights_offbalance=0.9)
    ball_pos = 0  # b.width // 2

    for i in range(1):
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
