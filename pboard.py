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
from collections import namedtuple

DEBUG = 0
SEED = 0
np.random.seed(SEED)

RED_COLOR = "\033[31m"
GREEN_COLOR = "\033[32m"
RESET_COLOR = "\033[0m"


logger = logging.Logger(f"Board logger", logging.DEBUG if DEBUG else logging.INFO)


Side = namedtuple("Side", ["val", "name"])

LEFT_SIDE = Side(0, "L")
RIGHT_SIDE = Side(1, "R")
SIDES = [LEFT_SIDE, RIGHT_SIDE]

SIDE_VAL_TO_SIDE = {side.val: side for side in SIDES}


class Board:
    rng = np.random.default_rng(seed=SEED)
    block_chars = np.array([" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"])

    def __init__(self, width, height, R_init_chance=False, offbalance=False) -> None:
        self.width = width
        self.height = height

        self.nd_states = np.random.choice(
            [s.val for s in SIDES], size=(self.height, self.width), p=[1 - R_init_chance, R_init_chance]
        )

        # 2 (switch state), 2 (incoming ball side), board height, N
        self.fall_weights = self._init_weight(0, offbalance, extra_dims=(len(SIDES), len(SIDES)))
        self.switch_weights = self._init_weight(0, offbalance, extra_dims=(len(SIDES), len(SIDES)))

    _switch_render_map = {0: f"{RED_COLOR}\\_{RESET_COLOR}", 1: f"{RED_COLOR}_/{RESET_COLOR}"}

    def _two_char_board(self):
        return [
            [self._switch_render_map[pos_state] for pos_state in row[: self.row_width(rn)]]
            for rn, row in enumerate(self.nd_states)
        ]

    def render_with_distr(self, distr) -> str:
        two_char_board = self._two_char_board()
        return "\n".join(
            ("" if self.is_wide_row(rn) else "  ")
            + (self._render_ball_distr(distr[rn]) if rn < len(distr) else "")
            + ("" if row and self.is_wide_row(rn) else "  ")
            + ("  ".join(row) if row else "")
            for rn, row in enumerate(
                two_char_board + [None]
            )  # add None so last distr get rendered but not the extra row
        )

    def __repr__(self):
        return self.render_with_distr([])

    def _init_weight(self, main_mode, off_main=0, extra_dims=tuple()):
        return np.array(
            ## Deterministic for now
            # self.rng.random(extra_dims + (self.height, self.width)) * off_main + main_mode * (1 - off_main),
            np.ones(extra_dims + (self.height, self.width)) * off_main + main_mode * (1 - off_main),
            # dtype=np.float16, # can be used for less verbose debug
        )

    @staticmethod
    def is_wide_row(row_n) -> bool:
        return row_n % 2 == 1

    def row_width(self, row_n) -> bool:
        return self.width if self.is_wide_row(row_n) else (self.width - 1)

    def _row_n_zeros(self, row_n):
        """returns zeros array size: 2 * row_width"""
        return np.zeros((2, self.row_width(row_n=row_n)))

    def nd_roll_from_pos(self, pos):  # returns end bin distribution and state update distribution
        # starting with prob = [0, 0, ..., 1 (pos), 0, ..., 0]
        in_distr = self._row_n_zeros(0)
        in_distr[LEFT_SIDE.val, pos] = 0.5
        in_distr[RIGHT_SIDE.val, pos] = 0.5

        per_row_distrs = [in_distr]

        if DEBUG:
            print(self.render_with_distr(per_row_distrs))

        for row_n in range(self.height):
            prev_distr = per_row_distrs[-1]

            next_distr, _ignore_new_state_distr = self._nd_roll_one_row(prev_distr, row_n)
            # ^ ignoring new state distr for now.

            per_row_distrs.append(next_distr)

            if DEBUG:
                print("---")
                print(self.render_with_distr(per_row_distrs))

        # TODO return state distrs too
        return per_row_distrs

    @staticmethod
    def _inverse_probs_with_state(probs, state):
        """
        Inverses array of probs depending on the state
        t=np.array([0,0.25,0.75]*2)
        s=np.array([0,0,0,1,1,1])
        f(s,t)
        >>> array([0.  , 0.25, 0.75, 1.  , 0.75, 0.25])
        """
        return probs + state - 2 * probs * state

    def _row_probs(self, row_n):
        """
        Using current state of the switches

        returns:
        fall_probs: float array of size 2 (incoming ball side) * row_width
        switch_probs: float array of size 2 (incoming ball sides) * row_width
        """

        # Considering N (row width) switches in states (0s and 1s): s = [s1, ..., sn]
        # and the weight matrix with dimensions:
        # 2 (switch state), 2 (incoming ball side), board height, N

        N = self.row_width(row_n)
        s = self.nd_states[row_n, :N]

        # To make / support non-deterministic states weights should be lin-comb of corresponding state probs

        fall_probs = self._inverse_probs_with_state(self.fall_weights[s, ..., row_n, range(N)].swapaxes(0, 1), s)

        switch_probs = self._inverse_probs_with_state(self.switch_weights[s, ..., row_n, range(N)].swapaxes(0, 1), s)

        return fall_probs, switch_probs

    def _nd_roll_one_row(self, in_distr, row_n):
        """
        in_probs - float array of size 2 (sides) * row_width

        returns: float array of size 2 (sides) * next_row_w
        """

        out_distr = self._row_n_zeros(row_n + 1)  # it goes to the next row - thus +1 ?
        out_state_distr = self._row_n_zeros(row_n)

        if DEBUG > 1:
            print("sided_in_distr, fall_probs, l_sided_out_distr, (1 - fall_probs), r_sided_out_distr")

        # TODO 'for side' can possibly be optimized to use just one matmult instead of a loop.
        for side, sided_in_distr, fall_probs, switch_probs in zip(SIDES, in_distr, *self._row_probs(row_n)):
            # FIXME!! needs board boundaries adjustment or does it not?

            #                       State 0                State 1
            #                   From 0  From 1          From 0  From 1
            # Falls left        p00     p01             p10     p11
            # Falls right       1-p00   1-p01           1-p10   1-p11
            #
            # "Normal" switch:  p00 ~= 0, p01 ~= 0, p10 ~= 1, p11 ~= 1
            # "Cross" switch:   p00 ~= 0, p01 ~= 1, p10 ~= 0, p11 ~= 1 (essentially no state)
            # "Bounce" switch:  p00 ~= 1, p01 ~= 0, p10 ~= 0, p11 ~= 1 (essentially no state)
            #
            # Not sure about this - see MAJOR TODO on correlation between falls and switches.
            # Switches to 0     s00     s01             s10     s11
            # Switches to 1
            #

            l_sided_out_distr = sided_in_distr * fall_probs
            r_sided_out_distr = sided_in_distr * (1 - fall_probs)

            if DEBUG > 1:
                print(sided_in_distr, fall_probs, l_sided_out_distr, (1 - fall_probs), r_sided_out_distr)

            if not self.is_wide_row(row_n):
                # [:-1] and [1:] denote shift of narrow to wide row
                out_distr[1][:-1] += l_sided_out_distr
                out_distr[0][1:] += r_sided_out_distr

            else:
                # [:-1] and [1:] denote shift of wide to narrow row
                out_distr[1] += l_sided_out_distr[1:]
                out_distr[0] += r_sided_out_distr[:-1]

            # # MAJOR TODO - new state should be most of the time the same as the out_distr!
            # sided_out_state = sided_in_distr * switch_probs
            # out_state_distr += sided_out_state

        return out_distr, out_state_distr

    def _render_ball_distr(self, ball_distr, norm=True):  # norm is meh
        # flat
        flat = ball_distr.swapaxes(0, 1)
        # scale to block char max val and optionally norm first
        rescaled = (flat / (ball_distr.max() if norm else 1) * (len(self.block_chars) - 1) // 1).astype(int)

        return (
            "  ".join("".join(self.block_chars[pair]) for pair in rescaled)
            + (" - " + "".join(repr(flat.astype(np.float16))[6:-16].split()) if DEBUG else "")
            + "\n"
        )


def sample_probs_with_temp(probs, temp=1.0):
    if not temp:
        return probs.argmax()

    # Sample according to the adjusted probabilities
    scaled_logits = probs / temp
    # Compute softmax values
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))  # For numerical stability
    softmax = exp_logits / np.sum(exp_logits)

    return np.random.choice(len(softmax), size=1, p=softmax)[0]


import time

RENDER = 0.1
N = 10
temp = 1
DEBUG = 1  # also set in the begining
if __name__ == "__main__":
    b = Board(
        10,
        10,
        R_init_chance=0.5,
        offbalance=1 / 8,  # because our distr rendering can show only 8 vals
    )

    pos = (b.width - 1) // 2
    for _ in range(N):
        per_row_distrs = b.nd_roll_from_pos(pos)

        # FIXME maybe? this ignores "from L / R" out bins, should it be that or should we forward this to the next roll input?
        out_distr_flat = per_row_distrs[-1].sum(0)

        pos = sample_probs_with_temp(out_distr_flat, temp)

        pos = min(pos, b.row_width(0))

        if not DEBUG and RENDER:  # or it's already printed
            print(b.render_with_distr(per_row_distrs))
            time.sleep(RENDER)
        print(pos)
