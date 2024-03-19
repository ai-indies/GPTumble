"""
switch numbering in rows:
row
0       0       1       2       3       4
1   0       1       2       3       4       5
2       0       1       2       3       4

Transition from odd row is same/+1, from even row -1/same
"""

try:
    import jax

    JAX = True
    import jax.numpy as np
except:
    JAX = False
    import numpy as np

from enum import Enum
import logging
from collections import namedtuple

FIXED_WEIGTHS = True

DEBUG = None
SEED = 0

import numpy

numpy.random.seed(SEED)
RNG = numpy.random.default_rng(seed=SEED)

RED_COLOR = "\033[31m"
GREEN_COLOR = "\033[32m"
REV_COLOR = "\033[7m"
RED_BG = "\033[41m"
RESET_COLOR = "\033[0m"

RENDER_SPACER = "  "


logger = logging.Logger(f"Board logger", logging.DEBUG if DEBUG else logging.INFO)


Side = namedtuple("Side", ["val", "name"])

LEFT_SIDE = Side(0, "L")
RIGHT_SIDE = Side(1, "R")
SIDES = [LEFT_SIDE, RIGHT_SIDE]

SIDE_VAL_TO_SIDE = {side.val: side for side in SIDES}

# Pseudo states to mark up changes with colors
TO_ZERO_FROM_ONE = -2
TO_ONE_FROM_ZERO = -1

PSEUDO_TO_TRUE_STATE = {TO_ZERO_FROM_ONE: 0, TO_ONE_FROM_ZERO: 1, 0: 0, 1: 1}


class Board:
    block_chars = numpy.array([" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"])

    def __init__(self, width, height, R_init_chance=False, offbalance=False) -> None:
        assert height / 2 == height // 2, "board height should be even number"
        self.width = width
        self.height = height

        self.states = RNG.choice(
            np.array([s.val for s in SIDES]),
            (self.height, self.width),
            p=np.array([1 - R_init_chance, R_init_chance]),
        )

        if JAX:
            self.states = np.array(self.states)

        # 2 (switch state), 2 (incoming ball side), board h, w
        self.fall_weights = self._init_weight(0, offbalance, extra_dims=(len(SIDES), len(SIDES)))

        # 2 (switch state), 2 (incoming ball side), board h, w
        self.switch_weights = self._init_weight(0, offbalance, extra_dims=(len(SIDES), len(SIDES)))

    _switch_render_map = {
        0: f"\\_",
        1: f"_/",
        TO_ZERO_FROM_ONE: f"{RED_BG}_/{RESET_COLOR}",
        TO_ONE_FROM_ZERO: f"{RED_BG}\\_{RESET_COLOR}",
    }

    def _two_char_board(self):
        return [
            [self._switch_render_map[int(pos_state)] for pos_state in row[: self.row_width(rn)]]
            for rn, row in enumerate(self.states)
        ]

    def render_with_distr(self, distr) -> str:
        res = []

        # zip shortest and `[None]` is added for if the distr after the last row need to be rendered
        for rn, (row_distr, two_char_row) in enumerate(zip(distr, self._two_char_board() + [None])):
            if row_distr is None:
                continue  # so we print only needed rows

            is_wide = self.is_wide_row(rn)

            if row_distr.any():
                r_row = ""
                if not is_wide:
                    r_row += RENDER_SPACER
                r_row += self._render_ball_distr(row_distr)
                res.append(r_row)

            if two_char_row:  # or None for the one after last.
                r_row = ""
                if not is_wide:
                    r_row += RENDER_SPACER
                r_row += RENDER_SPACER.join(two_char_row or "")
                res.append(r_row)

        return "\n".join(res)

    def __repr__(self):
        # FIXME OMG :FP:
        return self.render_with_distr(np.zeros(self.height))

    def _init_weight(self, main_mode, off_main=0, extra_dims=tuple()):
        if FIXED_WEIGTHS:
            return np.ones(extra_dims + (self.height, self.width)) * off_main + main_mode * (1 - off_main)

        return RNG.random(extra_dims + (self.height, self.width)) * off_main + main_mode * (1 - off_main)

        # np.array(..., dtype=np.float16) # can be used for less verbose debug

    @staticmethod
    def is_wide_row(row_n) -> bool:
        return row_n % 2 == 1

    def row_width(self, row_n) -> bool:
        return self.width if self.is_wide_row(row_n) else (self.width - 1)

    def _row_n_zeros(self, row_n):
        """returns zeros array size: 2 * row_width"""
        return np.zeros((2, self.row_width(row_n=row_n)))

    def roll_from_pos(self, pos):  # returns each row distributions
        assert pos < b.row_width(0)

        # starting with prob = [0, 0, ..., 1 (pos), 0, ..., 0]
        in_distr = self._row_n_zeros(0)
        if JAX:
            in_distr = in_distr.at[LEFT_SIDE.val, pos].set(0.5)
            in_distr = in_distr.at[RIGHT_SIDE.val, pos].set(0.5)
        else:
            in_distr[LEFT_SIDE.val, pos] = 0.5
            in_distr[RIGHT_SIDE.val, pos] = 0.5

        per_row_distrs = [in_distr]

        if DEBUG:
            # FIXME should be 'render_row_with_distr'
            print(self.render_with_distr(per_row_distrs))

        for row_n in range(self.height):
            prev_distr = per_row_distrs[-1]

            next_distr = self._roll_one_row(prev_distr, row_n)

            if DEBUG:
                # FIXME should be 'render_row_with_distr'
                print(self.render_with_distr([None] * len(per_row_distrs) + [next_distr]))

            per_row_distrs.append(next_distr)

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
        """

        # Considering N (row width) switches in states (0s and 1s): s = [s1, ..., sn]
        # and the weight matrix with dimensions:
        # 2 (switch state), 2 (incoming ball side), row, positions

        N = self.row_width(row_n)
        s = self.states[row_n, :N]

        # To make / support non-deterministic states weights should be lin-comb of corresponding state probs

        if JAX:
            fall_probs = self._inverse_probs_with_state(
                self.fall_weights[s, ..., row_n, np.array(range(N))].swapaxes(0, 1), s
            )
        else:
            fall_probs = self._inverse_probs_with_state(self.fall_weights[s, ..., row_n, range(N)].swapaxes(0, 1), s)

        return fall_probs

    def _roll_one_row(self, in_distr, row_n):
        """
        in_probs - float array of size 2 (sides) * row_width

        returns: float array of size 2 (sides) * next_row_w
        """

        out_distr = self._row_n_zeros(row_n + 1)  # it goes to the next row - thus +1

        if DEBUG and DEBUG > 2:
            print("sided_in_distr, fall_probs, l_sided_out_distr, (1 - fall_probs), r_sided_out_distr")

        # TODO this for loop can possibly be optimized to do all in one pass.
        for side, sided_in_distr, fall_probs in zip(SIDES, in_distr, self._row_probs(row_n)):
            #                          State 0 (left)        State 1 (right)
            #                          From 0  From 1        From 0  From 1
            # Falls to 0 (left)        p00     p01           p10     p11
            # Falls to 1 (right)       1-p00   1-p01         1-p10   1-p11
            #
            # "Normal" switch `=1-state` (no in-direction dependency):
            #   Falls to 0 (left)      0      0              1      1
            #   Falls to 1 (right)     1      1              0      0
            # "Cross" switch `=1-from` (no state dependency):
            #   Falls to 0 (left)      0      1              0      1
            #   Falls to 1 (right)     1      0              1      0
            # "Bounce back" switch `=from` (no state dependency):
            #   Falls to 0 (left)      1      0              1      0
            #   Falls to 1 (right)     0      1              0      1
            # "Always one side" (left) `=0` (no state or direction dependency)
            #   Falls to 0 (left)      1      1              1      1
            #   Falls to 1 (right)     0      0              0      0
            #
            # Could be modeled differently:
            #
            #                          State 0 (left)        State 1 (right)
            #                          From 0  From 1        From 0  From 1
            # Falls opposite of state  p00     p01           p10     p11
            # Falls same side as state 1-p00   1-p01         1-p10   1-p11
            #
            # "Normal" switch (no in-direction dependency):
            #   Falls state opposite   1      1              1      1
            #   Falls state side       0      0              0      0
            # "Cross" switch (no state dependency):
            #   Falls state opposite   1      0              0      1
            #   Falls state side       0      1              1      0
            # "Bounce back" switch (no state dependency):
            #   Falls state opposite   0      1              1      0
            #   Falls state side       1      0              0      1
            # "Always one side" (left) `=0` (no state or direction dependency)
            #   Falls state opposite   0      0              1      1
            #   Falls state side       1      1              0      0
            #
            # This one is nicer to use, because "normal" switch is easily modeled with just 4 weights of 0.
            #
            # ^ Both of these sets of 4 should constitute a basis of all possible switches as linear combos.
            #

            l_sided_out_distr = sided_in_distr * fall_probs
            r_sided_out_distr = sided_in_distr * (1 - fall_probs)

            if DEBUG and DEBUG > 2:
                print(sided_in_distr, fall_probs, l_sided_out_distr, (1 - fall_probs), r_sided_out_distr)

            if not self.is_wide_row(row_n):
                # [:-1] and [1:] denote shift of narrow to wide row
                if JAX:
                    out_distr = out_distr.at[1, :-1].add(l_sided_out_distr)
                    out_distr = out_distr.at[0, 1:].add(r_sided_out_distr)
                else:
                    out_distr[1][:-1] += l_sided_out_distr
                    out_distr[0][1:] += r_sided_out_distr

            else:
                # [:-1] and [1:] denote shift of wide to narrow row
                # TODO FIX MAYBE? This is where balls fall off the board?
                # TODO FIX MAYBE - ball falling off the board behavior is inconsistent between np and jnp
                if JAX:
                    out_distr = out_distr.at[1].add(l_sided_out_distr[1:])
                    out_distr = out_distr.at[0].add(r_sided_out_distr[:-1])
                else:
                    out_distr[1] += l_sided_out_distr[1:]
                    out_distr[0] += r_sided_out_distr[:-1]

        return out_distr

    def _render_ball_distr(self, ball_distr, norm=True):  # norm is meh
        flat = ball_distr.swapaxes(0, 1)
        # optionally norm first
        rescaled = flat / (ball_distr.max() if norm else 1)
        # scale to block char max val
        rescaled *= (len(self.block_chars) - 1) // 1
        rescaled = rescaled.astype(int)

        return RENDER_SPACER.join("".join(self.block_chars[pair]) for pair in rescaled) + (
            " - " + "".join(repr(flat.astype(np.float16))[6:-16].split()) if DEBUG else ""
        )

    def update_states(self, end_pos, ball_distrs, temp):
        pos = end_pos
        for rn, distr in reversed(list(enumerate(ball_distrs, -1))):
            if rn < 0:
                break

            from_l_r_prop = distr[..., pos]

            # 0 means left, 1 means right.
            from_which_side = from_l_r_prop.argmax()  # TODO use resampling temp

            new_pos = pos + from_which_side

            # TODO probably need some of that
            if not self.is_wide_row(rn):  # FIXME boundary bug - wraps around from left to right
                new_pos -= 1

            # Basically self.states[rn][new_pos] = 1 - self.states[rn][new_pos]
            if self.states[rn][new_pos] == 1:
                if JAX:
                    self.states = self.states.at[rn, new_pos].set(TO_ZERO_FROM_ONE)
                else:
                    self.states[rn][new_pos] = TO_ZERO_FROM_ONE

            elif self.states[rn][new_pos] == 0:
                if JAX:
                    self.states = self.states.at[rn, new_pos].set(TO_ONE_FROM_ZERO)
                else:
                    self.states[rn][new_pos] = TO_ONE_FROM_ZERO

            if DEBUG:
                print(f"rn {rn}, pos {pos}, probs {from_l_r_prop}, argmax {from_which_side}, new {new_pos}")
                print(self)
            pos = new_pos

    def normalize_states(self):
        if JAX:
            self.states = self.states.at[self.states == TO_ZERO_FROM_ONE].set(0)
            self.states = self.states.at[self.states == TO_ONE_FROM_ZERO].set(1)
        else:
            self.states[self.states == TO_ZERO_FROM_ONE] = 0
            self.states[self.states == TO_ONE_FROM_ZERO] = 1


def sample_probs_with_temp(probs, temp=1.0):
    if not temp:
        return probs.argmax()

    # Sample according to the adjusted probabilities
    scaled_logits = probs / temp
    # Compute softmax values
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))  # For numerical stability

    softmax = numpy.array(exp_logits).astype("float64")
    softmax /= softmax.sum()  # stupid floating point doesn't normalize from the first time

    return RNG.choice(len(softmax), 1, p=softmax)


import time

RENDER = 0.1
N = 5
temp = 0
DEBUG = 0  # also set in the begining
if __name__ == "__main__":
    b = Board(
        6,
        5,
        R_init_chance=0,
        offbalance=0,  # because our distr rendering can show only 8 vals
    )

    pos = (b.width - 1) // 2
    for _ in range(N):
        if DEBUG:
            print("rolling from", pos)
        per_row_distrs = b.roll_from_pos(pos)

        # FIXME maybe? this ignores "from L / R" out bins, should it be that or should we forward this to the next roll input?
        out_distr_flat = per_row_distrs[-1].sum(0)
        pos = sample_probs_with_temp(out_distr_flat, temp)
        pos = min(pos, b.row_width(0) - 1)

        if DEBUG is not None:
            print("chose", pos, "from", out_distr_flat)
        else:
            print("chose:", pos)

        b.update_states(pos, per_row_distrs, temp)
        if not DEBUG and RENDER:  # with updates now
            print(b.render_with_distr(per_row_distrs))
            time.sleep(RENDER)

        b.normalize_states()
