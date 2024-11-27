"""
switch numbering in rows:
row
0       0       1       2       3       4
1   0       1       2       3       4       5
2       0       1       2       3       4

Transition from odd row is same/+1, from even row -1/same
"""

def get_np_module(use_jax=True):
    """Return the appropriate numpy module based on the use_jax flag."""
    if use_jax:
        try:
            import jax.numpy as jnp
            return jnp
        except ImportError:
            raise ImportError("JAX is required but not installed.")
    import numpy as np
    return np

try:
    import jax.numpy as jnp
    import jax.random as jrandom
    np = get_np_module(use_jax=True)
    JAX = True
    JAX_KEY = jrandom.PRNGKey(SEED)
except ImportError:
    np = get_np_module(use_jax=False)
    jnp = False
    JAX = False
    JAX_KEY = None

from enum import Enum
import logging
from collections import namedtuple

FIXED_WEIGTHS = True

DEBUG = None
RENDER = False
SEED = 0

import numpy as raw_numpy

raw_numpy.set_printoptions(5)

def initialize_random_state(seed=SEED):
    """Initialize the random state for reproducibility."""
    raw_numpy.random.seed(seed)
    return raw_numpy.random.default_rng(seed=seed)

RNG = initialize_random_state(SEED)

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
    block_chars = raw_numpy.array([" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"])

    def __init__(self, width, height, R_init_chance=False, offbalance=False, np_impl=None) -> None:
        assert height / 2 == height // 2, "board height should be even number"
        self.width = width
        self.height = height
        self.np = np_impl if np_impl is not None else np

        if self.np is jnp:
            global JAX_KEY
            JAX_KEY, subkey = jrandom.split(JAX_KEY)
            self.states = jrandom.choice(
                subkey,
                self.np.array([s.val for s in SIDES]),
                shape=(self.height, self.width),
                p=self.np.array([1 - R_init_chance, R_init_chance]),
            )
        else:
            self.states = RNG.choice(
                self.np.array([s.val for s in SIDES]),
                (self.height, self.width),
                p=self.np.array([1 - R_init_chance, R_init_chance]),
            )

        self.states = self._to_array(self.states, dtype=int)

        # 2 (switch state), 2 (incoming ball side), board h, w
        self.fall_weights = self._init_weight(0, offbalance, extra_dims=(len(SIDES), len(SIDES)))

        # 2 (switch state), 2 (incoming ball side), board h, w
        self.switch_weights = self._init_weight(0, offbalance, extra_dims=(len(SIDES), len(SIDES)))

    def _to_array(self, arr, dtype=None):
        """Convert to array with optional dtype."""
        return self.np.array(arr, dtype=dtype)

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
        return self.render_with_distr(self.np.zeros(self.height))

    def _init_weight(self, main_mode, off_main=0, extra_dims=tuple()):
        if FIXED_WEIGTHS:
            return self.np.ones(extra_dims + (self.height, self.width)) * off_main + main_mode * (1 - off_main)
        
        if self.np is jnp:
            global JAX_KEY
            JAX_KEY, subkey = jrandom.split(JAX_KEY)
            return jrandom.uniform(subkey, extra_dims + (self.height, self.width)) * off_main + main_mode * (1 - off_main)
        return RNG.random(extra_dims + (self.height, self.width)) * off_main + main_mode * (1 - off_main)

        # np.array(..., dtype=np.float16) # can be used for less verbose debug

    @staticmethod
    def is_wide_row(row_n) -> bool:
        return row_n % 2 == 1

    def row_width(self, row_n) -> bool:
        return self.width if self.is_wide_row(row_n) else (self.width - 1)

    def _row_n_zeros(self, row_n):
        """returns zeros array size: 2 * row_width"""
        return self.np.zeros((2, self.row_width(row_n=row_n)))

    def _set_array(self, arr, idx, val):
        """Set value(s) in array at given index/indices."""
        if self.np is jnp:
            return arr.at[idx].set(val)
        arr[idx] = val
        return arr

    def _add_array(self, arr, idx, val):
        """Add value(s) to array at given index/indices."""
        if self.np is jnp:
            return arr.at[idx].add(val)
        arr[idx] += val
        return arr

    def roll_from_pos(self, pos):  # returns each row distributions
        assert pos < self.row_width(0)

        # starting with prob = [0, 0, ..., 1 (pos), 0, ..., 0]
        in_distr = self._row_n_zeros(0)
        in_distr = self._set_array(in_distr, (LEFT_SIDE.val, pos), 0.5)
        in_distr = self._set_array(in_distr, (RIGHT_SIDE.val, pos), 0.5)

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

        fall_probs = self._inverse_probs_with_state(
            self.fall_weights[s, ..., row_n, self._to_array(range(N))].swapaxes(0, 1), s
        )

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
                out_distr = self._add_array(out_distr, (1, slice(None, -1)), l_sided_out_distr)
                out_distr = self._add_array(out_distr, (0, slice(1, None)), r_sided_out_distr)
            else:
                # [:-1] and [1:] denote shift of wide to narrow row
                # TODO FIX MAYBE? This is where balls fall off the board?
                # TODO FIX MAYBE - ball falling off the board behavior is inconsistent between np and jnp
                out_distr = self._add_array(out_distr, (1,), l_sided_out_distr[1:])
                out_distr = self._add_array(out_distr, (0,), r_sided_out_distr[:-1])

        return out_distr

    def _render_ball_distr(self, ball_distr, norm=True):  # norm is meh
        flat = ball_distr.swapaxes(0, 1)
        # optionally norm first
        rescaled = flat / (ball_distr.max() if norm else 1)
        # scale to block char max val
        rescaled *= (len(self.block_chars) - 1) // 1
        rescaled = rescaled.astype(int)

        return RENDER_SPACER.join("".join(self.block_chars[pair]) for pair in rescaled) + (
            " - " + rarr(flat.astype(np.float16)) if DEBUG else ""
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
            if not self.is_wide_row(rn):
                new_pos -= 1

            new_pos = max(new_pos, 0)

            # Basically self.states[rn][new_pos] = 1 - self.states[rn][new_pos]
            if self.states[rn][new_pos] == 1:
                self.states = self._set_array(self.states, (rn, new_pos), TO_ZERO_FROM_ONE)

            elif self.states[rn][new_pos] == 0:
                self.states = self._set_array(self.states, (rn, new_pos), TO_ONE_FROM_ZERO)

            if DEBUG:
                print(f"rn {rn}, pos {pos}, probs {rarr(from_l_r_prop)}, argmax {from_which_side}, new {new_pos}")
                print(self)
            pos = new_pos

    def normalize_states(self):
        self.states = self._set_array(self.states, self.states == TO_ZERO_FROM_ONE, 0)
        self.states = self._set_array(self.states, self.states == TO_ONE_FROM_ZERO, 1)

    def run_sim(self, n_steps, temp=0, initial_pos=None, render=False, debug=None):
        """Run simulation for n_steps with given temperature and initial position.
        
        Args:
            n_steps: Number of simulation steps
            temp: Temperature parameter for probability sampling (default: 0)
            initial_pos: Starting position (default: center of board)
            render: Whether to render simulation steps (default: False)
            debug: Debug mode flag (default: None)
        
        Returns:
            List of positions visited during simulation
        """
        history = []
        pos = initial_pos if initial_pos is not None else (self.width - 1) // 2

        for _ in range(n_steps):
            if debug:
                print("rolling from", pos)
            per_row_distrs = self.roll_from_pos(pos)

            # FIXME maybe? this ignores "from L / R" out bins, should it be that or should we forward this to the next roll input?
            out_distr_flat = per_row_distrs[-1].sum(0)
            pos = sample_probs_with_temp(out_distr_flat, temp)
            pos = min(pos, self.row_width(0) - 1)

            self.update_states(pos, per_row_distrs, temp)
            if not debug and render:  # with updates now
                print(self.render_with_distr(per_row_distrs))

            if debug is not None:
                print("chose", pos, "from", rarr(out_distr_flat))
            else:
                print("chose:", pos)

            if not debug and render:
                time.sleep(RENDER)

            self.normalize_states()
            history.append(int(pos) if isinstance(pos, (int, float)) else int(pos.item()))  # handle both numpy and jax arrays

        return history

    def _sample_choice(self, size, p=None):
        """Sample from range(size) with given probabilities."""
        if self.np is jnp:
            global JAX_KEY
            JAX_KEY, subkey = jrandom.split(JAX_KEY)
            return int(jrandom.choice(subkey, size, p=p))
        return RNG.choice(size, 1, p=p)


def sample_probs_with_temp(probs, temp=1.0):
    if not temp:
        return probs.argmax()

    # We wanna sample only "reachable from input bins"
    # so we use mask for all probs that are === 0
    # so it's non-standard "sample with temp"
    mask = raw_numpy.array(probs != 0.0)

    # Sample according to the adjusted probabilities
    scaled_logits = probs / temp
    # Compute softmax values
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))  # For numerical stability

    softmax = raw_numpy.array(exp_logits).astype("float64")
    softmax *= mask
    softmax /= softmax.sum()

    if JAX:
        global JAX_KEY
        JAX_KEY, subkey = jrandom.split(JAX_KEY)
        return int(jrandom.choice(subkey, len(softmax), p=softmax))
    return RNG.choice(len(softmax), 1, p=softmax)


def rarr(arr):
    return " ".join(filter(None, str(arr).split()))


import time

RENDER = 0#.001
N = 100
temp = 0  # .1
# DEBUG = 0  # also set in the begining
if __name__ == "__main__":
    b = Board(
        15,
        28,
        R_init_chance=0.5,
        offbalance=1/8,  # because our distr rendering can show only 8 vals - this better be >=1/8
    )

    history = b.run_sim(
        n_steps=N,
        temp=temp,
        render=RENDER,
        debug=DEBUG
    )

    # # print(history)
    # from collections import Counter
    # for _ in Counter(zip(history, history[1:])).most_common():
    #     print(_)

    print('JAX', JAX)
