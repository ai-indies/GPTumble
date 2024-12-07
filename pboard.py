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
    JAX_KEY = jrandom.PRNGKey(0)
except ImportError:
    np = get_np_module(use_jax=False)
    JAX = False
    JAX_KEY = None
    jnp = False

import numpy as raw_numpy

raw_numpy.set_printoptions(5)

def initialize_random_state(seed=0):
    """Initialize the random state for reproducibility."""
    raw_numpy.random.seed(seed)
    return raw_numpy.random.default_rng(seed=seed)

RNG = initialize_random_state(0)

RED_COLOR = "\033[31m"
GREEN_COLOR = "\033[32m"
REV_COLOR = "\033[7m"
RED_BG = "\033[41m"
RESET_COLOR = "\033[0m"

RENDER_SPACER = "  "

from enum import Enum
from collections import namedtuple
import click

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
    block_chars = ["0", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"]

    def __init__(self, width, height, R_init_chance=False, offbalance=False, np_impl=None, fixed_weights=True, verbose=0) -> None:
        assert height / 2 == height // 2, "board height should be even number"
        self.width = width
        self.height = height
        self.np = np_impl if np_impl is not None else np
        self.fixed_weights = fixed_weights
        self.verbose = verbose

    
        global jnp
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
        """Render the board with distribution of balls.
        
        Args:
            distr: List of distribution arrays, one per row.
                  Each array has shape (2, width) where width matches the row width.
                  Can include one extra row for output distribution.
        """
        # Validate distribution dimensions
        assert len(distr) <= self.height + 1, f"Distribution has too many rows: {len(distr)} > {self.height + 1}"
        for rn, row_distr in enumerate(distr):
            if rn < self.height:
                # For board rows, use the row's width
                expected_width = self.row_width(rn)
            else:
                # For output row, use input row width
                expected_width = self.row_width(0)
            assert row_distr.shape == (2, expected_width), \
                f"Distribution for row {rn} has wrong shape: {row_distr.shape} != (2, {expected_width})"

        res = []
        two_char_rows = self._two_char_board()

        # Render board rows
        for rn, (row_distr, two_char_row) in enumerate(zip(distr[:self.height], two_char_rows)):
            is_wide = self.is_wide_row(rn)
            
            # Render ball distribution if any non-zero probabilities
            if row_distr.any():
                r_row = ""
                if not is_wide:
                    r_row += RENDER_SPACER
                r_row += self._render_ball_distr(row_distr)
                res.append(r_row)

            # Render switch row
            r_row = ""
            if not is_wide:
                r_row += RENDER_SPACER
            r_row += RENDER_SPACER.join(two_char_row)
            res.append(r_row)

        # Render output distribution if present
        if len(distr) > self.height:
            output_distr = distr[-1]
            if output_distr.any():
                r_row = ""
                if not self.is_wide_row(0):  # Use input row width
                    r_row += RENDER_SPACER
                r_row += self._render_ball_distr(output_distr)
                res.append(r_row)

        return "\n".join(res)

    def __repr__(self):
        # FIXME OMG :FP:
        empty_distr_row_n = lambda row_n: self.np.zeros((2,self.row_width(row_n)))
        empty_distr = [empty_distr_row_n(row_n) for row_n in range(0, self.height)]
        return self.render_with_distr(empty_distr)

    def _init_weight(self, main_mode, off_main=0, extra_dims=tuple()):
        if self.fixed_weights:
            return self.np.ones(extra_dims + (self.height, self.width)) * off_main + main_mode * (1 - off_main)
        if self.np is jnp:
            key = jrandom.PRNGKey(0)
            return jrandom.uniform(key, extra_dims + (self.height, self.width)) * off_main + main_mode * (1 - off_main)
        return RNG.random(extra_dims + (self.height, self.width)) * off_main + main_mode * (1 - off_main)

        # np.array(..., dtype=np.float16) # can be used for less verbose debug

    @staticmethod
    def is_wide_row(row_n) -> bool:
        return row_n % 2 == 1

    def row_width(self, row_n) -> bool:
        return self.width if self.is_wide_row(row_n) else (self.width - 1)

    def _row_n_zeros(self, row_n):
        """returns zeros array size: 2 * row_width"""
        if self.verbose >= 2:
            print("row", row_n, "width", self.row_width(row_n))
        return self.np.zeros((2, self.row_width(row_n=row_n)))

    def _roll_one_row(self, in_distr, row_n):
        """
        in_probs - float array of size 2 (sides) * row_width

        returns: float array of size 2 (sides) * next_row_w
        """
        if self.verbose >= 2:
            print("rolling row", row_n)
        out_distr = self._row_n_zeros(row_n + 1)  # it goes to the next row - thus +1

        if self.verbose >= 3:
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

            if self.verbose >= 3:
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

    def _row_probs(self, row_n):
        """
        Using current state of the switches

        returns:
        fall_probs: float array of size 2 (incoming ball side) * row_width
        """
        if self.verbose >= 3:
            print("getting row probs for", row_n)
        # Considering N (row width) switches in states (0s and 1s): s = [s1, ..., sn]
        # and the weight matrix with dimensions:
        # 2 (switch state), 2 (incoming ball side), row, positions

        N = self.row_width(row_n)
        s = self.states[row_n, :N]

        # To make / support non-deterministic states weights should be lin-comb of corresponding state probs

        fall_probs = self._inverse_probs_with_state(
            self.fall_weights[s, ..., row_n, self._to_array(range(N))].swapaxes(0, 1), s
        )

        if self.verbose >= 3:
            print("fall probs", rarr(fall_probs))
        return fall_probs

    @classmethod
    def _render_p_as_bar_char(cls, p):
        return cls.block_chars[int(p*(len(cls.block_chars)-1))]

    @classmethod
    def _render_ball_distr(cls, ball_distr):
        # Normalize the entire array
        ball_distr = ball_distr / max(ball_distr.max(), 1e-10)  # Avoid division by zero
        pairs = []
        for i in range(ball_distr.shape[1]):
            pair = ball_distr[:, i]
            pairs.append(cls._render_p_as_bar_char(pair[0]) + cls._render_p_as_bar_char(pair[1]))
        return "  ".join(pairs)

    def update_states(self, end_pos, ball_distrs, temp):
        if self.verbose >= 2:
            print("updating states")
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

            if self.verbose >= 2:
                print(f"rn {rn}, pos {pos}, probs {rarr(from_l_r_prop)}, argmax {from_which_side}, new {new_pos}")
                print(self)
            pos = new_pos

    def normalize_states(self):
        self.states = self._set_array(self.states, self.states == TO_ZERO_FROM_ONE, 0)
        self.states = self._set_array(self.states, self.states == TO_ONE_FROM_ZERO, 1)

    def run_sim(self, n_steps, temp=0, initial_pos=None, render=0, verbose=None):
        """Run simulation for n_steps with given temperature and initial position.
        
        Args:
            n_steps: Number of simulation steps
            temp: Temperature parameter for probability sampling (default: 0)
            initial_pos: Starting position (default: center of board)
            render: Render verbosity level (default: 0)
                0 - no rendering
                1 - show board with updates
                2 - show board with distribution
            verbose: Verbosity level (default: board.verbose)
            
        Returns:
            List of positions visited during simulation
        """
        history = []
        pos = initial_pos if initial_pos is not None else (self.width - 1) // 2

        verbose = self.verbose if verbose is None else verbose
        if render > 0 and verbose == 0:
            verbose = 1

        if verbose == 0:
            print(pos, end=" ", flush=True)  # Print initial position
        if verbose == 1:
            print("initial pos:", pos)

        for _ in range(n_steps):
            if verbose >= 3:
                print("rolling from", pos)
            per_row_distrs = self.roll_from_pos(pos)

            out_distr = per_row_distrs[-1]
            out_distr_flat = out_distr.reshape(-1)
            
            # Apply temperature and sample
            if temp > 0:
                out_distr_flat = sample_probs_with_temp(out_distr_flat, temp)
            pos = self._sample_choice(len(out_distr_flat), out_distr_flat)
            pos = min(pos, self.row_width(0) - 1)

            self.update_states(pos, per_row_distrs, temp)
            
            if render == 1:  # Show board   
                print(self)
            if render >= 2:  # Show board with distribution
                print(self.render_with_distr(per_row_distrs))

            if verbose == 0:
                print(pos, end=" ", flush=True)  # Print just the position
            elif verbose == 1:
                print("chose:", pos)
            elif self.verbose == 2:
                print("chose", pos, "from", rarr(out_distr_flat))
            elif self.verbose >= 3:
                print("per row distrs:", rarr(per_row_distrs))
                print("out distr:", rarr(per_row_distrs[-1]))
                print("chose:", pos, "from", rarr(out_distr_flat))


            self.normalize_states()
            history.append(int(pos) if isinstance(pos, (int, float)) else int(pos.item()))  # handle both numpy and jax arrays

        if verbose == 0:
            print()  # End the line of positions

        return history

    def _sample_choice(self, size, p=None):
        """Sample from range(size) with given probabilities."""
        if p is None:
            if self.np is jnp:
                key = jrandom.PRNGKey(0)
                return int(jrandom.randint(key, 0, size, ()))
            return RNG.integers(0, size)

        # Normalize probabilities
        p = p / p.sum() if p.sum() > 0 else self.np.ones_like(p) / len(p)

        if self.np is jnp:
            key = jrandom.PRNGKey(0)
            return int(jrandom.choice(key, size, p=p))
        return int(RNG.choice(size, 1, p=p))

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

        # if self.verbose >= 2:
        #     # FIXME should be 'render_row_with_distr'
        #     print(self.render_with_distr(per_row_distrs))

        for row_n in range(self.height):
            prev_distr = per_row_distrs[-1]

            next_distr = self._roll_one_row(prev_distr, row_n)

            # if self.verbose >= 2:
            #     # FIXME should be 'render_row_with_distr'
            #     print(self.render_with_distr([None] * len(per_row_distrs) + [next_distr]))

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
    exp_logits = raw_numpy.array(exp_logits).astype("float64")
    softmax = raw_numpy.array(exp_logits).astype("float64")
    softmax *= mask
    softmax /= softmax.sum()

    if JAX:
        key = jrandom.PRNGKey(0)
        return int(jrandom.choice(key, len(softmax), p=softmax))
    return RNG.choice(len(softmax), 1, p=softmax)


def rarr(arr):
    return " ".join(filter(None, str(arr).split()))


import time

@click.command()
@click.option('--width', '-w', default=10, help='Board width')
@click.option('--height', '-h', default=10, help='Board height')
@click.option('--init-chance-r', '-i', default=0.5, help='Initial chance for R state')
@click.option('--offbalance', '-o', default=1/8, help='Offbalance parameter (should be >= 1/8 for render to work)')
@click.option('--steps', '-n', default=100, help='Number of simulation steps')
@click.option('--temp', '-t', default=0.0, help='Temperature parameter')
@click.option('--render', '-r', count=True, help='Render verbosity: 1 for board, 2 for board+distribution')
@click.option('--verbose', '-v', default=0, count=True, help='Verbosity level (use multiple times for more detail)')
@click.option('--seed', '-s', default=0, help='Random seed for reproducibility')
@click.option('--fixed-weights/--no-fixed-weights', default=True, help='Use fixed weights')
def main(width, height, init_chance_r, offbalance, steps, temp, render, verbose, seed, fixed_weights):
    """Run the tumbling simulation with specified parameters."""
    # Initialize random state with provided seed
    initialize_random_state(seed)
    if JAX:
        global JAX_KEY
        JAX_KEY = jrandom.PRNGKey(seed)

    b = Board(
        width,
        height,
        R_init_chance=init_chance_r,
        offbalance=offbalance,
        fixed_weights=fixed_weights,
        verbose = verbose  
    )

    history = b.run_sim(
        n_steps=steps,
        temp=temp,
        render=render,
    )

    print('JAX', JAX)

if __name__ == "__main__":
    main()
