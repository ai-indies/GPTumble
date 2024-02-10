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
import random

DEBUG = 0


logger = logging.Logger(f"Board logger", logging.DEBUG if DEBUG else logging.INFO)


Side = namedtuple("Side", ["val", "name"])

LEFT_SIDE = Side(0, "L")
RIGHT_SIDE = Side(1, "R")
SIDES = [LEFT_SIDE, RIGHT_SIDE]

SIDE_VAL_TO_SIDE = {side.val: side for side in SIDES}


class Board:
    rng = np.random.default_rng()
    block_chars = np.array([" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"])

    def __init__(self, width, height, R_init_chance=False, offbalance=False) -> None:
        self.width = width
        self.height = height

        self.nd_states = np.random.choice(
            [s.val for s in SIDES], size=(self.width, self.height), p=[1 - R_init_chance, R_init_chance]
        )

        # converting to normal list so we can store the RIGH/LEFT_SIDE tuples directly
        self.states = [[SIDE_VAL_TO_SIDE[pos] for pos in row] for row in self.nd_states]

        # 2 (switch state), 2 (incoming ball side), board height, N
        self.fall_weights = self._init_weight(0, offbalance, extra_dims=(len(SIDES), len(SIDES)))
        self.switch_weights = self._init_weight(0, offbalance, extra_dims=(len(SIDES), len(SIDES)))

    def _render(self, ball_pos=None) -> str:
        board_as_slash = self._one_char_board()
        if ball_pos:
            board_as_slash[ball_pos[0]] = (
                board_as_slash[ball_pos[0]][: ball_pos[1] + ball_pos[0] % 2]
                + ["*"]
                + board_as_slash[ball_pos[0]][ball_pos[1] + ball_pos[0] % 2 :]
            )
        return "\n".join(
            ("" if self.is_wide_row(rn) else " ") + " ".join(row) for rn, row in enumerate(board_as_slash)
        ).replace(" * ", "*")

    @staticmethod
    def _state_to_slash(state):
        return "\\" if state.name == "L" else "/"

    def _one_char_board(self):
        return [[self._state_to_slash(pos) for pos in row[: self.row_width(rn)]] for rn, row in enumerate(self.states)]

    _switch_render_map = {0: "\\_", 1: "_/"}

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
        return self._render()

    def roll_from_w(self, w) -> int:
        in_row_pos = w
        last_move = RIGHT_SIDE
        if DEBUG:
            print("Input:")
            print(" " + (" |" * w + "*" + "| " * (self.width - w - 1)))  # ball over the board
            print(self._render(), "\n")

        for row in range(self.height):
            move = self._roll_w_h(in_row_pos, row, last_move)

            if self.is_wide_row(row):  # wide row
                in_row_pos += move.val - 1
            else:
                in_row_pos += move.val

            if DEBUG:
                print(self._render((row, in_row_pos)), "\n")

        return in_row_pos

    def _roll_w_h(self, w, h, last_shift: Side) -> Side:
        if self.is_wide_row(h):
            if w == 0:
                return RIGHT_SIDE
            if w == self.width - 1:
                return LEFT_SIDE

        old_state = state = self.states[h][w]

        param_state = self.params[state]

        new_param_pos = param_state.get("ball_from_" + last_shift.name)

        to_l_prob = new_param_pos.get("ball_to_l")[h][w]
        to_R_prob = new_param_pos.get("to_state_R")[h][w]

        side_roll = self.rng.random()
        state_roll = self.rng.random()

        new_state = state = LEFT_SIDE if state_roll > to_R_prob else RIGHT_SIDE

        # print(f"- new {h}:{w} state {'<' if state == 'L' else '>'}")

        self.states[h][w] = new_state

        move = RIGHT_SIDE if side_roll > to_l_prob else LEFT_SIDE

        if DEBUG:
            print(f"{h}:{w} {self._state_to_slash(old_state)} => {self._state_to_slash(new_state)}, move: {move.name}")

        return move

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
        # flat and optionally norm'ed
        flat = ball_distr.swapaxes(0, 1) / (ball_distr.max() if norm else 1)
        # scale to block char max val
        rescaled = (flat * (len(self.block_chars) - 1) // 1).astype(int)

        return (
            "  ".join("".join(self.block_chars[pair]) for pair in rescaled)
            + (" - " + "".join(repr(flat).split()) if DEBUG else "")
            + "\n"
        )


def softmax_with_temp(logits, temperature=1.0):
    # Adjust logits according to the temperature
    scaled_logits = logits / temperature
    # Compute softmax values
    exp_logits = np.exp(scaled_logits - np.max(scaled_logits))  # For numerical stability
    return exp_logits / np.sum(exp_logits)


def sample_probs(probs, size=1):
    # Sample according to the adjusted probabilities
    return np.random.choice(len(probs), size=size, p=probs)


import time

N = 10
temp = 0
if __name__ == "__main__":
    b = Board(
        20,
        20,
        R_init_chance=0.5,
        offbalance=1 / 8,  # because our distr rendering can show only 8 vals
    )

    pos = (b.width - 1) // 2
    for _ in range(N):
        per_row_distrs = b.nd_roll_from_pos(pos)
        out_distr_flat = per_row_distrs[-1].sum(0)
        if temp:
            pos = sample_probs(softmax_with_temp(out_distr_flat, temp))[0]
        else:
            pos = out_distr_flat.argmax()

        if not DEBUG:  # or it's already printed
            print(b.render_with_distr(per_row_distrs))
            time.sleep(1)
