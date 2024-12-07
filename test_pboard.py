import pytest
import numpy as np
from pboard import Board, LEFT_SIDE, RIGHT_SIDE


def test_render_ball_distr():
    # Test rendering with zeros and ones
    rendered = Board._render_ball_distr(np.array([[0, 1], [1, 0]]))
    assert rendered == "0█  █0"

    # Test rendering with equal values
    rendered = Board._render_ball_distr(np.ones((2, 3)))
    assert rendered == "██  ██  ██"


    # Test rendering with ascending sequence
    arr = np.array([
        [0.0, 0.125, 0.375, 0.625, 0.875],
        [0.0, 0.25, 0.5, 0.75, 1.0]
    ])
    rendered = Board._render_ball_distr(arr)
    assert rendered == "00  ▁▂  ▃▄  ▅▆  ▇█"



def test_render_with_distr_empty_board():
    board = Board(width=3, height=2)
    distr = [np.zeros((2, 2)), np.zeros((2, 3))]  # 2 rows: narrow(2) and wide(3)
    rendered = board.render_with_distr(distr)
    expected = ["  \\_  \\_",
                "\\_  \\_  \\_"]
    assert rendered.split("\n") == expected

def test_render_with_distr_with_ball():
    board = Board(width=3, height=2)
    # Ball in leftmost position of first row
    distr = [
        np.array([[1.0, 0], [0, 0]]),  # Left side, first position
        np.zeros((2, 3))
    ]
    rendered = board.render_with_distr(distr)
    expected = ["  █0  00",
                "  \\_  \\_",
                "\\_  \\_  \\_",
    ]
    assert rendered.split("\n") == expected

def test_render_with_distr_with_split_ball():
    board = Board(width=3, height=2)
    # Ball split between positions
    distr = [
        np.array([[0.25, 0], [0.25, 0]]),  # Split between left and right
        np.zeros((2, 3))
    ]
    rendered = board.render_with_distr(distr)
    expected = "  ██  00\n  \\_  \\_\n\\_  \\_  \\_"
    assert rendered == expected

def test_render_with_distr_multiple_rows():
    board = Board(width=3, height=4)
    distr = [
        np.array([[1.0, 0], [0, 0]]),  # First row (narrow)
        np.array([[0, 0, 0], [0, 1.0, 0]]),  # Second row (wide)
        np.array([[0, 0], [0, 0]]),  # Third row (narrow)
        np.array([[0, 0, 0], [0, 0, 0]])  # Fourth row (wide)
    ]
    rendered = board.render_with_distr(distr)
    expected = (
        "  █0  00\n"
        "  \\_  \\_\n"
        "00  0█  00\n"
        "\\_  \\_  \\_\n"
        "  \\_  \\_\n"
        "\\_  \\_  \\_"
    )
    assert rendered == expected



def test_render_with_distr_invalid_width():
    board = Board(width=3, height=2)
    # Distribution array wider than board width
    distr = [np.zeros((2, 4)), np.zeros((2, 5))]
    with pytest.raises(AssertionError):
        board.render_with_distr(distr)

def test_render_with_distr_invalid_height():
    board = Board(width=3, height=2)
    # More distribution rows than board height
    distr = [np.zeros((2, 2)), np.zeros((2, 3)), np.zeros((2, 2)), np.zeros((2, 2))]
    with pytest.raises(AssertionError, match=r"Distribution has too many rows"):
        board.render_with_distr(distr)

def test_render_with_distr_output_row():
    board = Board(width=3, height=2)
    distr = [
        np.array([[0, 0], [0, 0]]),  # First row (narrow)
        np.array([[0, 0, 0], [0, 0, 0]]),  # Second row (wide)
        np.array([[0.5, 0], [0.5, 0]])  # Output row (narrow, same as input)
    ]
    rendered = board.render_with_distr(distr)
    expected = (
        "  \\_  \\_\n"
        "\\_  \\_  \\_\n"
        "  ██  00"  # Output row should show equal probabilities
    )
    assert rendered == expected
