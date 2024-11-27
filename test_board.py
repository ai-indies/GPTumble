import numpy as np
from pboard import Board, initialize_random_state, get_np_module
import jax
SEED = 42

def create_board(np_impl):
    """Helper to create a board with consistent initial state"""
    initialize_random_state(seed=42)  # Reset RNG for consistent board init
    return Board(
        width=5,
        height=4,
        R_init_chance=0.5,
        offbalance=0,
        np_impl=np_impl
    )

def test_sim_determinism_numpy():
    """Test that numpy simulation is deterministic"""
    board1 = create_board(np)
    history1 = board1.run_sim(n_steps=2)
    
    board2 = create_board(np)
    history2 = board2.run_sim(n_steps=2)
    
    assert history1 == history2, "Numpy simulation should be deterministic"

def test_sim_determinism_jax():
    """Test that JAX simulation is deterministic (if available)"""
    try:
        jnp = get_np_module(use_jax=True)
    except ImportError:
        print("JAX not available, skipping test")
        return

    board1 = create_board(jnp)
    history1 = board1.run_sim(n_steps=2)

    board2 = create_board(jnp)
    history2 = board2.run_sim(n_steps=2)

    assert history1 == history2, "JAX simulation should be deterministic"
    
    # Also verify JAX and numpy give same results
    numpy_board = create_board(np)
    numpy_history = numpy_board.run_sim(n_steps=2)
    assert history1 == numpy_history, "JAX and numpy simulations should give same results"