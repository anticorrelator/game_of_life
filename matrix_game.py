import numpy as np
import time


def shift_matrix_up(m):
    return np.vstack((m[1:], m[0]))


def shift_matrix_down(m):
    return np.vstack((m[-1], m[:-1]))


def shift_matrix_right(m):
    return np.hstack((m[:, -1].reshape(-1, 1), m[:, :-1]))


def shift_matrix_left(m):
    return np.hstack((m[:, 1:], m[:, 0].reshape(-1, 1)))


def vertical_collector(dimension):
    ii = np.identity(dimension, dtype=int)
    return shift_matrix_down(ii) + shift_matrix_up(ii)


def collect_live_neighbors(state):
    operator = vertical_collector(state.shape[0])

    verticals = np.dot(operator, state)
    horizontals = np.dot(state, operator)
    left_diags = np.dot(operator, shift_matrix_right(state))
    right_diags = np.dot(operator, shift_matrix_left(state))
    return horizontals + verticals + left_diags + right_diags


def advance(state):
    neighbors = collect_live_neighbors(state)
    sum_rule = np.array(state + neighbors == 3, dtype=int)
    product_rule = np.array(state * neighbors == 3, dtype=int)
    return sum_rule | product_rule


def random_state(dimension):
    return np.floor(2 * np.random.rand(dimension, dimension))


def draw(state):
    draw_map = {0: " ", 1: "X"}
    for row in state:
        print(str.join(" ", [draw_map[e] for e in row]))
    print("-" * (state.shape[0] * 2 - 1))


def play(state):
    while True:
        state = advance(state)
        draw(state)
        time.sleep(.1)


def random_play(dimension):
    state = random_state(dimension)
    while True:
        state = advance(state)
        draw(state)
        time.sleep(.1)
