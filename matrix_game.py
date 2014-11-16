import numpy as np
import time


def rotate_u(matrix):
    return np.vstack((matrix[1:], matrix[0]))


def rotate_d(matrix):
    return np.vstack((matrix[-1], matrix[:-1]))


def rotate_l(matrix):
    return np.hstack((matrix[:, 1:], matrix[:, 0].reshape(-1, 1)))


def rotate_r(matrix):
    return np.hstack((matrix[:, -1].reshape(-1, 1), matrix[:, :-1]))


def vertical_collector(dimension):
    ii = np.identity(dimension, dtype=int)
    return rotate_d(ii) + rotate_u(ii)


def neighbor_transform(state):
    operator = vertical_collector(state.shape[0])

    verticals = np.dot(operator, state)
    horizontals = np.dot(state, operator)
    left_diags = np.dot(operator, rotate_r(state))
    right_diags = np.dot(operator, rotate_l(state))
    return horizontals + verticals + left_diags + right_diags


def advance(state):
    neighbor_sum = neighbor_transform(state)
    sum_rule = state + neighbor_sum == 3
    product_rule = state * neighbor_sum == 3
    return np.array(sum_rule | product_rule, dtype=int)


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
