"""
Q-learning Functions
"""

import sys
import numpy as np


def pick_strategies(game, s, t):
    """Pick strategies by exploration vs exploitation"""
    a = np.zeros(game.n).astype(int)
    pr_explore = np.exp(- t * game.beta)
    e = (pr_explore > np.random.rand(game.n))
    for n in range(game.n):
        if e[n]:
            a[n] = np.random.randint(0, game.k)
        else:
            a[n] = np.argmax(game.Q[(n,) + tuple(s)])
    return a


def update_q(game, s, a, s1, pi):
    """Update Q matrix"""
    for n in range(game.n):
        subj_state = (n,) + tuple(s) + (a[n],)
        old_value = game.Q[subj_state]
        max_q1 = np.max(game.Q[(n,) + tuple(s1)])
        new_value = pi[n] + game.delta * max_q1
        old_argmax = np.argmax(game.Q[(n,) + tuple(s)])
        game.Q[subj_state] = (1 - game.alpha) * old_value + game.alpha * new_value
        # Check stability
        new_argmax = np.argmax(game.Q[(n,) + tuple(s)])
        same_argmax = (old_argmax == new_argmax)
        game.stable = (game.stable + same_argmax) * same_argmax
    return game


def check_convergence(game, t, tstable, tmax):
    """Check if game converged"""
    if (t % tstable == 0) & (t > 0):
        sys.stdout.write("\rt=%i" % t)
        sys.stdout.flush()
    if game.stable > tstable:
        print('Converged!')
        return True
    if t == tmax:
        print('ERROR! Not Converged!')
        return True
    return False


def simulate_game(game, tstable=1e5, tmax=1e7):
    """
    Simulate game

    Parameters
    -----
    game: game
        the game
    tmax: int
        maximum number of iterations performed
    tstable: int
        minimum number of iterations without updates needed to consider the game stable
    """
    s = game.s0
    # Iterate until convergence
    for t in range(int(tmax)):
        a = pick_strategies(game, s, t)
        pi = game.PI[tuple(a)]
        s1 = a
        game = update_q(game, s, a, s1, pi)
        s = s1
        if check_convergence(game, t, tstable, tmax):
            break
    return game
