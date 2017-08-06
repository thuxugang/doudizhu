from __future__ import division
import numpy as np


class UCB1(object):
    """
    The typical bandit upper confidence bounds algorithm.
    """
    def __init__(self, c):
        self.c = c

    def __call__(self, action_node):
        if self.c == 0:  # assert that no nan values are returned
                        # for action_node.n = 0
            return action_node.q

        return (action_node.q +
                self.c * np.sqrt(2 * np.log(action_node.parent.n) /
                                 action_node.n))


def flat(_):
    """
    All actions are considered equally useful
    :param _:
    :return:
    """
    return 0