from __future__ import division
from .graph import StateNode, ActionNode


class Bellman(object):
    """
    A dynamical programming update which resembles the Bellman equation
    of value iteration.

    See Feldman and Domshlak (2014) for reference.
    """
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, node):
        """
        :param node: The node to start the backups from
        """
        while node is not None:
            node.n += 1
            if isinstance(node, StateNode):
                node.q = max([x.q for x in node.children.values()])
            elif isinstance(node, ActionNode):
                n = sum([x.n for x in node.children.values()])
                node.q = sum([(self.gamma * x.q + x.reward) * x.n
                              for x in node.children.values()]) / n
            node = node.parent


def monte_carlo(node):
    """
    A monte carlo update as in classical UCT.

    See feldman amd Domshlak (2014) for reference.
    :param node: The node to start the backup from
    """
    r = node.reward
    while node is not None:
        node.n += 1
        node.q = ((node.n - 1)/node.n) * node.q + 1/node.n * r
        node = node.parent
