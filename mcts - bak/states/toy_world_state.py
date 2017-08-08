from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import rv_discrete, entropy
from copy import deepcopy


class ToyWorldAction(object):
    def __init__(self, action):
        self.action = action
        self._hash = 10*(action[0]+2) + action[1]+2

    def __hash__(self):
        return int(self._hash)

    def __eq__(self, other):
        return (self.action == other.action).all()

    def __str__(self):
        return str(self.action)

    def __repr__(self):
        return str(self.action)


class ToyWorld(object):
    def __init__(self, size, information_gain, goal, manual):
        self.size = np.asarray(size)
        self.information_gain = information_gain
        self.goal = np.asarray(goal)
        self.manual = manual


class ToyWorldState(object):
    def __init__(self, pos, world, belief=None):
        self.pos = pos
        self.world = world
        self.actions = [ToyWorldAction(np.array([0, 1])),
                        ToyWorldAction(np.array([0, -1])),
                        ToyWorldAction(np.array([1, 0])),
                        ToyWorldAction(np.array([-1, 0]))]
        if belief:
            self.belief = belief
        else:
            self.belief = dict((a, np.array([1] * 4)) for a in self.actions)

    def _correct_position(self, pos):
        upper = np.min(np.vstack((pos, self.world.size)), 0)
        return np.max(np.vstack((upper, np.array([0, 0]))), 0)

    def perform(self, action):
        # get distribution about outcomes
        probabilities = self.belief[action] / np.sum(self.belief[action])
        distrib = rv_discrete(values=(range(len(probabilities)),
                                      probabilities))

        # draw sample
        sample = distrib.rvs()

        # update belief accordingly
        belief = deepcopy(self.belief)
        belief[action][sample] += 1

        # manual found
        if (self.pos == self.world.manual).all():
            print("m", end="")
            belief = {ToyWorldAction(np.array([0, 1])): [50, 1, 1, 1],
                      ToyWorldAction(np.array([0, -1])): [1, 50, 1, 1],
                      ToyWorldAction(np.array([1, 0])): [1, 1, 50, 1],
                      ToyWorldAction(np.array([-1, 0])): [1, 1, 1, 50]}

        # build next state
        pos = self._correct_position(self.pos + self.actions[sample].action)

        return ToyWorldState(pos, self.world, belief)

    def real_world_perform(self, action):
        # update belief accordingly
        belief = deepcopy(self.belief)
        if (action.action == np.array([0, 1])).all():
            real_action = 0
        elif (action.action == np.array([0, -1])).all():
            real_action = 1
        elif (action.action == np.array([1, 0])).all():
            real_action = 2
        elif (action.action == np.array([-1, 0])).all():
            real_action = 3
        belief[action][real_action] += 1

        # manual found
        if (self.pos == self.world.manual).all():
            print("M", end="")
            belief = {ToyWorldAction(np.array([0, 1])): [50, 1, 1, 1],
                      ToyWorldAction(np.array([0, -1])): [1, 50, 1, 1],
                      ToyWorldAction(np.array([1, 0])): [1, 1, 50, 1],
                      ToyWorldAction(np.array([-1, 0])): [1, 1, 1, 50]}

        pos = self._correct_position(self.pos + action.action)
        return ToyWorldState(pos, self.world, belief)

    def is_terminal(self):
        return False

    def __eq__(self, other):
        return (self.pos == other.pos).all()

    def __hash__(self):
        return int(self.pos[0]*100 + self.pos[1])

    def __str__(self):
        return str(self.pos)

    def __repr__(self):
        return str(self.pos)

    def reward(self, parent, action):
        if (self.pos == self.world.goal).all():
            print("g", end="")
            return 100
        else:
            reward = -1
            if self.world.information_gain:
                for a in self.actions:
                    reward += entropy(parent.belief[a], self.belief[a])
            return reward
