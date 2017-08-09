from __future__ import print_function
from __future__ import absolute_import

import random
from .utils import rand_max
import copy
from .graph import StateNode
import time
class MCTS(object):
    """
    The central MCTS class, which performs the tree search. It gets a
    tree policy, a default policy, and a backup strategy.
    See e.g. Browne et al. (2012) for a survey on monte carlo tree search
    """
    def __init__(self, tree_policy, default_policy, backup, game):
        self.tree_policy = tree_policy
        self.default_policy = default_policy
        self.backup = backup
        self.game = game
        self.game_bak = copy.deepcopy(self.game)

    def __call__(self, s, n=1000):
        """
        Run the monte carlo tree search.

        :param root: The StateNode
        :param n: The number of roll-outs to be performed
        :return:
        """

        root = StateNode(None, s, self.game)
        
        if root.parent is not None:
            raise ValueError("Root's parent must be None.")
                
        for _ in range(n):
            #selection
            begin = time.time()
            node = _get_next_node(root, self.tree_policy)
            print(time.time()-begin)
            #simulation
            begin = time.time()
            node.reward = self.default_policy(node)
            print(time.time()-begin)
            #print(node.reward)
            #back
            begin = time.time()
            self.backup(node)
            print(time.time()-begin)
            
            begin = time.time()
            root.reset(copy.deepcopy(self.game_bak))
            print(time.time()-begin)
            print("========")
            
        #for i in root.children:
        #    print(root.children[i].__dict__)
        #    for j in root.children[i].children:
        #        print(root.children[i].children[j].__dict__)
        #    print("=======")
        return rand_max(root.children.values(), key=lambda x: x.q).action, rand_max(root.children.values(), key=lambda x: x.q).q


def _expand(state_node):
    action = random.choice(state_node.untried_actions)
    action = state_node.untried_actions[0]
    #print(action)
    return state_node.children[action].sample_state()


def _best_child(state_node, tree_policy):
    best_action_node = rand_max(state_node.children.values(),
                                      key=tree_policy)
    return best_action_node.sample_state()


def _get_next_node(state_node, tree_policy):
    while not state_node.is_terminal():
        if state_node.untried_actions:
            return _expand(state_node)
        else:
            state_node = _best_child(state_node, tree_policy)
    return state_node
