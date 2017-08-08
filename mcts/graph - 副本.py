from __future__ import absolute_import
from game.rlutil import get_state, get_actions, combine_mcts

class Node(object):
    def __init__(self, parent):
        self.parent = parent
        self.children = {}
        self.q = 0
        self.n = 0


class ActionNode(Node):
    """
    A node holding an action in the tree.
    """
    def __init__(self, parent, action):
        super(ActionNode, self).__init__(parent)
        self.action = action
        self.n = 0

    def sample_state(self):
        """
        Samples a state from this action and adds it to the tree if the
        state never occurred before.

        :param real_world: If planning in belief states are used, this can
        be set to True if a real world action is taken. The belief is than
        used from the real world action instead from the belief state actions.
        :return: The state node, which was sampled.
        """
        state = self.parent.perform(self.action)
        if state not in self.children:
            self.children[state] = StateNode(self, state, self.parent.game)

        return self.children[state]

    def __str__(self):
        return "Action: {}".format(self.action)


class StateNode(Node):
    """
    A node holding a state in the tree.
    """
    def __init__(self, parent, state, game):
        super(StateNode, self).__init__(parent)
        self.state = state
        self.reward = 0
        self.game = game
        
        next_move_types, next_moves = game.get_next_moves()
        self.actions = get_actions(next_moves, game.actions_lookuptable, game)

        for action in self.actions:
            self.children[action] = ActionNode(self, action)

    @property
    def untried_actions(self):
        """
        All actions which have never be performed
        :return: A list of the untried actions.
        """
        return [a for a in self.children if self.children[a].n == 0]

    @untried_actions.setter
    def untried_actions(self, value):
        raise ValueError("Untried actions can not be set.")

    def reward(self, parent, action):
        if self.game.playrecords.winner == 0:
            return 0
        elif self.game.playrecords.winner == 1:
            return 1
        else:
            return -1  
          
    def perform(self, action):
        
        if action in [429, 430]:
            action_id = action
        else:
            action_id = self.actions.index(action)
        
        while(self.game.i <= 2):
            self.game.get_next_moves()
            self.game.last_move_type, self.game.last_move, self.game.end, self.game.yaobuqi = self.game.players[self.game.i].play(self.game.last_move_type, self.game.last_move, self.game.playrecords, action_id)
            if self.game.yaobuqi:
                self.game.yaobuqis.append(self.game.i)
            else:
                self.game.yaobuqis = []
            #都要不起
            if len(self.game.yaobuqis) == 2:
                self.game.yaobuqis = []
                self.game.last_move_type = self.game.last_move = "start"
            if self.game.end:
                self.game.playrecords.winner = self.game.i+1
                break
            self.game.i = self.game.i + 1
        #一轮结束
        self.game.i = 0             
        
        #s_
        s_ = get_state(self.game.playrecords, player=1)
        #action
        next_move_types, next_moves = self.game.get_next_moves()
        actions = get_actions(next_moves, self.game.actions_lookuptable, self.game)
        s_ = combine_mcts(s_, actions)
    
        return s_
    
    def is_terminal(self):
        if self.game.playrecords.winner == 0:
            return False
        else:
            return True
    
    def __str__(self):
        return "State: {}".format(self.state)


def breadth_first_search(root, fnc=None):
    """
    A breadth first search (BFS) over the subtree starting from root. A
    function can be run on all visited nodes. It gets the current visited
    node and a data object, which it can update and should return it. This
    data is returned by the function but never altered from the BFS itself.
    :param root: The node to start the BFS from
    :param fnc: The function to run on the nodes
    :return: A data object, which can be altered from fnc.
    """
    data = None
    queue = [root]
    while queue:
        node = queue.pop(0)
        data = fnc(node, data)
        for child in node.children.values():
            queue.append(child)
    return data


def depth_first_search(root, fnc=None):
    """
    A depth first search (DFS) over the subtree starting from root. A
    function can be run on all visited nodes. It gets the current visited
    node and a data object, which it can update and should return it. This
    data is returned by the function but never altered from the DFS itself.
    :param root: The node to start the DFS from
    :param fnc: The function to run on the nodes
    :return: A data object, which can be altered from fnc.
    """
    data = None
    stack = [root]
    while stack:
        node = stack.pop()
        data = fnc(node, data)
        for child in node.children.values():
            stack.append(child)
    return data


def get_actions_and_states(node):
    """
    Returns a tuple of two lists containing the action and the state nodes
    under the given node.
    :param node:
    :return: A tuple of two lists
    """
    return depth_first_search(node, _get_actions_and_states)


def _get_actions_and_states(node, data):
    if data is None:
        data = ([], [])

    action_nodes, state_nodes = data

    if isinstance(node, ActionNode):
        action_nodes.append(node)
    elif isinstance(node, StateNode):
        state_nodes.append(node)

    return action_nodes, state_nodes