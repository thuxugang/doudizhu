import random

def random_terminal_roll_out(state_node):
    """
    Estimate the reward with the sum of a rollout till a terminal state.
    Typical for terminal-only-reward situations such as games with no
    evaluation of the board as reward.

    :param state_node:
    :return:
    """
    def stop_terminal(state):
        return state.is_terminal()

    return _roll_out(state_node, stop_terminal)


def _roll_out(state_node, stopping_criterion):
    reward = 0
    state = state_node
    action = state_node.actions
    while not stopping_criterion(state):
        reward += state.reward
        
        #random
        action = random.choice(state.actions)
        parent = state
        state = parent.perform(action)

    return reward
