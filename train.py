# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
from __future__ import absolute_import
from game.agent import Agent
from rl.dqn import DeepQNetwork
from game.rlutil import combine

import numpy as np

#rl
if __name__=="__main__":
    
    step = 0
    num_epochs = 500000
    agent = Agent(models=["rl","random","random"])
    RL = DeepQNetwork(agent.dim_actions, agent.dim_states,num_epochs,
                  learning_rate=0.01,
                  reward_decay=0.9,
                  e_greedy=0.9,
                  replace_target_iter=200,
                  memory_size=2000,
                  # output_graph=True
                  )
    winners = []
    win_rate = 0
    for episode in range(num_epochs):
        # initial observation
        s = agent.reset()
        done = False
        loss = 0
        while(not done):
            
            # RL choose action based on observation
            actions = agent.get_actions_space()
            s = combine(s, actions)
            #action to one-hot
            actions_ont_hot = np.zeros(agent.dim_actions)
            for k in range(len(actions)):
                actions_ont_hot[actions[k]] = 1
                
                
            action, action_id = RL.choose_action(s, actions_ont_hot, actions)
            
            # RL take action and get next observation and reward
            s_, r, done = agent.step(action_id=action_id)
            
            actions_ = agent.get_actions_space_state()
            s_ = combine(s_, actions_) #get_actions_space_state不改变game参数
            
            RL.store_transition(s, action, r, s_)

            if (step > 200) and (step % 5 == 0):
                loss = RL.learn()
                #if step%100 == 0:
                #    print("episode: ",episode,", loss: ", loss, ", win_rate: ",win_rate)

            # swap observation
            s = s_

            step += 1

        if episode%2000 == 0:
            RL.save_model("dqn", episode)
            print("episode: ",episode,", loss: ", loss, ", win_rate: ",win_rate)
                
        if r == 1:
            winners.append(1)
        else:
            winners.append(0)
            
        win_rate = np.mean(winners)

    # end of game
    print('game over')
    RL.plot_cost()





"""
#test
#win_rate = 0.36890000000000001
if __name__=="__main__":
    agent = Agent(models=["rl","random","random"])
    winners = []
    win_rate = 0
    for episode in range(10000):
        s = agent.reset()
        done = False
        while(not done):
            actions = agent.get_actions_space() #如果actions为[]，step()
            s_, r, done = agent.step(action_id=0)
            s = s_
        if r == 1:
            winners.append(1)
        else:
            winners.append(0)
            
        win_rate = np.mean(winners)  
"""



