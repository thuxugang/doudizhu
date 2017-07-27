# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
from __future__ import absolute_import
from game.agent import Agent
from rl.model import model_init
import numpy as np

#rl
if __name__=="__main__":
    
    step = 0
    num_epochs = 1000
    rl_model = "prioritized_dqn"
    start_iter=500000
    
    RL = model_init(rl_model, start_iter)
    agent = Agent(models=["random","xgmodel","xgmodel"], RL=RL)
    
    winners = []
    win_rate = 0
    for episode in range(num_epochs):
        # initial observation
        s = agent.reset()
        #print(agent.game.playrecords.show("========"))
        done = False
        loss = 0
        while(not done):
            
            #print(agent.game.get_record())
            # RL choose action based on observation
            actions = agent.get_actions_space()
            #print(actions)
            #action to one-hot
            actions_ont_hot = np.zeros(agent.dim_actions)
            for k in range(len(actions)):
                actions_ont_hot[actions[k]] = 1
                
            action, action_id = RL.choose_action(s, actions_ont_hot, actions)
            #print(action)
            # RL take action and get next observation and reward
            s_, r, done = agent.step(action_id=action_id)

            # swap observation
            s = s_

            step += 1

        if r == 1:
            winners.append(1)
        else:
            winners.append(0)
            
        win_rate = np.mean(winners)
        #print(agent.game.get_record().records)
        #print(r)
            
    # end of game
    print('game over')


