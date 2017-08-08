# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
from __future__ import absolute_import
from game.agent import Agent
import numpy as np
from game.rlutil import combine
from rl.init_model import model_init
from game.config import Config
#rl
if __name__=="__main__":
    
    step = 0
    num_epochs = 1000
    rl_model = "prioritized_dqn"
    start_iter=0
    
    my_config = Config()

    agent = Agent(models=["mcts","random","random"], my_config=my_config, RL=None, train=True)
    
    losss = []
    winrates = []
    es = []
    
    f = open('log.txt', 'a')
    winners = np.zeros(3)
    win_rate = 0
    learn_step_counter = 0
    loss = 0
    for episode in range(start_iter, num_epochs):
        # initial observation
        s = agent.reset()
        if episode%2000 == 0:
            print(agent.game.playrecords.show("==================="+str(episode)+"==================="))
        done = False
        while(not done):
            # RL choose action based on observation
            actions = agent.get_actions_space()
                
            # RL take action and get next observation and reward
            s_, r, done = agent.step()

            step += 1
        
        if agent.game.playrecords.winner == 1:
            winners[0] = winners[0] + 1
        elif agent.game.playrecords.winner == 2:
            winners[1] = winners[1] + 1
        elif agent.game.playrecords.winner == 3:
            winners[2] = winners[2] + 1

        
        win_rate = winners/np.sum(winners)
        #print(agent.game.get_record().records)
        #print(r)
        if episode%2000 == 0:
            losss.append(loss)
            winrates.append(win_rate[0])
        #    winners = np.zeros(3)
            
        if episode%2000 == 0:
            print("episode: ",episode, ", win_rate: ",win_rate)
            print(agent.game.get_record().records)
            
            f.write("episode: "+ str(episode) +  ", win_rate: "+ str(win_rate))
            f.write("\n")
            #f.write(str(agent.game.get_record().records))
            #f.write("\n")
            f.flush()
            
    # end of game
    print('game over')
    f.close()
    