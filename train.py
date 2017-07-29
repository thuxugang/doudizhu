# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
from __future__ import absolute_import
from game.agent import Agent
import numpy as np
from game.rlutil import combine

#rl
if __name__=="__main__":
    
    step = 0
    num_epochs = 2000001
    rl_model = "prioritized_ddpg"
    start_iter= 0
    
    learning_rate = 0.001
    e_greedy = 1
    
    dim_actions, dim_states = 431, 464
    #random 70%, 
    if rl_model == "dqn":
        from rl.dqn_max import DeepQNetwork
        RL = DeepQNetwork(dim_actions, dim_states,
                      learning_rate=learning_rate,
                      reward_decay=0.9,
                      e_greedy=e_greedy,
                      replace_target_iter=200,
                      memory_size=2000,
                      )
    #random 73%,
    elif rl_model == "prioritized_dqn":
        from rl.prioritized_dqn_max import DQNPrioritizedReplay
        RL = DQNPrioritizedReplay(dim_actions, dim_states,
                      learning_rate=learning_rate,
                      reward_decay=0.9,
                      e_greedy=e_greedy,
                      replace_target_iter=200,
                      memory_size=2000,
                      prioritized=True
                      )
    #cxgz 64.2%  
    elif rl_model == "dueling_dqn":
        from rl.dueling_dqn_max import DuelingDQN
        RL = DuelingDQN(dim_actions, dim_states,
                      learning_rate=learning_rate,
                      reward_decay=0.9,
                      e_greedy=e_greedy,
                      replace_target_iter=200,
                      memory_size=2000,
                      dueling=True
                      )
        
    elif rl_model == "ddpg":
        from rl.ddpg_max import DDPG
        RL = DDPG(dim_actions, dim_states,
                      lr_a=learning_rate,
                      lr_c=learning_rate,
                      )
    elif rl_model == "prioritized_ddpg":
        from rl.prioritized_ddpg_max import DDPGPrioritized
        RL = DDPGPrioritized(dim_actions, dim_states,
                      lr_a=learning_rate,
                      lr_c=learning_rate,
                      prioritized=True
                      )
        
    #fine-tune
    if start_iter!=0:
        RL.load_model(rl_model, start_iter)
    
    agent = Agent(models=["rl","random","random"], train=True, RL=RL)
    
    winners = np.zeros(3)
    win_rate = 0
    for episode in range(start_iter, num_epochs):
        # initial observation
        s = agent.reset()
        done = False
        loss = 0
        while(not done):
            
            # RL choose action based on observation
            actions = agent.get_actions_space()
            s = combine(s, actions)
            #action to one-hot
            actions_one_hot = np.zeros(agent.dim_actions)
            for k in range(len(actions)):
                actions_one_hot[actions[k]] = 1
                
            action, action_id = RL.choose_action(s, actions_one_hot, actions)
            
            action_one_hot = np.zeros(agent.dim_actions)
            action_one_hot[action] = 1
                
            # RL take action and get next observation and reward
            s_, r, done = agent.step(action_id=action_id)
            
            actions_ = agent.get_actions_space_state()
            #action to one-hot
            actions_one_hot_ = np.zeros(agent.dim_actions)
            for k in range(len(actions_)):
                actions_one_hot_[actions_[k]] = 1
                
            s_ = combine(s_, actions_) #get_actions_space_state不改变game参数
            
            RL.store_transition(s, actions_one_hot, actions_one_hot_, action_one_hot, r, s_)

            if (step > 200) and (step % 5 == 0):
                loss = RL.learn()

            # swap observation
            s = s_

            step += 1
        
        if agent.game.playrecords.winner == 1:
            winners[0] = winners[0] + 1
        elif agent.game.playrecords.winner == 2:
            winners[1] = winners[1] + 1
        elif agent.game.playrecords.winner == 3:
            winners[2] = winners[2] + 1

        
        win_rate = winners/(episode-start_iter)
        #print(agent.game.get_record().records)
        #print(r)
        if episode%2000 == 0:
            #保存模型
            if episode%50000 == 0 and episode != start_iter:
                RL.save_model(rl_model, episode)
                print("save: ",episode)
            print("episode: ",episode,", loss: ", loss, ", win_rate: ",win_rate)
            
            
    # end of game
    print('game over')

