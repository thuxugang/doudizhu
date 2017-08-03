# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
from __future__ import absolute_import
import tensorflow as tf
import re

def model_init(agent, rl_model, e_greedy=1, start_iter=0):
    
    model = "Model_dqn/"+str(start_iter)+".ckpt"
    #model_new = "Model_sa/"+rl_model+"_"+str(start_iter)+"_"+scope+".ckpt"
    
    #random 70%, 
    if rl_model == "dqn":
        from .dqn_max import DeepQNetwork
        RL = DeepQNetwork(agent.dim_actions, agent.dim_states, e_greedy=e_greedy)
    #random 73%,
    elif rl_model == "prioritized_dqn":
        from .prioritized_dqn_max import DQNPrioritizedReplay
        RL = DQNPrioritizedReplay(agent.dim_actions, agent.dim_states,
                                  e_greedy=e_greedy)
    #cxgz 64.2%  
    elif rl_model == "dueling_dqn":
        from .dueling_dqn_max import DuelingDQN
        RL = DuelingDQN(agent.dim_actions, agent.dim_states, e_greedy=e_greedy)
    
    #load model
    if start_iter !=0: 
        RL.load_model(model)
    return RL


############################################
#          scope中加入新的参数            #
############################################   
"""
500000->500001 pdqn中加入可以自己与前100000局参数对弈
"""
def rescope():
    model_old = "Model_sa/prioritized_dqn_500000.ckpt"
    model_new = "Model_sa/prioritized_dqn_500001.ckpt"
    
    vars = tf.contrib.framework.list_variables(model_old)
    with tf.Graph().as_default(), tf.Session().as_default() as sess:
    
      new_vars = []
      for name, shape in vars:
        v = tf.contrib.framework.load_variable(model_old, name)
        new_vars.append(tf.Variable(v, name=name))
        if re.findall("^eval_net",name) != []:
            new_name = name.replace("eval_net", "eval_net_model")
            new_vars.append(tf.Variable(v, name=new_name))
            print(new_name)
        print(name)
    
      saver = tf.train.Saver(new_vars)
      sess.run(tf.global_variables_initializer())
      saver.save(sess, model_new)
            