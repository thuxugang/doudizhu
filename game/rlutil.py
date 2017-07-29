# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
from __future__ import print_function
import numpy as np   

############################################
#                   LR相关                 #
############################################   
def get_state(playrecords, player):
    state = np.zeros(33+431).astype("int") #33+dim_action
    #手牌
    if player == 1:
        cards_left = playrecords.cards_left1
        state[30] = len(playrecords.cards_left1)
        state[31] = len(playrecords.cards_left2)
        state[32] = len(playrecords.cards_left3)
    elif player == 2:
        cards_left = playrecords.cards_left2
        state[30] = len(playrecords.cards_left2)
        state[31] = len(playrecords.cards_left3)
        state[32] = len(playrecords.cards_left1)
    else:
        cards_left = playrecords.cards_left3
        state[30] = len(playrecords.cards_left3)
        state[31] = len(playrecords.cards_left1)
        state[32] = len(playrecords.cards_left2)
    for i in cards_left:
        state[i.rank - 1] += 1
    #底牌
    for cards in playrecords.records:
        if cards[1] in ["buyao","yaobuqi"]:
            continue
        for card in cards[1]:
            state[card.rank - 1 + 15] += 1  
          
    return state    

def get_actions(next_moves, actions_lookuptable, game):
    """
    0-14: 单出， 1-13，小王，大王
    15-27: 对，1-13
    28-40: 三，1-13
    41-196: 三带1，先遍历111.2，111.3，一直到131313.12
    197-352: 三带2，先遍历111.22,111.33,一直到131313.1212
    353-366: 炸弹，1111-13131313，加上王炸
    367-402: 先考虑5个的顺子，按照顺子开头从小到达进行编码，共计8+7+..+1=36
    430: yaobuqi
    429: buyao
    """
    actions = []
    for cards in next_moves:
        key = []
        for card in cards:
            key.append(int(card.name))
        key.sort()
        actions.append(actions_lookuptable[str(key)])
    
    #yaobuqi
    if len(actions) == 0:
        actions.append(430)
    #buyao
    elif game.last_move != "start":
        actions.append(429)
        
    return actions

#结合state和可以出的actions作为新的state    
def combine(s, a):
    for i in a:
        s[33+i] = 1
    return s
    
############################################
#                  改变scope               #
############################################      
    
def rename(checkpoint_dir, replace_from, replace_to, add_prefix, dry_run=False):
    checkpoint = tf.train.get_checkpoint_state(checkpoint_dir)
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            # Load the variable
            var = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

            # Set the new name
            new_name = var_name
            if None not in [replace_from, replace_to]:
                new_name = new_name.replace(replace_from, replace_to)
            if add_prefix:
                new_name = add_prefix + new_name

            if new_name == var_name:
                continue

            if dry_run:
                print('%s would be renamed to %s.' % (var_name, new_name))
            else:
                print('Renaming %s to %s.' % (var_name, new_name))
                # Rename the variable
                var = tf.Variable(var, name=new_name)

        if not dry_run:
            # Save the variables
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())
            saver.save(sess, checkpoint.model_checkpoint_path)    

        
    
    
    