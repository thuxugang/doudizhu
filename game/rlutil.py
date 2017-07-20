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
    state = np.zeros(33).astype("int")
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

        
    
    
    