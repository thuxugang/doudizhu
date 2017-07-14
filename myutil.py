# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
import numpy as np

 
#展示扑克函数
def card_show(cards, info, n):
    
    #扑克牌记录类展示
    if n == 1:
        print info
        names = []
        for i in cards:
            names.append(i.name+i.color)
        print names    
    #Moves展示
    elif n == 2:
        if len(cards) == 0:
            return 0
        print info
        moves = []
        for i in cards:
            names = []
            for j in i:
                names.append(j.name+j.color)
            moves.append(names)
        print moves    

#在Player的next_moves中选择出牌方法,目前random
def choose(next_moves):
    return next_moves[np.random.randint(0,len(next_moves))]  
    
    
    
    
    
    
    
    
    