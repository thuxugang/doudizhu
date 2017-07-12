# -*- coding: utf-8 -*-
import numpy as np


class Cards(object):
    """
    一副扑克牌类,54张排,abcd四种花色,小王14-a,大王15-a
    """
    def __init__(self,):
        #初始化扑克牌名称
        self.cards_name = ['1-a', '1-b','1-c','1-d',
                           '2-a', '2-b','2-c','2-d',
                           '3-a', '3-b','3-c','3-d',
                           '4-a', '4-b','4-c','4-d',
                           '5-a', '5-b','5-c','5-d',
                           '6-a', '6-b','6-c','6-d',
                           '7-a', '7-b','7-c','7-d',
                           '8-a', '8-b','8-c','8-d',
                           '9-a', '9-b','9-c','9-d',
                           '10-a', '10-b','10-c','10-d',
                           '11-a', '11-b','11-c','11-d',
                           '12-a', '12-b','12-c','12-d',
                           '13-a', '13-b','13-c','13-d',
                           '14-a', '15-a']
        #初始化扑克牌类                  
        self.cards = self.get_cards()

        #定义扑克牌大小
        self.cards_orders = self.get_cards_orders()
        
    #初始化扑克牌类
    def get_cards(self):
        cards = []
        for name in self.cards_name:
            cards.append(Card(name))
        #打乱顺序
        np.random.shuffle(cards)
        return cards
    
    #定义扑克牌大小
    def get_cards_orders(self):
        cards_orders = {}
        j = 0
        #3-13
        for i in range(3,14):
            cards_orders[i] = j
            j = j + 1
        #1-2
        for i in range(1,3):
            cards_orders[i] = j
            j = j + 1     
        #14-15
        for i in range(14,16):
            cards_orders[i] = j
            j = j + 1    
        return cards_orders
                           
class Card(object):
    """
    扑克牌类
    """
    def __init__(self, name):
        self.name = name
        #大小
        self.rank = int(self.name.split('-')[0])
        #花色
        self.color = self.name.split('-')[1]
    
    #判断大小,参数为另一个扑克牌类Card实例和一副扑克牌类Cards实例
    def bigger_than(self, card_instance, cards_instance):
        if (cards_instance.cards_orders[self.rank] > 
            cards_instance.cards_orders[card_instance.rank]):
            return True
        else:
            return False
        

class PlayRecords(object):
    """
    扑克牌记录类
    """
    def __init__(self):
        self.cards_left1 = []
        self.cards_left2 = []
        self.cards_left3 = []
        
        #出牌记录
        self.records = []
    
    #展示
    def show(self):
        print "player 1"
        names = []
        for i in self.cards_left1:
            names.append(i.name)
        print names
        print "player 2"
        names = []
        for i in self.cards_left2:
            names.append(i.name)
        print names    
        print "player 3"
        names = []
        for i in self.cards_left3:
            names.append(i.name)
        print names 
        print "records"
        print self.records
        
class Player(object):
    """
    player类
    """
    def __init__(self, player_id):
        self.player_id = player_id
        self.cards_left = []
        #是否该出牌
        self.turn_on = False
        #下次出牌可选列表
        self.next_move = []
    
    #出牌
    def go():

#发牌
def game_start(players, playrecords, cards):
    
    players[0].cards_left = playrecords.cards_left1 = cards.cards[:18]
    players[1].cards_left = playrecords.cards_left2 = cards.cards[18:36]
    players[2].cards_left = playrecords.cards_left3 = cards.cards[36:]
    
    #player0先出
    players[0].turn_on = True
     
if __name__=="__main__":
    
    #初始化一副扑克牌类
    cards = Cards()
    
    #初始化players
    players = []
    for i in range(3):
        players.append(Player(i))
    
    #初始化扑克牌记录类
    playrecords = PlayRecords()
    
    #发牌
    game_start(players, playrecords, cards)
    playrecords.show()
    
    
    
    
    
    
    
    
    
    
    
    