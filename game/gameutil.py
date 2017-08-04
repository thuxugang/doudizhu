# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 21:55:58 2017

@author: XuGang
"""
from __future__ import print_function
import numpy as np
from .rlutil import get_state, get_actions, combine
import copy

#发牌
def game_init(players, playrecords, cards, train):
    
    if train:
        #洗牌
        np.random.shuffle(cards.cards)
        #排序
        p1_cards = cards.cards[:18]
        p1_cards.sort(key=lambda x: x.rank)
        p2_cards = cards.cards[18:36]
        p2_cards.sort(key=lambda x: x.rank)
        p3_cards = cards.cards[36:]
        p3_cards.sort(key=lambda x: x.rank)
        players[0].cards_left = playrecords.cards_left1 = p1_cards
        players[1].cards_left = playrecords.cards_left2 = p2_cards
        players[2].cards_left = playrecords.cards_left3 = p3_cards  
    else:
        #洗牌
        np.random.shuffle(cards.cards)
        #排序
        p1_cards = cards.cards[:20]
        p1_cards.sort(key=lambda x: x.rank)
        p2_cards = cards.cards[20:37]
        p2_cards.sort(key=lambda x: x.rank)
        p3_cards = cards.cards[37:]
        p3_cards.sort(key=lambda x: x.rank)
        players[0].cards_left = playrecords.cards_left1 = p1_cards
        players[1].cards_left = playrecords.cards_left2 = p2_cards
        players[2].cards_left = playrecords.cards_left3 = p3_cards    
    
    
    
#展示扑克函数
def card_show(cards, info, n):
    
    #扑克牌记录类展示
    if n == 1:
        print(info)
        names = []
        for i in cards:
            names.append(i.name+i.color)
        print(names)  
    #Moves展示
    elif n == 2:
        if len(cards) == 0:
            return 0
        print(info)
        moves = []
        for i in cards:
            names = []
            for j in i:
                names.append(j.name+j.color)
            moves.append(names)
        print(moves)    
    #record展示
    elif n == 3:
        print(info)
        names = []
        for i in cards:
            tmp = []
            tmp.append(i[0])
            tmp_name = []
            #处理要不起
            try:
                for j in i[1]:
                    tmp_name.append(j.name+j.color)
                tmp.append(tmp_name)
            except:
                tmp.append(i[1])
            names.append(tmp)
        print(names)
       

#在Player的next_moves中选择出牌方法
def choose(next_move_types, next_moves, last_move_type, last_move, cards_left, model, RL, agent, game, player_id, action):
    
    if model == "random":
        return choose_random(next_move_types, next_moves, last_move_type)
    elif model == "min":
        return choose_min(next_move_types, next_moves, last_move_type)
    elif model == "cxgz":
        return choose_cxgz(next_move_types, next_moves, last_move_type, last_move, cards_left, model)
    elif model in ["self","prioritized_dqn"]:
       return choose_xgmodel(next_move_types, next_moves, RL, agent, game, player_id)    
    #训练model
    elif model == "rl":
        if action[3][action[2]] == 429:
            return "buyao", []
        elif action[3][action[2]] == 430:
            return "yaobuqi", []
        else:
            return action[0][action[2]], action[1][action[2]] 
    #随机
    elif model == "combine":
        r = np.random.randint(0,6)
        if r == 0:
            return choose_random(next_move_types, next_moves, last_move_type)
        elif r == 1:
            return choose_min(next_move_types, next_moves, last_move_type)
        elif 2 <= r < 6:
            return choose_cxgz(next_move_types, next_moves, last_move_type, last_move, cards_left, model)
        #else:
        #    return choose_xgmodel(next_move_types, next_moves, RL, agent, game, player_id)

############################################
#                xgmodel                   #
############################################
def choose_xgmodel(next_move_types, next_moves, RL, agent, game, player_id):
    #要不起
    if len(next_moves) == 0:
        return "yaobuqi", []
    #state
    s = get_state(game.playrecords, player_id)
    #action
    actions = get_actions(next_moves, agent.actions_lookuptable, game)
    s = combine(s, actions)
    actions_ont_hot = np.zeros(agent.dim_actions)
    for k in range(len(actions)):
        actions_ont_hot[actions[k]] = 1
    action, action_id = RL.choose_action_model(s, actions_ont_hot, actions, e_greedy=1)
    #不要
    if actions[action_id] == 429:
        return "buyao", []
    return next_move_types[action_id], next_moves[action_id]   
   
############################################
#                  min                     #
############################################
def choose_min(next_move_types, next_moves, last_move_type):
    #要不起
    if len(next_moves) == 0:
        return "yaobuqi", []
    else:
        #start不能不要
        if last_move_type == "start":
            r_max = len(next_moves)
            r = np.random.randint(0,r_max)
        else:
            r_max = len(next_moves)+1
            r = 0
        #添加不要
        if r == len(next_moves):
            return "buyao", []
        
    return next_move_types[r], next_moves[r] 
        
############################################
#                 random                    #
############################################
def choose_random(next_move_types, next_moves, last_move_type):
    #要不起
    if len(next_moves) == 0:
        return "yaobuqi", []
    else:
        #start不能不要
        if last_move_type == "start":
            r_max = len(next_moves)
        else:
            r_max = len(next_moves)+1
        r = np.random.randint(0,r_max)
        #添加不要
        if r == len(next_moves):
            return "buyao", []
        
    return next_move_types[r], next_moves[r] 



############################################
#                 CX规则                   #
############################################
def choose_cxgz(next_move_types, next_moves, last_move_type, last_move, cards_left, model):
    '''
    开局start 的时候打牌策略
    '''
    if last_move_type=="start":
        return choose_start_policy(next_move_types, next_moves, last_move_type,cards_left,last_move)
    else:
        '''
        上家出牌时候的打牌策略
        '''
        return choose_orplay_policy(cards_left, last_move_type,last_move,next_move_types, next_moves)


'''
#对出牌类型进行排序
王炸>炸>
王炸 → 炸弹 → 单顺 → 双顺 → 三顺 → 三条 → 一对 → 单张
["dan", "dui", "san", "san_dai_yi", "san_dai_er", "shunzi"]
'''


def MoveTypeRank(next_move_types):
    type_rank = dict()
    for i in next_move_types:
        type_rank.get(i, 0)
        if i == 'dan':
            type_rank[i] = 0
        if i == "dui":
            type_rank[i] = 1
        if i == "san_dai_yi":
            type_rank[i] = 2
        if i == "san":
            type_rank[i] = 3
        if i == "san_dai_er":
            type_rank[i] = 4
        if i == "shunzi":
            type_rank[i] = 0.5
        if i == "bomb":
            type_rank[i] = 6

    return type_rank


###AI策略选择
def get_card_CombInfo(cards_left, last_move_type, last_move):
    # 出牌信息
    dan = []
    dui = []
    san = []
    san_dai_yi = []
    san_dai_er = []
    bomb = []
    shunzi = []
    liandui = []
    feiji = []
    feiji_dai_liangdan = []
    feiji_dai_liangdui = []

    # 牌数量信息 用字典统计牌的数量
    card_num_info = {}
    # 牌顺序信息,计算顺子
    card_order_info = []
    # 王牌信息
    king = []

    # 下次出牌
    next_moves = []
    # 下次出牌类型
    next_moves_type = []

    # 统计牌数量/顺序/王牌信息
    for i in cards_left:
        # 王牌信息
        if i.rank in [14, 15]:
            king.append(i)
        # 数量
        tmp = card_num_info.get(i.name, [])
        if len(tmp) == 0:
            card_num_info[i.name] = [i]  # 字典的key 对应的value是牌面为key的List
        else:
            card_num_info[i.name].append(i)
        # 顺序
        if i.rank in [13, 14, 15]:  # 不统计2,小王,大王
            continue
        elif len(card_order_info) == 0:
            card_order_info.append(i)
        elif i.rank != card_order_info[-1].rank:  # 跟上一张牌没出重复
            card_order_info.append(i)
    card_num_info_copy = copy.deepcopy(card_num_info)

    # 王炸
    if len(king) == 2:
        bomb.append(king)

    # 出单,出对,出三,炸弹(不考虑拆开)
    while card_num_info:
        for k, v in card_num_info.items():
            if len(v) == 4:
                bomb.append(v)
                card_num_info.pop(k)
                break
            elif len(v) == 3:
                if v[0].name in ["1","2"]:
                    san.append(v)
                    dui.append(v)
                    dan.append(v)
                    card_num_info.pop(k)
                else:
                    san.append(v)
                    card_num_info.pop(k)
                break
            elif len(v) == 2:
                if v[0].name in ["1","2"]:
                    dui.append(v)
                    dan.append(v)
                    card_num_info.pop(k)
                else:
                    dui.append(v)
                    card_num_info.pop(k)
                break
            elif len(v) == 1:
                dan.append(v)
                card_num_info.pop(k)
                break

    # 三带一,三带二
    # 三不带大牌
    for san_ in san:
        for dan_ in dan:
            # 防止重复
            if dan_[0].name != san_[0].name and dan_[0].name not in ["1", "2", "13", "14", "15"]:
                san_dai_yi.append(san_ + dan_)
        for dui_ in dui:
            # 防止重复
            if dui_[0].name != san_[0].name and dui_[0].name not in ["1", "2", "13", "14", "15"]:
                san_dai_er.append(san_ + dui_)
                # ============================================================
                # 考虑飞机
    san_order = []
    feiji_tmp = []
    # 对飞机进行排序
    san.sort(key=lambda x: x[0].rank)
    # card_show(san,"san_sorted: ",2)
    for idx in range(len(san)):
        if idx == 0 or len(san_order) == 0:
            #  san[idx] 是一个三张牌的list  [c1,c2,c3]
            san_order.append(san[idx])
        elif len(san_order) and san_order[-1][0].rank != san[idx][0].rank - 1:
            # 清空三的序列列表
            for j in range(len(san_order)):
                san_order.pop()
            # 从当前位置开始重新统计
            san_order.append(san[idx])
        elif san_order[-1][0].rank == san[idx][0].rank - 1:
            san_order.append(san[idx])
            if san[idx][0].rank > 11:  # 牌太大就不要连了
                san_order.pop()
            # 飞机两个长度就够了
            if len(san_order) >= 2:
                feiji_tmp.append(copy.deepcopy(san_order))

    if len(feiji_tmp):
        for kk in range(len(feiji_tmp)):
            dd = feiji_tmp[kk][0]
            for cc in range(1, len(feiji_tmp[kk])):
                dd += feiji_tmp[kk][cc]
            feiji.append(dd)
            # card_show(feiji[kk], "飞机%d" % kk, 1)
            # ============================================================
    # 考虑连对
    dui_order = []
    liandui_tmp = []
    # card_show(dui,"dui: ",2)
    # 对对子进行排序
    dui.sort(key=lambda x: x[0].rank)
    for idx in range(len(dui)):
        if idx == 0 or len(dui_order) == 0:
            dui_order.append(dui[idx])
        elif len(dui_order) and dui_order[-1][0].rank != dui[idx][0].rank - 1:
            for j in range(len(dui_order)):
                dui_order.pop()
            dui_order.append(dui[idx])
        elif dui_order[-1][0].rank == dui[idx][0].rank - 1:
            dui_order.append(dui[idx])
            if dui[idx][0].rank > 11:  # 牌太大就不要连了
                dui_order.pop()
            if len(dui_order) >= 3:
                liandui_tmp.append(copy.deepcopy(dui_order))
    if len(liandui_tmp):
        for kk in range(len(liandui_tmp)):
            dd = liandui_tmp[kk][0]
            for cc in range(1, len(liandui_tmp[kk])):
                dd += liandui_tmp[kk][cc]
            liandui.append(dd)
            # =========================================================================
    # 获取最长顺子
    max_len = []
    for i in card_order_info:
        if i == card_order_info[0]:  # max_len还是空的时候
            max_len.append(i)
        elif max_len[-1].rank == i.rank - 1:  # 出现了一张更大的并且连续的牌
            # 如果牌在炸弹、飞机、连对里面就不要拆顺子
            if_bomb = False
            if_feiji = False
            if_liandui = False
            for jj in feiji:
                if i.name == jj[0].name or i.name == jj[3].name:
                    if_feiji = True
            for cc in bomb:
                if i.name == cc[0].name:
                    if_bomb = True
            for ff in liandui:
                if i.name == ff[0].name or i.name == ff[2].name or i.name == ff[4].name:
                    if_liandui = True
            if if_liandui == False and if_bomb == False and if_feiji == False:
                max_len.append(i)
            else:
                continue
        else:
            if len(max_len) >= 5:  # 大于5 就直接成为顺子，这里所有牌都被拆了，要考虑所涉及到的对和顺的情况，是不是要拆
                shunzi.append(max_len)
            max_len = [i]
    # 最后一轮
    if len(max_len) >= 5:
        shunzi.append(max_len)
        # 拆顺子
    shunzi_sub = []
    shunzi_tmp = []
    for i in shunzi:
        len_total = len(i)
        n = len_total - 5
        # 遍历所有可能顺子长度
        while (n > 0):
            len_sub = len_total - n
            j = 0
            while (len_sub + j <= len(i)):
                # 遍历该长度所有组合
                shunzi_sub.append(i[j:len_sub + j])
                j = j + 1
            n = n - 1
    shunzi.extend(shunzi_sub)
    # 统计每种顺子所涉及到的重牌数量，对、等
    count_chongpai_lst = []
    cont_sanpai_lst = []
    for itlst in shunzi:
        # itlst是顺子的一种可能组合
        count_dui = 0
        count_san = 0
        for iit in itlst:
            it_name = iit.name
            it_lst = card_num_info_copy.get(it_name)
            it_len = len(it_lst)
            if it_len >= 2:
                count_dui += 1
            if it_len >= 3:
                count_san += 1
        count_chongpai_lst.append(count_dui)
        cont_sanpai_lst.append(count_san)

    shunzi_final = []
    # while kk<len(count_dui_lst):
    for kk in range(len(count_chongpai_lst)):
        if count_chongpai_lst[kk] <= len(shunzi[kk]) / 2 and cont_sanpai_lst[kk] <= 1:
            shunzi_final.append(shunzi[kk])

            # card_show(shunzi_final,"get_card_CombInfo::shunzi",2)
            # ======================================================================================
    if last_move_type == "start":
        # moves_types = ["dan", "dui", "san", "san_dai_yi", "san_dai_er", "shunzi","bomb","liandui","feiji"]

        moves_types = ["dan", "dui", "san", "san_dai_yi", "san_dai_er", "shunzi", "bomb"]
        i = 0
        # for move_zuhe in [dan, dui, san, san_dai_yi,san_dai_er, shunzi_final,bomb,liandui,feiji]:
        for move_zuhe in [dan, dui, san, san_dai_yi, san_dai_er, shunzi_final, bomb]:

            for mv_item in move_zuhe:
                next_moves.append(mv_item)
                next_moves_type.append(moves_types[i])
            i = i + 1
            # card_show(next_moves,"getCardInfo::next_moves",2)



    # 出单
    elif last_move_type == "dan":
        for move in dan:
            # 比last大
            if move[0].bigger_than(last_move[0]):
                next_moves.append(move)
                next_moves_type.append("dan")
    # 出对
    elif last_move_type == "dui":
        for move in dui:
            # 比last大
            if move[0].bigger_than(last_move[0]):
                next_moves.append(move)
                next_moves_type.append("dui")

    # 出三个
    elif last_move_type == "san":
        for move in san:
            # 比last大
            if move[0].bigger_than(last_move[0]):
                next_moves.append(move)
                next_moves_type.append("san")
    # 出三带一
    elif last_move_type == "san_dai_yi":
        for move in san_dai_yi:
            # 比last大
            if move[0].bigger_than(last_move[0]):
                next_moves.append(move)
                next_moves_type.append("san_dai_yi")
    # 出三带二
    elif last_move_type == "san_dai_er":
        for move in san_dai_er:
            # 比last大
            if move[0].bigger_than(last_move[0]):
                next_moves.append(move)
                next_moves_type.append("san_dai_er")
    # 出炸弹
    elif last_move_type == "bomb":
        for move in bomb:
            # 比last大
            if move[0].bigger_than(last_move[0]):
                next_moves.append(move)
                next_moves_type.append("bomb")
    # 出顺子
    elif last_move_type == "shunzi":
        # for move in shunzi:
        for move in shunzi_final:
            # 相同长度
            if len(move) == len(last_move):
                # 比last大
                if move[0].bigger_than(last_move[0]):
                    next_moves.append(move)
                    next_moves_type.append("shunzi")

    # 除了bomb,都可以出炸
    if last_move_type != "bomb":
        for move in bomb:
            next_moves.append(move)
            next_moves_type.append("bomb")
    # return next_moves_type,next_moves
    return next_moves_type, next_moves


def choose_start_chupai(cards_left, last_move_type, last_move):
    next_moves, next_moves_type = get_card_CombInfo(cards_left, last_move_type, last_move)
    return next_moves, next_moves_type


def choose_start_policy(next_move_types, next_moves, last_move_type, cards_left, last_move):
    # 要不起
    '''
        如果是start ，那么就出最小的move_type
        如果上家出牌的话，下家的牌的类型和上家是一样的

         poss_types,poss_moves是不拆牌情况下的下一步可能组合

    '''
    poss_types, poss_moves = get_card_CombInfo(cards_left, last_move_type, last_move)
    '''
    如果是后2-3把牌的开局的话，要先放大牌
    '''
    if len(cards_left) <= 5 and len(poss_moves) <= 3:
        follow_move_type = CheckNextPaiType(poss_types, "max")
        follow_move = GetMaxCardType(poss_types, follow_move_type, poss_moves)
    else:
        follow_move_type = CheckNextPaiType(poss_types, "max")
        follow_move = CheckNextPaiValue(poss_types, follow_move_type, poss_moves)
    # card_show(poss_moves,"choose_start_policy::poss_moves: ",2)
    # print("follow_move_type: ",follow_move_type)
    # print("follow_move: ",follow_move)
    # card_show(follow_move,"choose_start_policy::follow_move: ",1)
    return follow_move_type, follow_move


def choose_orplay_policy(cards_left, last_move_type, last_move, next_move_types, next_moves):
    '''
            如果上家出牌的话，下家的牌的类型和上家是一样的
            poss_types,poss_moves是不拆牌情况下的下一步可能组合
    '''
    poss_types, poss_moves = get_card_CombInfo(cards_left, last_move_type, last_move)

    '''
        如果上家出牌的话，下家的牌的类型和上家是一样的
        如果上家不要的话，自己要的起的话
    '''
    follow_move_type = last_move_type

    # 看牌剩下多少
    # 牌剩下的多的话采用出最小的方式
    # 牌只剩两把的话采用出最大的方式


    if len(cards_left) <= 2 and len(poss_moves) <= 2:
        follow_move = GetMaxCardType(poss_types, follow_move_type, poss_moves)
        if len(follow_move) == 0 or follow_move_type not in poss_types:
            return "yaobuqi", []
    else:
        follow_move = CheckNextPaiValue(poss_types, follow_move_type, poss_moves)
        if len(poss_moves) == 0:
            # 不拆牌情况下要不起的话，就考虑拆牌
            follow_move = CheckNextPaiValue(next_move_types, follow_move_type, next_moves)
            if len(follow_move) == 0 or follow_move_type not in poss_types:
                return "yaobuqi", []
                # return "yaobuqi", "yaobuqi"
        if poss_types[0] == "bomb" or poss_types == "bomb":
            if len(cards_left) <= 6 and len(poss_moves) <= 3:
                follow_move_type = "bomb"
                # 因为poss_move套了两个list
                follow_move = poss_moves[0]
            elif last_move[0].rank in [14,15]:
                follow_move_type = "bomb"
                follow_move = poss_moves[0]
            else:
                return "yaobuqi", []

    return follow_move_type, follow_move


def GetMaxCardType(next_move_types, follow_move_type, next_moves):
    # 找牌面最大的牌出
    # ==========================================================================
    next_move = None
    mlst = []
    # if follow_move_type in ["dan", "dui", "san", "san_dai_yi", "shunzi", "san_dai_er", "bomb","liandui","feiji"]:
    if follow_move_type in ["dan", "dui", "san", "san_dai_yi", "shunzi", "san_dai_er", "bomb"]:

        # 筛选出满足follow_move_type类型的牌的组合
        next_op_lst = []
        for j in range(len(next_moves)):
            if next_move_types[j] == follow_move_type:
                next_op_lst.append(next_moves[j])

                # 进行牌面的比较
        idx = 0
        for move in next_op_lst:
            # 比last大
            if idx == 0:
                mlst.append(move)
                # bigger_than比较了card rank，所一不用考虑1和2的情况
            if move[0].bigger_than(mlst[0][0]):
                mlst.pop()
                mlst.append(move)
            idx += 1
            # 返回可以出牌的组合里最小的move
        if len(mlst):
            next_move = mlst.pop()
        else:
            next_move = []
    return next_move


def CheckNextPaiType(poss_types_tmp, max_or_min):
    # 把符合出牌类型的牌放到一个列表里
    # 根据牌的rank,找出最小牌的位置，并把它设置为 next_move
    next_move_types_ranks = {}
    '''
    chupai_order_dict={"dan":1, "dui":2, "san":3, "san_dai_yi":4, "san_dai_er":5, "shunzi":7,"bomb":0,"feiji":6,
                       "liandui":8,"feiji_dai_erdan":9,"feiji_dai_erdui":10}
    '''
    chupai_order_dict = {"dan": 1, "dui": 2, "san": 3, "san_dai_yi": 4, "san_dai_er": 5, "shunzi": 7, "bomb": 0}
    for i in poss_types_tmp:
        val = chupai_order_dict.get(i)
        next_move_types_ranks[i] = val
    # 如果要出牌的话，要出next_move_types_ranks最小的一个
    min_idx = 100
    max_idx = 0

    # ========================返回最小出牌类型===========================
    follow_move_type = None

    for i, j in next_move_types_ranks.items():

        if j < min_idx:
            min_idx = j
        if j > max_idx:
            max_idx = j

    for i, j in chupai_order_dict.items():
        if max_or_min == "min":
            if j == min_idx:
                follow_move_type = i
        else:
            if j == max_idx:
                follow_move_type = i

    # 避免每次start都是出单

    if follow_move_type == "dan" and "dui" in poss_types_tmp:
        rdx = np.random.randint(0, 1)
        clst = ["dan", "dui"]
        follow_move_type = clst[rdx]

    return follow_move_type


def CheckNextPaiValue(next_move_types, follow_move_type, next_moves):
    # ==========================================================================
    # 如果是“对、三、炸”的
    next_move = None
    mlst = []
    # if follow_move_type in ["dan","dui","san","san_dai_yi","shunzi","san_dai_er","bomb","feiji","liandui"]:
    if follow_move_type in ["dan", "dui", "san", "san_dai_yi", "shunzi", "san_dai_er", "bomb"]:
        # 筛选出满足follow_move_type类型的牌的组合
        next_op_lst = []
        for j in range(len(next_moves)):
            if next_move_types[j] == follow_move_type:
                next_op_lst.append(next_moves[j])
        # card_show(next_op_lst,"CheckNextPaiValue:next_op_lst: ",2)
        # 进行牌面的比较
        idx = 0
        for move in next_op_lst:
            # 比last大
            if idx == 0:
                mlst.append(move)
            # bigger_than比较了card rank，所一不用考虑1和2的情况
            if follow_move_type=="san_dai_yi" or follow_move_type=="san_dai_er":
                score_mlst=mlst[0][0].rank*10+mlst[0][3].rank
                score_move=move[0].rank*10+move[3].rank
                if score_mlst>score_move:
                    mlst.pop()
                    mlst.append(move)
            else:
                if mlst[0][0].bigger_than(move[0]):
                    mlst.pop()
                    mlst.append(move)
            idx += 1
        # 返回可以出牌的组合里最小的move
        if len(mlst):
            next_move = mlst.pop()
        else:
            next_move = []
    return next_move
