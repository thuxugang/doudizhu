# 斗地主
deecamp斗地主

## master分支
提供了可以结合AI的程序引擎，在next_moves中提供按照规则的出牌所有可能性，需要自己实现从next_moves中选择所出的牌（myutil中的choose方法），默认random

## web分支
1.页面展示，提供可视化调试方法

2.可以选择跟人对战
### 使用方法
#### 1.启动server.py
#### 2.访问http://127.0.0.1:5000/ddz

## rl_pdqn分支
模仿OpenAI，提供了可以结合RL的程序引擎，可以选择对手为random或陈潇规则(cxgz)或自身(self)，但是训练时只能训练一个且为player 1。该分支rl模型为prioritized_dqn，具体模型参考https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow 。

目前胜率 vs random（90%）， cxgz（44%）

## multi-rl分支
模仿OpenAI，提供了可以结合RL的程序引擎，可以同时训练多个rl player

## mcts分支
mcts暴力解决（TODO：由于deepcopy牌局的回复速度比较慢，1000/6s）

## contributor
Deecamp第五组
