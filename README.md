# 斗地主
deecamp斗地主

## master分支
提供了可以结合AI的程序引擎，在next_moves中提供按照规则的出牌所有可能性，需要自己实现从next_moves中选择所出的牌（myutil中的choose方法），默认random

## web分支
页面展示
### 使用方法
#### 1.启动server.py
#### 2.访问http://127.0.0.1:6666/ddz

## rl分支
模仿OpenAI，提供了可以结合RL的程序引擎，可以选择对手为random或规则或其他RL，但是训练时只能训练一个且为player 1。

rl模型包括DQN、double_dqn、prioritized_dqn和dueling_dqn，具体模型参考https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow

## multi-rl分支
模仿OpenAI，提供了可以结合RL的程序引擎，可以同时训练多个rl player

## contributor
thuxugang
