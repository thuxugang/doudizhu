"""
Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.
Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0

original_vesion https://morvanzhou.github.io/tutorials/
modified by thuxugang
"""

import tensorflow as tf
import numpy as np

tf.set_random_seed(1)

###############################  Actor  ####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim, reward_decay=0.9, lr_a=0.01, lr_c=0.01, replace_target_iter=200, memory_size=2000, batch_size=32):
        self.memory = np.zeros((memory_size, s_dim * 2 + a_dim*3 + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.reward_decay, self.lr_a, self.lr_c = a_dim, s_dim, reward_decay, lr_a, lr_c
        self.replace_target_iter = replace_target_iter
        self.memory_size =memory_size
        self.batch_size = batch_size
        
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        
        self.ap = tf.placeholder(tf.float32, [None, self.a_dim], name='ap')  # input Action
        self.ap_ = tf.placeholder(tf.float32, [None, self.a_dim], name='ap_')  # input Action
        
        self.ah = tf.placeholder(tf.float32, [None, self.a_dim], name='ah')  # input Action
        
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, self.ap, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, self.ap_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        q_target = self.R + self.reward_decay * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(self.lr_c).minimize(self.td_error, var_list=self.ce_params)

        self.a_loss = -tf.reduce_mean(q)# + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ah,logits=self.a), name='A_error') # maximize the q

        self.atrain = tf.train.AdamOptimizer(self.lr_a).minimize(self.a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s, actions_one_hot, actions):
        a_all = self.sess.run(self.a, {self.S: s[np.newaxis, :], self.ap: actions_one_hot[np.newaxis, :]})
        action = np.argmax(a_all)
        if np.max(a_all) == 0.0:
            action_id = np.random.randint(0, len(actions))
            action = actions[action_id]   
        else:
            action_id = actions.index(action)
        return action, action_id

    def learn(self):
        # hard replace parameters
        if self.a_replace_counter % self.replace_target_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.at_params, self.ae_params)])
        if self.c_replace_counter % self.replace_target_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.ct_params, self.ce_params)])
        self.a_replace_counter += 1; self.c_replace_counter += 1

        # sample batch memory from all memory
        if self.pointer > self.memory_size:
            indices = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            indices = np.random.choice(self.pointer, size=self.batch_size)
            
        indices = np.random.choice(self.memory_size, size=self.batch_size)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        bap = bt[:, self.s_dim: self.s_dim + self.a_dim]
        bap_ = bt[:, self.s_dim + self.a_dim:self.s_dim + 2*self.a_dim]
        ba = bt[:, self.s_dim + 2*self.a_dim:self.s_dim + 3*self.a_dim]
        br = bt[:, self.s_dim + 3*self.a_dim]
        bs_ = bt[:, self.s_dim + 3*self.a_dim + 1:]
        
        _, a_loss = self.sess.run([self.atrain,self.a_loss], {self.S: bs, self.ap: bap, self.ah: ba})
        _, c_loss = self.sess.run([self.ctrain, self.td_error], {self.S: bs, self.R: br.reshape(self.batch_size,1), self.S_: bs_, self.a: ba, self.ap_: bap_})
        
        return [a_loss, c_loss]
        
    def store_transition(self, s, actions_one_hot, actions_one_hot_, a, r, s_):
        transition = np.hstack((s, actions_one_hot, actions_one_hot_, a, [r], s_))
        index = self.pointer % self.memory_size  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, action_possible, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 512, activation=tf.nn.relu, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 512, activation=tf.nn.relu, name='l2', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return a*action_possible

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            w1_s = tf.get_variable('w1_s', [self.s_dim, 512], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, 512], trainable=trainable)
            b1 = tf.get_variable('b1', [1, 512], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 512, activation=tf.nn.relu, name='l2', trainable=trainable)
            return tf.layers.dense(net, 1, activation=tf.nn.tanh, trainable=trainable)  # Q(s,a)

    #新增
    def save_model(self, name, episode):
        saver = tf.train.Saver() 
        saver.save(self.sess, "Model_sa/"+name+"_"+str(episode)+".ckpt") 

    #新增
    def load_model(self, name, episode):
        saver = tf.train.Saver() 
        saver.restore(self.sess, "Model_sa/"+name+"_"+str(episode)+".ckpt") 
        