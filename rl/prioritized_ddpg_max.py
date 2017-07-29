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

class SumTree(object):
    """
    This SumTree code is modified version and the original code is from: 
    https://github.com/jaara/AI-blog/blob/master/SumTree.py
    
    Story the data with it priority in tree and data frameworks.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity    # for all priority values
        self.tree = np.zeros(2*capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        self.data = np.zeros(capacity, dtype=object)    # for all transitions
        # [--------------data frame-------------]
        #             size: capacity

    def add_new_priority(self, p, data):
        leaf_idx = self.data_pointer + self.capacity - 1

        self.data[self.data_pointer] = data # update data_frame
        self.update(leaf_idx, p)    # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]

        self.tree[tree_idx] = p
        self._propagate_change(tree_idx, change)

    def _propagate_change(self, tree_idx, change):
        """change the sum of priority value in all parent nodes"""
        parent_idx = (tree_idx - 1) // 2
        self.tree[parent_idx] += change
        if parent_idx != 0:
            self._propagate_change(parent_idx, change)

    def get_leaf(self, lower_bound):
        leaf_idx = self._retrieve(lower_bound)  # search the max leaf priority based on the lower_bound
        data_idx = leaf_idx - self.capacity + 1
        return [leaf_idx, self.tree[leaf_idx], self.data[data_idx]]

    def _retrieve(self, lower_bound, parent_idx=0):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        left_child_idx = 2 * parent_idx + 1
        right_child_idx = left_child_idx + 1

        if left_child_idx >= len(self.tree):    # end search when no more child
            return parent_idx

        if self.tree[left_child_idx] == self.tree[right_child_idx]:
            return self._retrieve(lower_bound, np.random.choice([left_child_idx, right_child_idx]))
        if lower_bound <= self.tree[left_child_idx]:  # downward search, always search for a higher priority node
            return self._retrieve(lower_bound, left_child_idx)
        else:
            return self._retrieve(lower_bound-self.tree[left_child_idx], right_child_idx)

    @property
    def root_priority(self):
        return self.tree[0]     # the root


class Memory(object):   # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6     # [0~1] convert the importance of TD error to priority
    beta = 0.4      # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def store(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add_new_priority(max_p, transition)   # set the max p for new p

    def sample(self, n):
        batch_idx, batch_memory, ISWeights = [], [], []
        segment = self.tree.root_priority / n
        self.beta = np.min([1, self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.root_priority
        maxiwi = np.power(self.tree.capacity * min_prob, -self.beta)  # for later normalizing ISWeights
        for i in range(n):
            a = segment * i
            b = segment * (i + 1)
            lower_bound = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(lower_bound)
            prob = p / self.tree.root_priority
            ISWeights.append(self.tree.capacity * prob)
            batch_idx.append(idx)
            batch_memory.append(data)

        ISWeights = np.vstack(ISWeights)
        ISWeights = np.power(ISWeights, -self.beta) / maxiwi  # normalize
        return batch_idx, np.vstack(batch_memory), ISWeights

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def _get_priority(self, error):
        error += self.epsilon  # avoid 0
        clipped_error = np.clip(error, 0, self.abs_err_upper)
        return np.power(clipped_error, self.alpha)
    
###############################  Actor  ####################################


class DDPGPrioritized(object):
    def __init__(self, a_dim, s_dim, reward_decay=0.9, lr_a=0.01, lr_c=0.01,
                 replace_target_iter=200, memory_size=2000, batch_size=32,replace_target_iter_model=100000,
                 prioritized=True,):
        
        self.prioritized = prioritized
        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((memory_size, s_dim * 2 + a_dim*3 + 1), dtype=np.float32)
        
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
            
        self.pointer = 0
        self.sess = tf.Session()
        self.a_replace_counter, self.c_replace_counter = 0, 0

        self.a_dim, self.s_dim, self.reward_decay, self.lr_a, self.lr_c = a_dim, s_dim, reward_decay, lr_a, lr_c
        self.replace_target_iter = replace_target_iter
        self.memory_size =memory_size
        self.batch_size = batch_size
        
        self.replace_target_iter_model = replace_target_iter_model
        
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')
        
        self.ap = tf.placeholder(tf.float32, [None, self.a_dim], name='ap')  # input Action
        self.ap_ = tf.placeholder(tf.float32, [None, self.a_dim], name='ap_')  # input Action
        
        self.ah = tf.placeholder(tf.float32, [None, self.a_dim], name='ah')  # input Action
        
        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, self.ap, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, self.ap_, scope='target', trainable=False)
            
            #load model still
            self.a_model = self._build_a(self.S, self.ap, scope='model', trainable=False)
       
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
        
        self.am_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/model')

        q_target = self.R + self.reward_decay * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        if self.prioritized:
            self.abs_td_error = tf.reduce_sum(tf.abs(q_target - q), axis=1)   
            self.td_error = tf.reduce_mean(self.ISWeights * tf.squared_difference(q_target, q))
        else:
            self.td_error = tf.reduce_mean(tf.squared_difference(q_target, q))
        self.ctrain = tf.train.AdamOptimizer(self.lr_c).minimize(self.td_error, var_list=self.ce_params)

        self.a_loss = -tf.reduce_mean(q) + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.ah,logits=self.a), name='A_error') # maximize the q

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

    #load model still
    def choose_action_model(self, s, actions_one_hot, actions):
        a_all = self.sess.run(self.a_model, {self.S: s[np.newaxis, :], self.ap: actions_one_hot[np.newaxis, :]})
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
        
        if self.a_replace_counter % self.replace_target_iter_model == 0:
            self.sess.run([tf.assign(m, e) for m, e in zip(self.am_params, self.ae_params)])
        self.a_replace_counter += 1; self.c_replace_counter += 1

        if self.prioritized:
            tree_idx, bt, ISWeights = self.memory.sample(self.batch_size)
        else:       
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

        if self.prioritized:
            _, a_loss = self.sess.run([self.atrain,self.a_loss], {self.S: bs, self.ap: bap, self.ah: ba})
            _, abs_td_error, c_loss = self.sess.run([self.ctrain, self.abs_td_error, self.td_error], 
                                                    {self.S: bs, self.R: br.reshape(self.batch_size,1), 
                                                     self.S_: bs_, self.a: ba, self.ap_: bap_,
                                                     self.ISWeights: ISWeights})
            for i in range(len(tree_idx)):  # update priority
                idx = tree_idx[i]
                self.memory.update(idx, abs_td_error[i])
        else:        
            _, a_loss = self.sess.run([self.atrain,self.a_loss], {self.S: bs, self.ap: bap, self.ah: ba})
            _, c_loss = self.sess.run([self.ctrain, self.td_error], {self.S: bs, self.R: br.reshape(self.batch_size,1), self.S_: bs_, self.a: ba, self.ap_: bap_})
            
        return [a_loss, c_loss]
        
    def store_transition(self, s, actions_one_hot, actions_one_hot_, a, r, s_):
        if self.prioritized:    # prioritized replay
            transition = np.hstack((s, actions_one_hot, actions_one_hot_, a, [r], s_))
            self.memory.store(transition)    # have high priority for newly arrived transition
        else: 
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
        #读取最新模型
        self.sess.run([tf.assign(m, e) for m, e in zip(self.am_params, self.ae_params)])
        