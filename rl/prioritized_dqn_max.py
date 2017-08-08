"""
The DQN improvement: Prioritized Experience Replay (based on https://arxiv.org/abs/1511.05952)

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0

original_vesion https://morvanzhou.github.io/tutorials/
modified by thuxugang
"""
from __future__ import absolute_import
import numpy as np
import tensorflow as tf

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


class DQNPrioritizedReplay:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=500,
            replace_target_iter_model=5000,
            memory_size=5000,
            batch_size=32,
            e_greedy_increment=None,
            epsilon_init=0.5,
            output_graph=False,
            prioritized=True,
            sess=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon_init = epsilon_init
        self.epsilon = epsilon_init if e_greedy_increment is not None else self.epsilon_max

        self.prioritized = prioritized    # decide to use double q or not

        self.replace_target_iter_model = replace_target_iter_model
        
        self.learn_step_counter = 0

        self.n_l1 = 512
        self.n_l2 = 512

        self._build_net()
        
        if self.prioritized:
            self.memory = Memory(capacity=memory_size)
        else:
            self.memory = np.zeros((self.memory_size, n_features * 2 + 2 + n_actions))

        #显存占用20%
        #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)  
        
        if sess is None:
            #self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) 
            
            #accelerate MLP
            config = tf.ConfigProto()
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())
            print("use pdqn!")
        else:
            self.sess = sess

        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.cost_his = []

    #修改
    def _build_net(self):
        def build_layers(s, c_names, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, self.n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, self.n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)
                #l1 = tf.maximum(0.2 * l1, l1)
                
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [self.n_l1, self.n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
                #l2 = tf.maximum(0.2 * l2, l2)
                
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', [self.n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l2, w3) + b3
            return out, b3
        
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        if self.prioritized:
            self.ISWeights = tf.placeholder(tf.float32, [None, 1], name='IS_weights')
        with tf.variable_scope('eval_net'):
            c_names, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval, self.be = build_layers(self.s, c_names, w_initializer, b_initializer)

        with tf.variable_scope('eval_net_model'):
            c_names, w_initializer, b_initializer = \
                ['eval_net_params_model', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval_model, self.bem = build_layers(self.s, c_names, w_initializer, b_initializer)
            
        with tf.variable_scope('loss'):
            if self.prioritized:
                self.abs_errors = tf.reduce_sum(tf.abs(self.q_target - self.q_eval), axis=1)    # for updating Sumtree
                self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.q_target, self.q_eval))
            else:
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next, _  = build_layers(self.s_, c_names, w_initializer, b_initializer)

    def store_transition(self, s, actions_one_hot, a, r, s_):
        if self.prioritized:    # prioritized replay
            transition = np.hstack((s, actions_one_hot, [a, r], s_))
            self.memory.store(transition)    # have high priority for newly arrived transition
        else:       # random replay
            if not hasattr(self, 'memory_counter'):
                self.memory_counter = 0
            transition = np.hstack((s, actions_one_hot, [a, r], s_))
            index = self.memory_counter % self.memory_size
            self.memory[index, :] = transition
            self.memory_counter += 1

    #修改
    def choose_action(self, observation, actions_ont_hot, actions):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        a, b, c = 0, 0, 0
        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value, b3 = self.sess.run([self.q_eval,self.be], feed_dict={self.s: observation})
            #print("dz", b3)
            #print(actions_value)
            #print(actions_value*actions_ont_hot)
            action_value_pos = actions_value*actions_ont_hot
            action_value_pos[action_value_pos==0.0]=-np.inf
            action = np.argmax(action_value_pos)
            action_id = actions.index(action)
            c = action
            a = actions_value
            b = action_value_pos
        else:
            action_id = np.random.randint(0, len(actions))
            action = actions[action_id]
        return action, action_id, a, b, c

    #修改
    def choose_action_model(self, observation, actions_ont_hot, actions, e_greedy):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]
        if np.random.uniform() < e_greedy:
            # forward feed the observation and get q value for every actions
            actions_value, b3 = self.sess.run([self.q_eval_model, self.bem],feed_dict={self.s: observation})
            #print("nm",b3)
            action_value_pos = actions_value*actions_ont_hot
            action_value_pos[action_value_pos==0.0]=-np.inf
            action = np.argmax(action_value_pos)
            action_id = actions.index(action)
        else:
            action_id = np.random.randint(0, len(actions))
            action = actions[action_id]
        return action, action_id
    
    def _replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

    def _replace_target_params_model(self):
        m_params = tf.get_collection('eval_net_params_model')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(m, e) for m, e in zip(m_params, e_params)])
        self.epsilon = self.epsilon_init
        print("step: ", self.learn_step_counter, " update eval_net_model_params_model")
    
    def check_params(self):
        em = self.sess.run(tf.get_collection('eval_net_params_model'))
        e = self.sess.run(tf.get_collection('eval_net_params'))
        t = self.sess.run(tf.get_collection('target_net_params'))
        return tf.get_collection('eval_net_params_model'), em[1],tf.get_collection('eval_net_params'), e[1],tf.get_collection('target_net_params'), t[1]
    #修改
    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            #print('\ntarget_params_replaced\n')
        if self.learn_step_counter % self.replace_target_iter_model == 0 and self.learn_step_counter != 0:
            self._replace_target_params_model()
            
        if self.prioritized:
            tree_idx, batch_memory, ISWeights = self.memory.sample(self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
            batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.sess.run(
                [self.q_next, self.q_eval],
                feed_dict={self.s_: batch_memory[:, self.n_features+self.n_actions+2:],
                           self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features+self.n_actions].astype(int)
        reward = batch_memory[:, self.n_features+self.n_actions+1]

        action_possible = batch_memory[:, self.n_features:self.n_features+self.n_actions]
        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next*action_possible, axis=1)

        if self.prioritized:
            _, abs_errors, self.cost = self.sess.run([self._train_op, self.abs_errors, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target,
                                                    self.ISWeights: ISWeights})
            for i in range(len(tree_idx)):  # update priority
                idx = tree_idx[i]
                self.memory.update(idx, abs_errors[i])
        else:
            _, self.cost = self.sess.run([self._train_op, self.loss],
                                         feed_dict={self.s: batch_memory[:, :self.n_features],
                                                    self.q_target: q_target})

        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        return self.cost, self.learn_step_counter

    #新增
    def save_model(self, model):
        saver = tf.train.Saver() 
        saver.save(self.sess, model) 

    #新增
    def load_model(self, model):
        saver = tf.train.Saver() 
        saver.restore(self.sess, model) 
        #读取最新模型
        self._replace_target_params_model()
    #新增   
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

        