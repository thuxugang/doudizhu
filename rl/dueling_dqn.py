"""
The Dueling DQN based on this paper: https://arxiv.org/abs/1511.06581

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""

import numpy as np
import tensorflow as tf

tf.set_random_seed(1)


class DuelingDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            num_epochs,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            dueling=True,
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
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.dueling = dueling      # decide to use dueling DQN or not

        self.n_l1 = 512
        self.n_l2 = 512
        
        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*2+2))
        self._build_net()
        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
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
                
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [self.n_l1, self.n_l2], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_l2], initializer=b_initializer, collections=c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)

            if self.dueling:
                # Dueling DQN
                with tf.variable_scope('Value'):
                    w3 = tf.get_variable('w3', [self.n_l2, 1], initializer=w_initializer, collections=c_names)
                    b3 = tf.get_variable('b3', [1, 1], initializer=b_initializer, collections=c_names)
                    self.V = tf.nn.relu(tf.matmul(l2, w3) + b3)
                    
                with tf.variable_scope('Advantage'):
                    w3 = tf.get_variable('w3', [self.n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                    b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    self.A = tf.nn.relu(tf.matmul(l2, w3) + b3)
                with tf.variable_scope('Q'):
                    out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))     # Q = V(s) + A(s,a)
            else:
                with tf.variable_scope('Q'):
                    w3 = tf.get_variable('w3', [self.n_l2, self.n_actions], initializer=w_initializer, collections=c_names)
                    b3 = tf.get_variable('b3', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                    out = tf.nn.relu(tf.matmul(l2, w3) + b3)

            return out

        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            c_names, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, c_names, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    #修改
    def choose_action(self, observation, actions_ont_hot, actions):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value*actions_ont_hot)
            if np.max(actions_value*actions_ont_hot) == 0.0:
                action_id = np.random.randint(0, len(actions))
                action = actions[action_id]                
            else:
                action_id = actions.index(action)
        else:
            action_id = np.random.randint(0, len(actions))
            action = actions[action_id]
            
        return action, action_id

    def _replace_target_params(self):
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])

    def learn(self):
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replace_target_params()
            #print('\ntarget_params_replaced\n')

        sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval4next,  = self.sess.run(
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],    # next observation
                       self.s: batch_memory[:, -self.n_features:]})    # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

        return self.cost
    

    #新增
    def save_model(self, name, episode):
        saver = tf.train.Saver() 
        saver.save(self.sess, "Model_sa/"+name+"_"+str(episode)+".ckpt") 
    
    #新增
    def load_model(self, name, episode):
        saver = tf.train.Saver() 
        saver.restore(self.sess, "Model_sa/"+name+"_"+str(episode)+".ckpt") 
            
    #新增
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


