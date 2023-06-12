# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 10:48:48 2022

@author: KaigeT
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from MystepFun import StepFun
from norm_state_fun import norm_state
from ModularSupervisor import Permit
from NextObservation import Next
from random import choice

# %% deep Q network
class DeepQNetwork():
    def __init__(self,
                 n_actions,
                 n_features,
                 learning_rate = 0.005,
                 reward_decay = 0.95,
                 e_greedy = 0.9,
                 # iterations to update target network
                 replace_target_iteration = 300,
                 # memory size to store last experiences
                 memory_size = 1024,
                 # number of experiences to extract when update
                 batch_size = 128,
                 # initial epsilon value
                 epsilon_init = 0.1,
                 # decide whether change epsilon value with learning
                 epsilon_increment = False,
                 output_graph = False,
                 max_num_nextS = 10
                 ):
        # assign initial values
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iteration = replace_target_iteration
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.max_num_nextS = max_num_nextS
        
        self.epsilon_increment = epsilon_increment
        if epsilon_increment == False:
            self.epsilon = self.epsilon_max
        else:
            self.epsilon = epsilon_init
            
        # total learning step
        self.learn_step_counter = 0
        
        self.output_graph = output_graph
        
        # initialize training parameter
        self.learning_step_counter = 0
        # initialize memeory matrix
        # memory column: S, A, R, all_S_ -- S:(size: n_features), A & R: scalar
        # all_S_ -- all the next available states in the selected pattern
        # a memory: [S, A, R, all_S_]
        self.memory = np.zeros((self.memory_size, self.n_features*(max_num_nextS+1) + 2))
        
        # build network
        self._build_network()
        
        # extract network parameters
        target_params = tf.get_collection('target_net_params')
        eval_params = tf.get_collection('eval_net_params')
        
        # iterative replace parameter operation
        # pair [target_params, eval_params], and replace target(t) by eval(e) values
        self.replace_target_op = [
                tf.assign(t, e) for t, e in zip(target_params, eval_params)]
        
        # build sessions
        self.sess = tf.Session()
        
        # output the tensorboard file for network structure visualization
        if self.output_graph == True:
            # reset the graph which has already created
            tf.reset_default_graph()
            tf.summary.FileWriter("logs/",self.sess.graph)
        
        # initialize parameter in session
        self.sess.run(tf.global_variables_initializer())
        self.cost_history = []
       
        
    # build networks: evaluation network and q_target network
    def _build_network(self):
        
        ## define evaluation network, update each episode
        
        # collect observed state
        self.S = tf.placeholder(tf.float32, [None, self.n_features], name='S') 
        self.q_target = \
        tf.placeholder(tf.float32, [None, self.n_actions], name='q_target')
        
        # layer configuration
        n_l1 = 128
        n_l2 = 128
        
        with tf.variable_scope('eval_net'):
            # the name will be used when updating target_net params
            c_names = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            w_initializer = tf.random_normal_initializer(0, 1)
            b_initializer = tf.random_normal_initializer(0, 1)
            # w_initializer = tf.constant_initializer(0)
            # b_initializer = tf.constant_initializer(0)
            
            # the first layer of eval net, collections will be used when updating
            # target_net parameters
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', 
                                     [self.n_features, n_l1], 
                                     initializer = w_initializer,
                                     collections = c_names)
                b1 = tf.get_variable('b1', 
                                     [1, n_l1], 
                                     initializer = b_initializer,
                                     collections = c_names)
                l1 = tf.nn.relu(tf.matmul(self.S, w1) + b1)
                
            # the first layer of eval net, collections will be used when updating
            # target_net parameters
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', 
                                     [n_l1, n_l2], 
                                     initializer = w_initializer,
                                     collections = c_names)
                b2 = tf.get_variable('b2', 
                                     [1, n_l2], 
                                     initializer = b_initializer,
                                     collections = c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
        
            # the second layer of eval net, collections will be used when updating
            # target_net parameters
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', 
                                     [n_l2, self.n_actions], 
                                     initializer = w_initializer,
                                     collections = c_names)
                b3 = tf.get_variable('b3', 
                                     [1, self.n_actions], 
                                     initializer = b_initializer,
                                     collections = c_names)
                # output of the eval network
                self.q_eval = tf.matmul(l2, w3) + b3
        
        # define error calculation
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(
                    tf.squared_difference(self.q_target, self.q_eval))
        
        # train - gradient descent
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(
                    self.learning_rate).minimize(self.loss)
            # self._train_op = tf.train.AdamOptimizer().minimize(self.loss)
        
        
        ## define target network, update every n episodes
        ## the target network is identical to the eval network
        # collect observed state
        self.S_ = tf.placeholder(tf.float32, [None, self.n_features], name='S_') 
        
        with tf.variable_scope('target_net'):
            # the name will be used when updating target_net params
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            
            # the first layer of eval net, collections will be used when updating
            # target_net parameters
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', 
                                     [self.n_features, n_l1], 
                                     initializer = w_initializer,
                                     collections = c_names)
                b1 = tf.get_variable('b1', 
                                     [1, n_l1], 
                                     initializer = b_initializer,
                                     collections = c_names)
                l1 = tf.nn.relu(tf.matmul(self.S_, w1) + b1)
        
            # the second layer of target net, collections will be used when updating
            # target_net parameters
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', 
                                     [n_l1, n_l2], 
                                     initializer = w_initializer,
                                     collections = c_names)
                b2 = tf.get_variable('b2', 
                                     [1, n_l2], 
                                     initializer = b_initializer,
                                     collections = c_names)
                l2 = tf.nn.relu(tf.matmul(l1, w2) + b2)
                
            with tf.variable_scope('l3'):
                w3 = tf.get_variable('w3', 
                                     [n_l2, self.n_actions], 
                                     initializer = w_initializer,
                                     collections = c_names)
                b3 = tf.get_variable('b3', 
                                     [1, self.n_actions], 
                                     initializer = b_initializer,
                                     collections = c_names)
                # output of the target network
                self.q_next = tf.matmul(l2, w3) + b3
    
    
    # A = RL.choose_action(S) return the number index of the action, S -- type list
    def choose_action(self, S):
        # add a deminsion for tensorflow parse
        # o.w. Cannot feed value of shape (3,) for Tensor 'S:0', which has shape '(?, 3)'
   
        S = np.array(S)
        S = S[np.newaxis, :] # if error, try add np.array(S)
        
        if(np.random.uniform() > self.epsilon):
            action_index = np.random.randint(0, self.n_actions)
        else:
            # evaluate the current q_values of all actions based on state S
            action_value = self.sess.run(self.q_eval, feed_dict = {self.S: S})
            action_index = np.argmax(action_value)
        return action_index
    
    
    # calculate the estimated value of the next state-action pair E[Q(s', a')] = max(Q(s', a'))
    def state_next_value(self, S):
        # add a deminsion for tensorflow parse
        # o.w. Cannot feed value of shape (7,) for Tensor 'S:0', which has shape '(?, 7)'
        S = np.array(S)
        S = S[np.newaxis, :] # if error, try add np.array(S)
        action_value_vec = self.sess.run(self.q_next, feed_dict = {self.S_: S})
        S_A_next_val = max(action_value_vec)
        return S_A_next_val
    
    
    # store experience replay
    def store_exp(self, S, A, R, all_S_):
        # at first call, add an attribute to count memory list
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        # all_S_: N X 7 list (N assigned with the number of 5, maximal of 5 different S_)
        all_S_list = [-1]*(self.max_num_nextS*self.n_features)   # initialize all_S_list with all -1
        all_S_1D = sum(all_S_, []) # flatten to 1D, 1 X (7*num of all S_)
        if len(all_S_1D) < self.max_num_nextS*self.n_features:
            all_S_list[: len(all_S_1D)] = all_S_1D
        else:
            print('exceed max number of available events in a pattern!')
        
        # stack SARS_ in: [S, A, R, S_], column number identical
        new_memory = np.hstack((S,[A, R], all_S_list))      # size: 5 + 2 + 5X5
        memory_index = self.memory_counter % self.memory_size
        self.memory[memory_index, :] = new_memory
        self.memory_counter += 1
  
    

    
    # RL.learn(S, A, R, S_)
    def learn(self):
        # reset q target every replace_target_iteration interations
        if self.learn_step_counter % self.replace_target_iteration == 0:
            self.sess.run(self.replace_target_op)
            print('replace target network parameters at:',\
                  self.memory_counter, 'step\n')
        
        # sample random minibatch of transitions from memory
        if self.memory_counter > self.batch_size:
            sample_index = np.random.choice(self.memory_size, self.batch_size)
        else:
            # if the number of experiences is less than batch_size,
            # some of the experiences will be picked repetitively
            sample_index = np.random.choice(self.memory_counter, self.batch_size)
        batch_memory = self.memory[sample_index ,:]
        
        # get Q-evaluation value of the current state (Q(s)) from eval network
        train_q_eval = self.sess.run(self.q_eval, feed_dict = 
                                     {self.S: batch_memory[:, :self.n_features]})
        # get Q-target value of the next state (Q(s')) from target network
        #################### note: here q_next contains all possible S_ in a pattern ############################
        avail_S_num_ = []
        for i_sample in range(self.batch_size):
            avail_S_num_.append((batch_memory[i_sample].tolist().index(-1) - 2 - self.n_features)/self.n_features)  # calculate the number of avaiable next states in each sample
        train_q_target_ = np.zeros(self.batch_size, dtype= np.float32)
        for idx_act in range(int(max(avail_S_num_))):
            sample_idx = np.where(np.array(avail_S_num_) > idx_act)[0]
            vector_S_ = batch_memory[sample_idx, (idx_act+1)*self.n_features+2: (idx_act+2)*self.n_features+2]
            # this is the evaluation value of one S_ for an action in pattern, max(Q(S_, a_))
            vector_train_q_target_temp = np.max(self.sess.run(self.q_next, feed_dict = 
                                         {self.S_: vector_S_}), axis=1)
            train_q_target_[sample_idx] += vector_train_q_target_temp
        train_q_target_ = np.divide(train_q_target_, np.array(avail_S_num_))

        # train_q_target_ = self.sess.run(self.q_next, feed_dict = 
        #                              {self.S_: batch_memory[:, -self.n_features:]})
        
        
        q_target = train_q_eval.copy() # use .copy() here to avoid train_q_eval also change
        # generate a index list
        batch_index = np.arange(self.batch_size, dtype = np.int32)
        # get the action choice of each experience in memory
        eval_action_index = batch_memory[:, self.n_features].astype(int)
        # get reward value of each experience in memory
        eval_reward = batch_memory[:, self.n_features + 1]
        # replace the evaluation value to the target value R + gamma*max(Q(S'))
        q_target[batch_index, eval_action_index] = \
        eval_reward[batch_index] + self.gamma * train_q_target_[batch_index] # already perform np.max before when calculating all Q(S_, A_) value
        # q_target[batch_index, eval_action_index] = \
        # eval_reward[batch_index] + self.gamma * np.max(train_q_target_[batch_index, :], axis = 1)
        # 10 as maximal value for event number
        
        # calculate the cost based on batch samples
        _,self.cost = self.sess.run([self._train_op, self.loss],
                         feed_dict = {self.S: batch_memory[:, :self.n_features],
                                      self.q_target: q_target})
        
        self.cost_history.append(self.cost)
        
        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
    
        
    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()
        
    def plot_reward(self, reward_his, moving_avg_len):
        avg_list = np.convolve(reward_his, np.ones(moving_avg_len), 'valid')/moving_avg_len
        plt.plot(np.arange(len(avg_list)), avg_list)
        plt.ylabel('Average reward')
        plt.xlabel('Episode number')
        plt.show()
    
    def plot_epiEvent(self, good_event_history):
        plt.plot(np.arange(len(good_event_history)), good_event_history)
        plt.ylabel('Number of good events')
        plt.xlabel('Episode number')
        plt.show()
        
        
    
    #Add this definition to the file RL_brain_class.py
    
    def check_agent_path(self, check_pt_path, Num_step_test, Num_epi_test, Num_target, max_diff_num, plant_params):
        # record the test result
        BAD = []
        ENB = []
        STEP = []
        OBS =[]
        Complete = []
        # load the trained network
        meta_path = check_pt_path + '.meta'
        saver_test = tf.train.import_meta_graph(meta_path)
        saver_test.restore(self.sess, check_pt_path)
        for i_epi in range(Num_epi_test): # 100 rounds of simulation
            obs = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            step = 0
            Bad = 0
            tool_finish = 0
            while step <= Num_step_test:  #The number of steps for testing
                S_test = obs
                S_test = np.array(S_test)
                S_test = S_test[np.newaxis, :] # if error, try add np.array(S)
                action_value = self.sess.run(self.q_eval, feed_dict = {self.S: S_test})            
                action_index = np.argmax(action_value)  # The optimal action_index
                Enb = Permit(obs, plant_params)
                pattern = Enb                   
                
                
                if obs[-1] > max_diff_num: # if out-in number of products exceed max_diff_num, then disable sending an product
                    pattern = np.setdiff1d(pattern, 10)  # Restrict number of wafers input
                
                if not (len(pattern) == 1 or action_index == 16): # 16 means disable nothing
                    pattern = np.setdiff1d(pattern, plant_params.E_c[action_index]) # disable a controllable event
                
                
                # if not(action_index == 16 or len(pattern) == 1):
                #     pattern = np.setdiff1d(pattern, plant_params.E_c[action_index])            
                if len(pattern) != 0:                              
                    for event in pattern:
                        obs_ = Next(obs, event, plant_params)
                        Enb_ = Permit(obs_, plant_params)
                        if len(Enb_) == 0:
                            print('An event leads to a deadlock')
                            Bad = 1
                            break
                    if Bad == 1:
                        break
                    step += 1
                    event_exe = choice(pattern) 
                    if event_exe == 15:
                        tool_finish += 1 # No.15 action indicates a new tool manufactored
                    obs_next = Next(obs, event_exe, plant_params)
                    obs = obs_next 
                    if tool_finish == Num_target: # jump out the loop if the number of target tool is finished
                        break
                else:
                    print('The selected pattern is empty')
                    obs = obs
                    Bad = 2
                    break
            print('test number: ', i_epi)
            BAD.append(Bad)  # number of basd results
            ENB.append(pattern)  # the enabled event set of deadlock states or the final state of the checked model
            STEP.append(step)   # the maximal steps of the checked model
            OBS.append(obs)    # the terminate states of the checked model
            Complete.append(tool_finish)
        return OBS, STEP, BAD, ENB, Complete

    
