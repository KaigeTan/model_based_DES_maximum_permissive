# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 11:13:37 2023

@author: kaiget
"""

import numpy as np
from Cluster_tools_class import Cluster_tools
from MystepFun import StepFun
from norm_state_fun import norm_state
import random
from RL_brain_class import DeepQNetwork
import tensorflow as tf
import os
from datetime import datetime
import logging

# %% Load models
mat_file_name = 'Plant_compNEW.mat'  #New plant
plant_params = Cluster_tools('Plant_compNEW.mat')
# %% hyperparameters defination
random.seed(0)
NUM_ACTION = len(plant_params.E_c) + 1 #disable a controllable event to produce a control pattern at a state
NUM_OBS = 17  # 16 components and 1 counters
MEMORY_SIZE = np.exp2(13).astype(int)
BATCH_SIZE = np.exp2(11).astype(int)
NUM_EPISODE = 25000
TRAIN = 1
# initial state
INIT_OBS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0~15 : state number of Robor 1 ~ 5, INIT_OBS(16): number of input, INIT_OBS(16): delta of input-output
Num_out = 6  # First complete 6 outputs.
delta_max = 3

#%% build network
tf.reset_default_graph()
RL = DeepQNetwork(NUM_ACTION, 
                  NUM_OBS,
                  learning_rate = 5e-4,    
                  reward_decay = 0.98,
                  e_greedy = 0.95,
                  replace_target_iteration = 100,
                  memory_size = MEMORY_SIZE,
                  batch_size = BATCH_SIZE,
                  epsilon_increment = 1e-5,
                  epsilon_init = 0.05,
                  output_graph = False)
saver = tf.compat.v1.train.Saver(max_to_keep=None)
cwd = os.getcwd() + '\\' + datetime.today().strftime('%Y-%m-%d') + '\\Num_out_' + str(Num_out)
if not os.path.exists(cwd):
    os.makedirs(cwd)

total_step = 0
reward_history = []
good_event_history = []
episode_step_history = [0]
max_epi_reward = -50
logging.getLogger('tensorflow').disabled = True
# %% train
if TRAIN:
    for num_episode in range(NUM_EPISODE):
        # in each episode, reset initial state and total reward
        S = INIT_OBS
        S_norm = norm_state(S, delta_max)
        episode_reward = 0
        episode_step = 0
        epi_out_tool = 0
        epi_in_tool = 0
        epi_action_list = []
        # if num_episode > 35000 and num_episode < 75000:
        #     RL.learning_rate -= 2e-8
        if RL.learning_rate < 1e-4:
            RL.learning_rate = 1e-4
        else:
            if num_episode > 15000:
                RL.learning_rate -= 5e-8 # 5e-7
        while True:         
            # initialize the Action
            
            A = RL.choose_action(S_norm)
            # take action and observe
            [S_, all_S_, R, isDone, good_evt, stop_ind, selected_action] = \
                StepFun(S, A, plant_params, delta_max)
            S_norm_ = norm_state(S_, delta_max)
            all_S_norm_ = norm_state(all_S_, delta_max)
            
            # store transition
            RL.store_exp(S_norm, A, R, all_S_norm_)
            # control the learning starting time and frequency
            if total_step > MEMORY_SIZE and (total_step % 10 == 0):
                RL.learn()
            # update states
            episode_reward += R
            episode_step += 1
            epi_out_tool += good_evt
            S = S_
            # print(S_)
            S_norm = S_norm_
            if epi_out_tool == Num_out:
                isDone = 1
                stop_ind = 2
            if isDone: #episode_step > Num_out*32:
                if stop_ind == 1:
                    stop_reason = 'deadlock'
                elif stop_ind == 2:
                    stop_reason = 'Reach the object!'
                else:
                    stop_reason = 'reach maximal steps!'
                if max_epi_reward < episode_reward:
                    max_epi_reward = episode_reward
                epi_in_tool = epi_out_tool + S[-1]
                print('episode:', num_episode, '\n',
                      'episode reward:', round(episode_reward, 2), '\n',
                      'episode step:', episode_step, '\n',
                      'in tool:', epi_in_tool, '\n',
                      'out tool:', epi_out_tool, '\n',
                      # 'action list:', epi_action_list, '\n',
                      'maximal running step:', np.max(episode_step_history), '\n',
                      'maximal episode reward:', max_epi_reward, '\n',
                      'total good event:', np.sum(good_event_history), '\n',
                      'learning rate:', RL.learning_rate, '\n',
                      'epsilon:', RL.epsilon, '\n',
                      stop_reason, '\n',
                      '*******************************************'
                      )
                reward_history.append(episode_reward)
                good_event_history.append(epi_out_tool)
                episode_step_history.append(episode_step)
                
                # save checkpoint model, if a good model is received
                if episode_reward > 350:
                    save_path = cwd + '\\' + str(num_episode) + '_reward' + str(episode_reward) + 'step' + str(episode_step) + '.ckpt'
                    saver.save(RL.sess, save_path)
                break
                
                
            total_step += 1
            epi_action_list.append(selected_action)
            
    
    #%% draw cost curve
    RL.plot_cost()
    RL.plot_reward(reward_history, 250)
    RL.plot_epiEvent(good_event_history)
    # save_path_reward_mat = cwd + 'reward_his.mat'
    # sio.savemat(save_path_reward_mat, mdict={'reward': reward_history})

else:
    #Add the following codes to the file DQN_action_set.py to test the obtaoned model
    #%%
    #check_pt_path = cwd + '14999_reward195.67476760083986step194.ckpt'
    check_pt_path = 'C:\\Users\\kaiget\\OneDrive - KTH\\work\\MB_DQN_Junjun\\Cluster_tool_disable\\2023-06-09\\Num_out_6\\24044_reward386.558932262132step194.ckpt'
    #4 output
    #check_pt_path = '/home/sca-346/Desktop/JUnjunYang_case/model_based_DES_maximum_permissive-main//2023-04-24/Num_out_4/13077_reward159.40773214319398step128.ckpt'
    Num_step_test = 270 # 200 steps is set as the maximal number of steps 
    Num_target = 6
    Num_epi_test = 100
    OBS, STEP, BAD, ENB, Complete = RL.check_agent_path(check_pt_path, Num_step_test, Num_epi_test, Num_target, delta_max, plant_params) 
    # print(BAD, '\n') 
    # print(ENB, '\n') 
    # print(STEP, '\n') 
    # print(OBS, '\n') 
        