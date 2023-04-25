import scipy.io as sio
import numpy as np
from MystepFun import StepFun
from norm_state_fun import norm_state
import random
from RL_brain_class import DeepQNetwork
import tensorflow as tf
import os
from datetime import datetime

# %% Load models
load_data = sio.loadmat('Plant_Comp.mat')  #Get the model file from MATLAB 

Rob1 = load_data['Rob1'] 
Rob2 = load_data['Rob2'] 
Rob3 = load_data['Rob3'] 
Rob4 = load_data['Rob4'] 

Sc11 = load_data['Sc11'] 
Sc12 = load_data['Sc12'] 
Sc21 = load_data['Sc21']
Sc22 = load_data['Sc22']
Sc31 = load_data['Sc31']
Sc32 = load_data['Sc32']
Sc41 = load_data['Sc41']     
Sc42 = load_data['Sc42'] 
Sc43 = load_data['Sc43']

Sb1 = load_data['Sb1'] 
Sb2 = load_data['Sb2'] 
Sb3 = load_data['Sb3'] 


E_c_1 = range(10,18,2)  # Rob1
E_c_2 = range(20,28,2) # Rob2
E_c_3 = range(30,38,2)  # Rob3
E_c_4 = range(40,48,2) # Rob4

E_c = set.union(set.union(set.union(set(E_c_1), set(E_c_2)), set(E_c_3)), set(E_c_4))
E_c = list(E_c)

# %% hyperparameters defination
random.seed(0)
NUM_ACTION = len(E_c) + 1 #number of controllable events pluses one
NUM_OBS = 18  # 16 components and 2 counters
MEMORY_SIZE = np.exp2(14).astype(int)
BATCH_SIZE = np.exp2(12).astype(int)
NUM_EPISODE = 15000 # 15000
# initial state
INIT_OBS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0~15 : state number of Robor 1 ~ 5, INIT_OBS(16): number of input, INIT_OBS(17): number of output
Num_out = 6  # First complete 10 outputs.

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
cwd = os.getcwd() + '\\' + datetime.today().strftime('%Y-%m-%d')

total_step = 0
reward_history = []
good_event_history = []
episode_step_history = [0]
max_epi_reward = -30

# %% train
for num_episode in range(NUM_EPISODE):
    # in each episode, reset initial state and total reward
    S = INIT_OBS
    S_norm = norm_state(S, Num_out)
    episode_reward = 0
    episode_step = 0
    epi_good_event = 0
    epi_action_list = []
    if num_episode > 7500:
        RL.learning_rate -= 4e-7 # 5e-7
    while True:         
        # initialize the Action
        A = RL.choose_action(S_norm)
        # take action and observe
        [S_, all_S_, R, isDone, IfAppear16, stop_ind, selected_action] = \
            StepFun(S, A, Rob1, Rob2, Rob3, Rob4, Sc11, Sc12, Sc21, Sc22, Sc31, Sc32, Sc41, Sc42, Sc43, Sb1, Sb2, Sb3, E_c, Num_out)
        S_norm_ = norm_state(S_, Num_out)
        all_S_norm_ = norm_state(all_S_, Num_out)
        
        # store transition
        RL.store_exp(S_norm, A, R, all_S_norm_)
        # control the learning starting time and frequency
        if total_step > MEMORY_SIZE and (total_step % 10 == 0):
            RL.learn()
        # update states
        episode_reward += R
        episode_step += 1
        epi_good_event += IfAppear16
        S = S_
        # print(S_)
        S_norm = S_norm_
        if isDone or episode_step > Num_out*32:
            if stop_ind == 1:
                stop_reason = 'deadlock'
            elif stop_ind == 2:
                stop_reason = 'Reach the object!'
            else:
                stop_reason = 'reach maximal steps!'
            if max_epi_reward < episode_reward:
                max_epi_reward = episode_reward
            print('episode:', num_episode, '\n',
                  'episode reward:', episode_reward, '\n',
                  'episode step:', episode_step, '\n',
                  'good event:', epi_good_event, '\n',
                  'epsilon value:', RL.epsilon, '\n',
                  'action list:', epi_action_list, '\n',
                  'maximal running step:', np.max(episode_step_history), '\n',
                  'maximal episode reward:', max_epi_reward, '\n',
                  'total good event:', np.sum(good_event_history), '\n',
                  stop_reason, '\n',
                  '*******************************************'
                  )
            # if RL.epsilon > 0.94:
            #     print('num_episode')
            reward_history.append(episode_reward)
            good_event_history.append(epi_good_event)
            episode_step_history.append(episode_step)
            
            # save checkpoint model, if a good model is received
            if episode_reward > 350:
                save_path = cwd + '\\Num_out_' + str(Num_out) + '\\' + str(num_episode) + '_reward' + str(episode_reward) + 'step' + str(episode_step) + '.ckpt'
                saver.save(RL.sess, save_path)
            break
            
            
        total_step += 1
        epi_action_list.append(selected_action)
        

#%% draw cost curve
RL.plot_cost()
RL.plot_reward(reward_history, 250)
RL.plot_epiEvent(good_event_history)


    