import numpy as np
from random import choice
from NextObservation import Next
from ModularSupervisor import Permit
from reward_definition import reward_cal


# StepFun input: current state, obs: 1 X 18
#                action from RL policy, pattern_index: 0, 1, ... , 16 
# ...
# StepFun output: next state observation, obs_: obs: 1 X 18
#                 reward: 1 X 1
#%%
def StepFun(obs, pattern_index, plant_params, max_diff_num):
        # %% Determine the available event set at the current state
        pattern = Permit(obs, plant_params)
        # pattern contains both controlable and uncontrolable events under the current state
        if obs[-1] > max_diff_num: # if out-in number of products exceed max_diff_num, then disable sending an product
            pattern = np.setdiff1d(pattern, 10)  # Restrict number of wafers input
        if not(pattern_index == 16 or len(pattern) == 1): # when non-disable (#16) or only one action exists in pattern, not disable a event
            pattern = np.setdiff1d(pattern, plant_params.E_c[pattern_index]) # disable a controllable event

        # %% iterate to the next state
        # Calculate the running cost
        isDone = 0
        reward = 0.1*len(pattern)
        stop_ind = 0
        all_S_ = []
        is_deadlock = 0
        good_evt = 0
        
        # new version
        if len(pattern) != 0:
            action = choice(pattern)        # random selection of the executed event
            obs_ = Next(obs, action, plant_params)    # iterate to the next state
            # iterate all available actions in the pattern and calculate the all_S_
            for event in pattern:
                S_temp_ = Next(obs, event, plant_params)
                all_S_.append(S_temp_)
                Enable_P_temp_ = Permit(S_temp_, plant_params)
                if len(Enable_P_temp_) == 0:
                    is_deadlock = 1  # a deadlock is reached
                    obs_ = obs
                    all_S_ = [obs]
                    action = -1
                    stop_ind = 1
                    # break
            
        else:
            action = -2
            is_deadlock = 1 # a deadlock is reached
            obs_ = obs
            all_S_ = [obs]
            stop_ind = 1
            isDone = 1
        # if no possible actions in the next state/intersection is empty set, terminate the episode
        if is_deadlock == 1:
             # isDone = 1
             reward = -1 - isDone*9   # deadlock,
        else:
            # a new tool is manufactured
            if action == 15:
                good_evt = 1
            # TODO: revise the reward definition
            # here we calculate the average value of all possible actions
            for i_action in pattern:
                i_reward = 0
                if i_action == 15:
                    i_reward = 15*reward_cal(obs[-1])
                elif i_action == 14:
                    i_reward = 10*reward_cal(obs[-1])
                elif i_action == 24:
                    i_reward = 7.5*reward_cal(obs[-1])
                elif i_action == 34:
                    i_reward = 4*reward_cal(obs[-1])
                elif i_action == 40:
                    i_reward = 2*reward_cal(obs[-1])
                elif i_action == 30:
                    i_reward = 1*reward_cal(obs[-1])
                elif i_action == 20:
                    i_reward = 0.5*reward_cal(obs[-1])
                    
                reward += i_reward
                
            reward /= len(pattern)
            # if obs_[-1] == max_num:
            #     isDone = 1 #Complete!
            #     stop_ind = 2
            #     reward += 50
                

        return(obs_, all_S_, reward, isDone, good_evt, stop_ind, action)
        
    
    
    