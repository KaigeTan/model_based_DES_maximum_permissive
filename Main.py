# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 19:48:22 2023

@author: junju
"""

import scipy.io as sio
from NextObservation import Next


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

#%%
INIT_OBS = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 0~15 : state number of Robor 1 ~ 5, INIT_OBS(16): number of input, INIT_OBS(17): number of output
Num_out = 1 #number of outputs
Test_Action_list = [10, 9, 12, 11, 20, 19, 22, 21, 30, 29, 32, 31, 40, 39, 42, 41, 44, 43, 46, 45, 34, 33, 36, 35, 24, 23, 26, 25, 14, 13, 16, 15]
obs = INIT_OBS
for event in Test_Action_list:
    obs_ = Next(obs, event, Rob1, Rob2, Rob3, Rob4, Sc11, Sc12, Sc21, Sc22, Sc31, Sc32, Sc41, Sc42, Sc43, Sb1, Sb2, Sb3)
    obs = obs_   # the result shows that obs(16) = 1, obs(17) = 1, which means that the number of inputs and outputs are 1 and 1, respectively, after the execution of 
    #the event list.