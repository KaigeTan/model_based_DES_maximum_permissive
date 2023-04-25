# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 10:17:11 2022

@author: Jonny
"""
import numpy as np

def Next(State, action, plant_params):
    
    # the action 10 here is the action 11 in MATLAB
    
    X0 = State[0]   #plants
    X1 = State[1]
    X2 = State[2]
    X3 = State[3] 
    
    X4 = State[4]  # modular supervisors
    X5 = State[5]
    X6 = State[6]
    X7 = State[7]
    X8 = State[8]
    X9 = State[9]
    X10 = State[10]
    X11 = State[11]
    X12 = State[12]
    X13 = State[13]
    X14 = State[14]
    X15 = State[15]
    
    X16 = State[16] #counters
    
    # Plants
    if action in [9,10,11,12,13,14,15,16]:
        X0_ = np.where(plant_params.Rob1[X0, :, action] == 1)
        X0_ = X0_[0][0]
    else:
        X0_ = X0
    
    
    if action in [19,20,21,22,23,24,25,26]:
        X1_ = np.where(plant_params.Rob2[X1, :, action] == 1)
        X1_ = X1_[0][0]
    else:
        X1_ = X1
    
    
    if action in [29,30,31,32,33,34,35,36]:
        X2_ = np.where(plant_params.Rob3[X2, :, action] == 1)
        X2_ = X2_[0][0]
    else:
        X2_ = X2
       
    
    if action in [39,40,41,42,43,44,45,46]:
        X3_ = np.where(plant_params.Rob4[X3, :, action] == 1)
        X3_ = X3_[0][0]
    else:
       X3_ = X3
        
        
    # modular supevisors
    X4_ = np.where(plant_params.Sc11[X4, :, action] == 1)   
    X4_ = X4_[0][0]
    
    X5_ = np.where(plant_params.Sc12[X5, :, action] == 1)   
    X5_ = X5_[0][0]
    
    X6_ = np.where(plant_params.Sc21[X6, :, action] == 1)   
    X6_ = X6_[0][0]
    
    X7_ = np.where(plant_params.Sc22[X7, :, action] == 1)   
    X7_ = X7_[0][0]
    
    X8_ = np.where(plant_params.Sc31[X8, :, action] == 1)   
    X8_ = X8_[0][0]
    
    X9_ = np.where(plant_params.Sc32[X9, :, action] == 1)   
    X9_ = X9_[0][0]
    
    X10_ = np.where(plant_params.Sc41[X10, :, action] == 1)   
    X10_ = X10_[0][0]
    
    X11_ = np.where(plant_params.Sc42[X11, :, action] == 1)   
    X11_ = X11_[0][0]
    
    X12_ = np.where(plant_params.Sc43[X12, :, action] == 1)   
    X12_ = X12_[0][0]
    
    X13_ = np.where(plant_params.Sb1[X13, :, action] == 1)   
    X13_ = X13_[0][0]
    
    X14_ = np.where(plant_params.Sb2[X14, :, action] == 1)   
    X14_ = X14_[0][0]
    
    X15_ = np.where(plant_params.Sb3[X15, :, action] == 1)   
    X15_ = X15_[0][0]
    
    # Counters of in put and output, respectively
    if action == 11-1: # enter one tool
        X16_ = X16 + 1
    elif action == 16-1: # leave one tool
        X16_ = X16 - 1
    else:
        X16_ = X16
    
    State_ = [X0_, X1_, X2_, X3_, X4_, X5_, X6_, X7_, X8_, X9_, X10_, X11_, X12_, X13_, X14_, X15_, X16_]
    
    return(State_)
    
    
    
    
    
    
    
    
    
    