# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 14:27:25 2023

@author: kaiget
"""
import numpy as np
# reward definition
def reward_cal(x):
    return -75/(2 + np.exp(-10*(x-4))) + 1 
# -2/(1+np.exp(-2*(x-2.5))) + 1
