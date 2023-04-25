# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 12:52:44 2022

@author: KaigeT
"""
import numpy as np

def norm_state(S, delta_max):
    max_state = np.array([16, 16, 16, 16, 2, 2, 2, 2, 2, 2, 2, 2, 2, 5, 5, 5, delta_max])
    S_norm_arr = np.array(S)/max_state
    S_norm = S_norm_arr.tolist()
    return S_norm