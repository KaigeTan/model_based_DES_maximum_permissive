# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 13:58:26 2023

@author: kaiget
"""
import scipy.io as sio

class Cluster_tools:
    def __init__(self, mat_file_name):
        # %% Load models
        load_data = sio.loadmat(mat_file_name)  #Get the model file from MATLAB 

        self.Rob1 = load_data['Rob1'] 
        self.Rob2 = load_data['Rob2'] 
        self.Rob3 = load_data['Rob3'] 
        self.Rob4 = load_data['Rob4'] 

        self.Sc11 = load_data['Sc11'] 
        self.Sc12 = load_data['Sc12'] 
        self.Sc21 = load_data['Sc21']
        self.Sc22 = load_data['Sc22']
        self.Sc31 = load_data['Sc31']
        self.Sc32 = load_data['Sc32']
        self.Sc41 = load_data['Sc41']     
        self.Sc42 = load_data['Sc42'] 
        self.Sc43 = load_data['Sc43']

        self.Sb1 = load_data['Sb1'] 
        self.Sb2 = load_data['Sb2'] 
        self.Sb3 = load_data['Sb3'] 

        self.E_c_1 = range(10,18,2)  # Rob1
        self.E_c_2 = range(20,28,2) # Rob2
        self.E_c_3 = range(30,38,2)  # Rob3
        self.E_c_4 = range(40,48,2) # Rob4

        self.E_c = set.union(set.union(set.union(set(self.E_c_1), 
                                                 set(self.E_c_2)), set(self.E_c_3)), set(self.E_c_4))
        self.E_c = list(self.E_c)