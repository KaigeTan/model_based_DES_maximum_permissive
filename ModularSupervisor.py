

from AvailableEvents import Enb
import numpy as np
import random


def Permit(obs, plant_params):
    
    # Plants
    Enable_P1 = Enb(obs[0], plant_params.Rob1)   
    Enable_P2 = Enb(obs[1], plant_params.Rob2)
    Enable_P3 = Enb(obs[2], plant_params.Rob3)
    Enable_P4 = Enb(obs[3], plant_params.Rob4)
    
    
    Enable_P = np.union1d(Enable_P1, Enable_P2)
    Enable_P = np.union1d(Enable_P, Enable_P3)
    Enable_P = np.union1d(Enable_P, Enable_P4)
    

    # modular supervisors
    Enable_Sc11 = Enb(obs[4], plant_params.Sc11)
    Enable_Sc12 = Enb(obs[5], plant_params.Sc12)
    Enable_Sc21 = Enb(obs[6], plant_params.Sc21)
    Enable_Sc22 = Enb(obs[7], plant_params.Sc22)
    Enable_Sc31 = Enb(obs[8], plant_params.Sc31)
    Enable_Sc32 = Enb(obs[9], plant_params.Sc32)
    Enable_Sc41 = Enb(obs[10], plant_params.Sc41)
    Enable_Sc42 = Enb(obs[11], plant_params.Sc42)
    Enable_Sc43 = Enb(obs[12], plant_params.Sc43)
    Enable_Sb1 = Enb(obs[13], plant_params.Sb1)
    Enable_Sb2 = Enb(obs[14], plant_params.Sb2)
    Enable_Sb3 = Enb(obs[15], plant_params.Sb3)
    
    # PM capacity
    Enable_c1 = np.intersect1d(Enable_Sc11, Enable_Sc12)
    Enable_c2 = np.intersect1d(Enable_Sc21, Enable_Sc22)
    Enable_c3 = np.intersect1d(Enable_Sc31, Enable_Sc32)
    Enable_c4 = np.intersect1d(Enable_Sc41, Enable_Sc42)
    Enable_c4 = np.intersect1d(Enable_c4, Enable_Sc43)
    
    Enable_c = np.intersect1d(Enable_c1, Enable_c2)
    Enable_c = np.intersect1d(Enable_c, Enable_c3)
    Enable_c = np.intersect1d(Enable_c, Enable_c4)
    
    # Buffer capacity
    Enable_b = np.intersect1d(Enable_Sb1, Enable_Sb2)
    Enable_b = np.intersect1d(Enable_b, Enable_Sb3)
    # PM and buffer       
    Enable = np.intersect1d(Enable_c, Enable_b) 
    # plant, PM, and buffer      
    Enable_P_S = np.intersect1d(Enable_P, Enable)
    
    return Enable_P_S

def random_pick(some_list, pro):
    x = random.uniform(0,1)
    cumul = 0
    for item, item_pro in zip(some_list, pro):
        cumul += item_pro
        if x < cumul:
            break
    return item