# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Tlstm_unit(nn.Module):
    """Base unit for T_lstm
    """
    
    def __init__(self, hid_mem, concat_input):
        
        # 
        super(Tlstm_unit, self).__init__()
        
    def forward(self,x, hidden)

