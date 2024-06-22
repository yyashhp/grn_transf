import numpy as np



class DataPair:
    def __init__(self, sample_data, true_grn_ID, true_grn):
        self.sample_data = sample_data
        self.true_grn_ID = true_grn_ID
        self.true_grn = true_grn

class Data_Wendy_True:
    
    def __init__(self, sample_data, true_grn_ID, true_grn, wendy_estimate):
        self.sample_data = sample_data
        self.true_grn_ID = true_grn_ID
        self.true_grn = true_grn
        self.wendy_estimate = wendy_estimate