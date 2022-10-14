from ast import Raise
import numpy as np
from typing import List, Union, Optional
import logging
import pandas as pd

log = logging.getLogger("rich")
from rich import print

def load_etf(etfs_path : str, etfs : Union[str, List[str]], start = 0, end = None):
    
    xl_file = pd.ExcelFile(etfs_path)
    available_etfs = xl_file.sheet_names
    
    if etfs=='all' : etfs = available_etfs
    
    if type(etfs)== str :

        assert etfs in available_etfs, ValueError(f"{etfs} is not available")
        X =  xl_file.parse(etfs).iloc[:, 0].str.split(',', expand=True).iloc[:,4].to_numpy(np.float64)

    else :
        X_s = []
        
        for el in etfs :
            assert el in  available_etfs, ValueError(f"{el} is not available")
            X_s.append(xl_file.parse(el).iloc[:, 0].str.split(',', expand=True).iloc[:,4].to_numpy(np.float64))

        
        X = np.array(X_s).T

    if end is None :
        end = X.shape[0]


    X = np.log(X)[start:end,:]

    return np.diff(X, axis=0)


class Environment: 

    def __init__(self, etfs_path : str, start=0, end=None):
        
        self.returns = load_etf(etfs_path, 'all', start, end)
    
        self.F_s = [[0]*(self.returns.shape[1])]#initialisation

        self.timespan = self.returns.shape[0]


    def get_state(self,t:int, window : int, is_final = False):
        selected_returns = self.returns[t-window-1:t,:][::-1].flatten()
        last_F_s = self.F_s[-1]
        
        if is_final : 
            return np.array([1, *selected_returns, *last_F_s]), self.returns[window:], np.array(self.F_s)

        else :
            return np.array([1, *selected_returns, *last_F_s])

    def set_action(self, F_t):
        self.F_s.append(F_t)

    def compute_ratio(self):
        pass

    def reset(self):
        self.F_s =[[0]*(self.returns.shape[1])]#i
        print("Env was reset :weary:")




    
        

if __name__ == '__main__':

    print(load_etf('data/Reinforcement Data.xlsx', etfs='all'))
