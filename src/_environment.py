from ast import Raise
import numpy as np
from typing import List, Union, Optional
import logging
import pandas as pd

log = logging.getLogger("rich")


def load_etf(etfs_path : str, etfs : Union[str, List[str]]):
    xl_file = pd.ExcelFile(etfs_path)
    available_etfs = xl_file.sheet_names
    dfs = {}
    
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

    X = np.log(X)

    return np.diff(X)


class Environment: 

    def __init__(self, etfs_path : str):
        
        self.returns = load_etf(etfs_path, 'all')
        self.F_s = [[1]]#initialisation

        self.timespan = self.returns.shape[0]


    def get_state(self,t:int, window : int):
        selected_returns = self.returns[t:t-window,:].flatten()
        last_F_s = self.F_s[-1]
        return np.array([1, *selected_returns, *last_F_s])

    def set_action(self, F_t):
        self.F_s.append(F_t)

    def compute_ratio(self):
        pass

    def reset(self):
        self.F_s = [[1]]
        log.info("Env was reset")




    
        

if __name__ == '__main__':

    print(load_etf('data/Reinforcement Data.xlsx', etfs='all'))
