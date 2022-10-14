from ast import Raise
import numpy as np
from typing import List, Union, Optional

import pandas as pd

def load_etf(etfs_path : str, etfs : Union[str, List[str]]):
    xl_file = pd.ExcelFile(etfs_path)
    available_etfs = xl_file.sheet_names
    dfs = {}
    
    if etfs=='all' : etfs = available_etfs
    if type(etfs)== str :

        assert etfs in available_etfs, ValueError(f"{etfs} is not available")
        X =  xl_file.parse(etfs).iloc[:, 0].str.split(',', expand=True).iloc[:,1].to_numpy(np.float64)

    else :

        X_s = []
        for el in etfs :
            assert el in  available_etfs, ValueError(f"{el} is not available")
            X_s.append(xl_file.parse(el).iloc[:, 0].str.split(',', expand=True).iloc[:,1].to_numpy(np.float64))
        X = np.array(X_s).T

    return X






class environment: 

    def init(self, etfs_path : str):
        
        etf = load_etf(etfs_path, 'all')




if __name__ == '__main__':

    print(load_etf('data/Reinforcement Data.xlsx', etfs='all')[)
