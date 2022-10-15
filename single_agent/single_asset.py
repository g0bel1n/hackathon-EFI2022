#%%
import numpy as np
import pandas as pd
from rich.progress import track
from rich import print


def load_etf(etfs_path : str, etfs = 'IWD', start = 0, end = 2960):

    xl_file = pd.ExcelFile(etfs_path)
    available_etfs = xl_file.sheet_names
    
    dates = xl_file.parse(available_etfs[0]).iloc[:, 0].str.split(',', expand=True).iloc[:,0]
    print(dates)
    assert etfs in available_etfs, ValueError(f"{etfs} is not available")
    X =  xl_file.parse(etfs).iloc[:, 0].str.split(',', expand=True).iloc[:,4].to_numpy(np.float64)
    X = X.reshape((X.shape[0],1))

            

    return X[1:len(X),:]-X[:len(X)-1,:], dates


#%%
class DRL_Portfolio_Opt:

    def __init__(self, returns: np.ndarray, delta) -> None:
        self.etf_returns = returns
        self.delta = delta
        

    def sharpe_ratio(self):
        return self.returns.mean() / self.returns.std()
 


    def returns(self, Ft):
        T = len(Ft)
        portfolio_returns = Ft[: T - 1] * self.etf_returns[1:T] - self.delta * np.abs(Ft[1:T] - Ft[: T - 1])

        return np.concatenate([[0], portfolio_returns])


    def train(self,agent, epochs=2000):
        sharpes = np.zeros(epochs) # store sharpes over time
        for i in track(range(epochs), description="Optimizing..."):
            R = self.returns(agent.positions(self.etf_returns))
            sharpe = agent.update(self.etf_returns,self.delta,R)
            sharpes[i] = sharpe
        
        
        return sharpes


# %%
