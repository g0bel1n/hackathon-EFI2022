#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rich.progress import track


from single_asset_agent import SingleAssetAgent


def load_etf(etfs_path : str, etfs = 'CLY', start = 0, end = 2960):

    xl_file = pd.ExcelFile(etfs_path)
    available_etfs = xl_file.sheet_names
    

    assert etfs in available_etfs, ValueError(f"{etfs} is not available")
    X =  xl_file.parse(etfs).iloc[:, 0].str.split(',', expand=True).iloc[:,4].to_numpy(np.float64)
    X = X.reshape((X.shape[0],1))

            

    return X[1:len(X),:]-X[:len(X)-1,:]


class DRL_Portfolio_Opt:

    def __init__(self, returns: np.ndarray, delta) -> None:
        self.etf_returns = returns
        self.delta = delta
        

    def sharpe_ratio(self):
        return self.returns.mean() / self.returns.std()
 


    def returns(self, Ft):
        T = len(self.etf_returns)
        portfolio_returns = Ft[0:T - 1] * self.etf_returns[1:T] - self.delta * np.abs(Ft[1:T] - Ft[0:T - 1])
        return np.concatenate([[0], portfolio_returns])


    def train(self,agent, epochs=2000):
        sharpes = np.zeros(epochs) # store sharpes over time
        for i in track(range(epochs), description="Optimizing..."):
            R = self.returns(agent.positions(self.etf_returns))
            sharpe = agent.update(self.etf_returns,self.delta,R)
            sharpes[i] = sharpe
        
        
        print("finished training")
        return sharpes


# %%
x = load_etf("../../hackathon-efi2022/data/Reinforcement Data.xlsx", etfs='IWD', start=0, end=2000)

x_train = x[:2000]
x_test = x[2001:]

std = np.std(x_train)
mean = np.mean(x_train)

x_train = ((x_train - mean) / std).reshape((len(x_train)))
x_test = ((x_test - mean) / std).reshape((len(x_test)))
#%%

#%%
agent = SingleAssetAgent(M=8,lr=0.3)
optPort = DRL_Portfolio_Opt(x_train, delta=0.0025)
np.random.seed(0)
sharpes = optPort.train(agent, epochs=1000)

# %%
plt.plot(sharpes)
plt.xlabel('Epoch Number')
plt.ylabel('Sharpe Ratio')

# %%
train_returns = optPort.returns(positions(x_train, theta), x_train, 0.0025)
plt.plot((train_returns).cumsum(), label="Reinforcement Learning Model", linewidth=1)
plt.plot(x_train.cumsum(), label="Buy and Hold", linewidth=1)
plt.xlabel('Ticks')
plt.ylabel('Cumulative Returns');
plt.legend()
plt.title("RL Model vs. Buy and Hold - Training Data");

# %%
test_returns = returns(positions(x_test, theta), x_test, 0.0025)
plt.plot((test_returns).cumsum(), label="Reinforcement Learning Model", linewidth=1)
plt.plot(x_test.cumsum(), label="Buy and Hold", linewidth=1)
plt.xlabel('Ticks')
plt.ylabel('Cumulative Returns');
plt.legend()
plt.title("RL Model vs. Buy and Hold - Test Data");

