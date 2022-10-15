import argparse

import numpy as np
import pandas as pd
import plotly.express as px

from single_agent import DRL_Portfolio_Opt, SingleAssetAgent, load_etf


def main(etf_name: str, show: bool):
    x, dates = load_etf("data/Reinforcement Data.xlsx", etfs=etf_name, start=0)
    cut = int(len(x)*0.8)
    print(cut)
    dates_train, dates_test = dates.iloc[:cut], dates.iloc[cut:]
    x_train = x[:cut]
    x_test = x[cut:]

    std = np.std(x_train)
    mean = np.mean(x_train)

    x_train = ((x_train - mean) / std).reshape((len(x_train)))
    x_test = ((x_test - mean) / std).reshape((len(x_test)))

    agent = SingleAssetAgent(M=8,lr=0.3)
    optPort = DRL_Portfolio_Opt(x_train, delta=0.0025)

    sharpes = optPort.train(agent, epochs=2000)

    fig = px.line(y=sharpes)

    fig.update_layout({'title' :'Sharpe Ratio' })
    

    train_returns = optPort.returns(agent.positions(x_train))
    df = pd.DataFrame({"RL" : (train_returns).cumsum(), " B&H" : x_train.cumsum()})
    df.index = dates_train
    fig1 = px.line(df)
    fig1.update_layout({'title' :'Cumulative returns train set' })
    

    test_returns = optPort.returns(agent.positions(x_test))
    df2 = pd.DataFrame({"RL" : (test_returns).cumsum(), " B&H" : x_test.cumsum()})
    df2.index = dates_test.iloc[1:]
    fig2 = px.line(df2)
    fig2.update_layout({'title' :'Cumulative returns Test set' })
    

    if show:
        fig.show()
        fig1.show()
        fig2.show()


    fig.write_image(f"figs/{etf_name}_sharpe.png")
    fig1.write_image(f"figs/{etf_name}_cr_train.png")
    fig2.write_image(f"figs/{etf_name}_cr_test.png")
    
    print("Training ended whithout issues")

if __name__ == "__main__":  
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", default="IWD")
    parser.add_argument("--show", default=False)

    args = parser.parse_args()
    main(args.n, args.show)
