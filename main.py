import argparse

import numpy as np
import pandas as pd
import plotly.express as px

from single_agent import DRL_Portfolio_Opt, SingleAssetAgent, load_etf


def main(etf_name):
    x = load_etf("data/Reinforcement Data.xlsx", etfs=etf_name, start=0)
    cut = int(len(x)*0.8)
    print(cut)
    x_train = x[:cut]
    x_test = x[cut:]

    std = np.std(x_train)
    mean = np.mean(x_train)

    x_train = ((x_train - mean) / std).reshape((len(x_train)))
    x_test = ((x_test - mean) / std).reshape((len(x_test)))

    agent = SingleAssetAgent(M=8,lr=0.3)
    optPort = DRL_Portfolio_Opt(x_train, delta=0.0025)

    sharpes = optPort.train(agent, epochs=2000)

    fig = px.line(sharpes)

    fig.update_layout({'title' :'Sharpe Ratio' })
    fig.show()

    train_returns = optPort.returns(agent.positions(x_train))
    df = pd.DataFrame({"RL" : (train_returns).cumsum(), " B&H" : x_train.cumsum()})

    fig1 = px.line(df)
    fig1.update_layout({'title' :'Cumulative returns' })
    fig1.show()

    test_returns = optPort.returns(agent.positions(x_test))
    df = pd.DataFrame({"RL" : (test_returns).cumsum(), " B&H" : x_test.cumsum()})

    fig2 = px.line(df)
    fig2.update_layout({'title' :'Cumulative returns' })
    fig2.show()

    fig.write_image(f"figs/{etf_name}_sharpe.png")
    fig1.write_image(f"figs/{etf_name}_cr_train.png")
    fig2.write_image(f"figs/{etf_name}_cr_test.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-n", default="IWD")

    args = parser.parse_args()
    main(args.n)
