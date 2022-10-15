#%%
import logging
import pandas as pd
from rich.console import Group
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           SpinnerColumn, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)

from src._environment import Environment
from agent.Agent import Agent

import plotly.express as px
import plotly.io as pio

pio.renderers.default = 'notebook_connected'


FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)

def compute_performance(x_t, F, M, mu):
    returns = x_t[:,M]
    return mu * (returns@ F)

def main():

    env = Environment('data/Reinforcement Data.xlsx', end=2370, etfs='all')
    M = 104
    mu=100
    agent = Agent(M=M, N=env.n_etf, rho=.004, mu=mu, delta=0, T=env.timespan)
    agent.load("weights.npy")
    epoch_progress = Progress(TextColumn("[bold blue] Epoching",), BarColumn(), MofNCompleteColumn(), TextColumn('[ elapsed'), TimeElapsedColumn(), TextColumn('| eta'), TimeRemainingColumn())
    iter_progress = Progress(TextColumn("[bold blue] Run through dataset",), BarColumn(), MofNCompleteColumn())

    epoch_task = epoch_progress.add_task("0", total=n_epoch)

    progress_group = Group(
    Panel(Group(epoch_progress, iter_progress))
)
    sharpe_ratios = []
    perfs = []
    with Live(progress_group):

        iter_task = iter_progress.add_task("zebi",total=env.timespan)
        
        current_perf= []
        for t in range(M+1,env.timespan):
            x_t  = env.get_state(t=t,window = M)
            F = agent.forward(x_t)
            env.set_action(F)
            
            iter_progress.advance(iter_task)
            current_perf.append(compute_performance(x_t,F, M, mu))
        x_T,r, F_s = env.get_state(t=env.timespan, window=M, is_final=True)
        agent.compute_derivatives(r, x_T, F_s=F_s)
        sharpe_ratios.append(agent.compute_sharpe_ratio())
        env.reset()
        epoch_progress.update(task_id=epoch_task, advance=1)
        iter_progress.update(task_id=iter_task, visible=False)
        iter_progress.stop_task(iter_task)
        perfs.append(current_perf)
        
    print(sharpe_ratios)
    perf_df = pd.DataFrame(data=perfs)
    return perf_df


        #agent.save_weights



#%%
if __name__ == '__main__':


    main()

# %%
