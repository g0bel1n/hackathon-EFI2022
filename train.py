import logging

from rich.console import Group
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           SpinnerColumn, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)

from src._environment import Environment
from agent.Agent import Agent


FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)


def train(n_epoch : int = 100):

    env = Environment('data/Reinforcement Data.xlsx', end=2370, etfs='CLY')
    M = 104
    agent = Agent(M=M, N=env.n_etf, rho=.004, mu=100, delta=0, T=env.timespan)

    epoch_progress = Progress(TextColumn("[bold blue] Epoch n°{task.description}",), SpinnerColumn(spinner_name='growHorizontal'), BarColumn(), MofNCompleteColumn(), TextColumn('[ elapsed'), TimeElapsedColumn(), TextColumn('| eta'), TimeRemainingColumn())
    iter_progress = Progress(TextColumn("[bold blue] Run through dataset",), SpinnerColumn(spinner_name='growHorizontal'), BarColumn(), MofNCompleteColumn())

    epoch_task = epoch_progress.add_task("0", total=n_epoch+1)

    progress_group = Group(
    Panel(Group(epoch_progress, iter_progress))
)
    sharpe_ratios = []
    with Live(progress_group):
        for n in range(n_epoch):
            iter_task = iter_progress.add_task("zebi",total=env.timespan)
            

            for t in range(105,env.timespan):
                x_t  = env.get_state(t=t,window = M)
                F = agent.forward(x_t)
                env.set_action(F)
                if n==n_epoch-1 :
                    sharpe_ratios.append(agent.compute_sharpe_ratio())
                iter_progress.advance(iter_task)
            x_T,r, F_s = env.get_state(t=env.timespan, window=M, is_final=True)
            agent.compute_derivatives(r, x_T, F_s=F_s)
            agent.gradient_ascent()
            env.reset()
            epoch_progress.update(task_id=epoch_task, description=f"{n}")
            iter_progress.update(task_id=iter_task, visible=False)
            iter_progress.stop_task(iter_task)

        #agent.save_weights



if __name__ == "__main__":

    train()
