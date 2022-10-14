import logging

import numpy as np
from rich.console import Group
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import (BarColumn, MofNCompleteColumn, Progress,
                           SpinnerColumn, TextColumn, TimeElapsedColumn,
                           TimeRemainingColumn)

from src._environment import Environment

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()]
)


def main():
    env = Environment('data/Reinforcement Data.xlsx', start=2371)
    M = 104
    n_epoch = 10

    #init agent
    epoch_progress = Progress(TextColumn("[bold blue] Epoch n°{task.description}",), SpinnerColumn(spinner_name='growHorizontal'), BarColumn(), MofNCompleteColumn(), TextColumn('[ elapsed'), TimeElapsedColumn(), TextColumn('| eta'), TimeRemainingColumn(),TextColumn(' ]'))
    iter_progress = Progress(TextColumn("[bold blue] Run through dataset",), SpinnerColumn(spinner_name='growHorizontal'), BarColumn(), MofNCompleteColumn())

    epoch_task = epoch_progress.add_task("0", total=n_epoch+1)

    progress_group = Group(
    Panel(Group(epoch_progress, iter_progress))
)

    sharpe_ratios = []
    with Live(progress_group):
        for n in range(n_epoch):
            iter_task = iter_progress.add_task("zebi",total=env.timespan)

            for t in range(env.timespan):
                x_t = env.get_state(t=t,window = M)
                #get F from agent(x_t)
                #env.set_action(F)
                #sharpe_ratios.append(agent.get_sharpe())

                iter_progress.advance(iter_task)
                
            env.reset()
            epoch_progress.update(task_id=epoch_task, description=f"{n}")
            iter_progress.update(task_id=iter_task, visible=False)
            iter_progress.stop_task(iter_task)

        #agent.save_weights




if __name__ == '__main__':
    main()
