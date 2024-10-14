import numpy as np
from qibocal.auto.execute import Executor
from qibolab import pulses
from qibocal.cli.report import report


#ramsey, flipping, drag, randomized benchmarking

target = "D1"
platform = "qw11q"
path = "sequence" 

with Executor.open(
    "myexec",
    path = path,
    platform = platform,
    targets = [target],
    update = True,
    force = True,
) as e:
    
    e.platform.settings.nshots = 1024

    ramsey = e.ramsey(
        delay_between_pulses_end = 1000,
        delay_between_pulses_start = 10,
        delay_between_pulses_step = 20,
        detuning = 3000000,
        relaxation_time = 200000,
    )

    flipping = e.flipping(
        nflips_max = 20,
        nflips_step = 1 
    )

    drag_output = e.drag_tuning(
         beta_start = -4,
         beta_end = 4,
         beta_step = 0.5
    )

report(e.path, e.history)
