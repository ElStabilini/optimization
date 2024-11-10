import numpy as np
import os
import time
import pickle
from qibocal.auto.execute import Executor
from qibocal.cli.report import report
from rb_optimization import rb_optimization
from scipy.optimize import Bounds
from qibocal.protocols.utils import round_report

target = "D1"
platform = "qw11q"
executor_path = f'./test'

with Executor.open(
    "myexec",
    path=executor_path,
    platform=platform,
    targets=[target],
    update=True,
    force=True,
) as e:
 
    #Frequency fine tuning using ramsey
    e.platform.settings.nshots = 1024
    ramsey_output = e.ramsey(
        delay_between_pulses_end = 1000,
        delay_between_pulses_start = 10,
        delay_between_pulses_step = 20,
        detuning = 3_000_000,
        relaxation_time = 200000,
    )

    print(ramsey_output.results)
    print(type(ramsey_output.results))
