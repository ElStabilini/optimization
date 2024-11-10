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
    flipping_output = e.flipping(
        nflips_max = 20,
        nflips_step = 1,
    )

    print(flipping_output.results)
    print(type(flipping_output.results))
