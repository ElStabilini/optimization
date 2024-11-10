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
    e.platform.settings.nshots = 2000
    drag_output = e.drag_tuning(
         beta_start = -1,
         beta_end = 1,
         beta_step = 0.1
    )

    print(drag_output.results)

report(e.path, e.history)

