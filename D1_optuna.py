import numpy as np
import os
import time
import datetime
from qibocal.auto.execute import Executor
from qibocal.cli.report import report
from rb_optuna import rb_optimization, log_optimization

start_time = time.time()
now = datetime.datetime.now()
formatted_time = now.strftime("%Y%m%d_%H%M%S")

target = "D1"
platform = "qw11q"

executor_path = f"../optimization_data/{target}_{formatted_time}"
study_name = f"{formatted_time}"
study_path = f"../optuna_data/{target}_{study_name}"
os.makedirs(os.path.dirname(study_path), exist_ok=True)

with Executor.open(
    "myexec",
    path=executor_path,
    platform=platform,
    targets=[target],
    update=True,
    force=True,
) as e:

    e.platform.settings.nshots = 2000

    ampl_RX = e.platform.qubits[target].native_gates.RX.amplitude
    freq_RX = e.platform.qubits[target].native_gates.RX.frequency

    init_guess = {"amplitude": ampl_RX, "frequency": freq_RX}

    bounds = [
        [-0.5, 0.5],
        [freq_RX - 4e6, freq_RX + 4e6],
    ]

    opt_result = rb_optimization(
        e,
        target,
        init_guess,
        bounds,
        study_name=study_name,
        storage=f"sqlite:///{study_path}.db",
    )

report(e.path, e.history)

end_time = time.time()
elapsed_time = end_time - start_time

log_optimization(study_name, elapsed_time, "../optuna_data/time_log.txt")
