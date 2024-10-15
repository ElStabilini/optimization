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

    ramsey_output = e.ramsey(
        delay_between_pulses_end = 1000,
        delay_between_pulses_start = 10,
        delay_between_pulses_step = 20,
        detuning = 3000000,
        relaxation_time = 200000,
    )

    if ramsey_output.results.chi2[target][0] > 2:
        raise RuntimeError(
            f"Ramsey fit has chi2 {ramsey_output.results.chi2[target][0]} greater than 2. Stopping."
        )
    
    else:
        ramsey_output.update_platform(e.platform)

    flipping_output = e.flipping(
        nflips_max = 20,
        nflips_step = 1 
    )

    drag_output = e.drag_tuning(
         beta_start = -4,
         beta_end = 4,
         beta_step = 0.5
    )

    if drag_output.results.chi2[target][0] > 2:
        raise RuntimeError(
            f"Drag fit has chi2 {drag_output.results.chi2[target][0]} greater than 2. Stopping."
        )
    
    else:
        drag_output.update_platform(e.platform) #non sono sicura che sia necessario

report(e.path, e.history)
