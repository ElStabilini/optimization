import numpy as np
from qibocal.auto.execute import Executor
from qibolab import pulses
from qibocal.cli.report import report
from pathlib import Path

AVG_GATE = 1.875
#ramsey, flipping, drag, randomized benchmarking

target = "D1"
platform = "qw11q"
path = Path().parent / "optimization_data" / "sequence" 

with Executor.open(
    "myexec",
    path = path,
    platform = platform,
    targets = [target],
    update = False, #True, 
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
        nflips_step = 1 #valutare nel caso di flip iterativi
    )

    rb_output = e.rb_ondevice(
        num_of_sequences=1000,
        max_circuit_depth=1000,
        delta_clifford=10,
        n_avg=1,
        save_sequences=True,
        apply_inverse=True
    )

    pars = rb_output.results.pars.get(target)
    one_minus_p = 1 - pars[2]
    r_c = one_minus_p * (1 - 1 / 2**1)
    r_g = r_c / AVG_GATE

    
    #fare flipping iterativi allungando la frequenza (fino a 200-300 flip) <--

report(e.path, e.history)
