'''IDEA: costruire un ciclo di ottimizzazione della RB al variare dei
parametri dell'impulso di DRAG'''

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import numpy as np
import numpy.typing as npt
from qibolab.qubits import QubitId
from scipy.optimize import minimize

from qibocal import update
from qibocal.auto.execute import Executor
from qibocal.cli.report import report

import numpy as np
from scipy.optimize import minimize

def test_rb_optimization(
        platform,
        target,
        method,
        init_guess
    ):
    
    # Objective function to minimize r_g
    def objective(params):

        with Executor.open(
            "myexec",
            path="test_rb",
            platform=platform,
            targets=[target],
            update=True,
            force=True,
        ) as e:
  
            e.platform.settings.nshots = 2000 

            amplitude, frequency = params

            # Update RX amplitude and frequency
            e.platform.qubits[target].native_gates.RX.amplitude = amplitude
            e.platform.qubits[target].native_gates.RX.frequency = frequency

            # Run DRAG tuning (optional, depending on your setup)
            drag_output = e.drag_tuning(
                beta_start=-4, 
                beta_end=4, 
                    beta_step=0.5
                )

            rb_output = e.rb_ondevice(
                num_of_sequences=1000,
                max_circuit_depth=500,
                delta_clifford=10,
                n_avg=1,
                save_sequences=True,
                apply_inverse=True
            )

        # Calculate infidelity
        cov = rb_output.results.cov
        pars = rb_output.results.pars.get(target)
        stdevs = np.sqrt(np.diag(np.reshape(cov[target], (3, 3))))
        one_minus_p = 1 - pars[2]
        r_c = one_minus_p * (1 - 1 / 2**1)
        r_g = r_c / 1.875  # 1.875 is the average number of gates in a clifford operation
        r_c_std = stdevs[2] * (1 - 1 / 2**1)
        r_g_std = r_c_std / 1.875
        return r_g, r_g_std

    # Initialize the executor
        
        # Perform the minimization
    res = minimize(objective, init_guess, method=method, options={'xatol': 1e-8, 'disp': True})

    report(e.path, e.history)

    return res


target = "D1"
platform = "qw11q"
method = 'nelder-mead'
#init_guess = 
#init guess è da determinare sulla base di quali sono i parametri su cui vado a fare ottimizzazione



