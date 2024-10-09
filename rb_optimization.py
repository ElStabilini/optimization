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


#objective function to minimize
def objective(params, platform, target):

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

        e.platform.qubits[target].native_gates.RX.amplitude = amplitude
        e.platform.qubits[target].native_gates.RX.frequency = frequency

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

        report(e.path, e.history)

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

def test_rb_optimization(
        platform,
        target,
        method,
        init_guess
    ):
    
    res = minimize(objective, init_guess, args=(platform, target), method=method, options={'xatol': 1e-2, 'disp': True}, maxiter = 100)
    
    return res


target = "0"
platform = "dummy"
method = 'nelder-mead' #forse non la migliore? Non ho idea del landscape
init_guess = [0.041570229140026074, 4958263653]


"""TO DO: 
    * vedere quali di questi parametri potrebbe essere interessante variare
    * maxiter
    * xatol (considerato anche con quali ordini di grandezza sto lavorando)
    * leggere l'initial guess direttamente dalla migliore calibrazione su qibolab
    * nshot
    * ha senso provare a variare la duration
    * provare a variare beta in un piccolo intervallo intorno a quello suggerito da drag
    * spostare report ?

"RX": (D1) {
                    "duration": 40,
                    "amplitude": 0.05,
                    "shape": "Gaussian(5)",
                    "frequency": 4900000000,
                    "relative_start": 0,
                    "phase": 0,
                    "type": "qd"
"""



