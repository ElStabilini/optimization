'''IDEA: costruire un ciclo di ottimizzazione della RB al variare dei
parametri dell'impulso di DRAG'''

from dataclasses import dataclass, field
from typing import Optional
import json

import numpy as np
import numpy.typing as npt
from qibolab import AcquisitionType, AveragingMode, ExecutionParameters
from qibolab.qubits import QubitId
#from scipy.optimize import curve_fit
from scipy.optimize import minimize

from qibocal.auto.execute import Executor
from qibocal import update
from qibocal.config import log
from qibocal.protocols.drag import _acquisition as drag_acquisition, _fit as drag_fit
from qibocal.protocols.randomized_benchmarking.standard_rb import _acquisition as rb_acquisition, _fit as rb_fit

target = "D1"
platform = "qw11q"

def test_rb_optimization(
        executor, 
        platform,
        target,
        method,
        init_guess #forse potrebbe essere utile fissarla all'interno su un valore teorico
    ):
    
    #eseguo darg con certi paramteri e poi trovo la rb
    #drag_output = e.drag_tuning(beta_start=-4, beta_end=4, beta_step=0.5)         
    params = ([1, 5, 10, 20], 20, None, False, None, None, 100)
    #depths, niter, uncertainties, unrolling, seed, noise_model, noise_params, nshots)
    #? I params potrebbe essere utile tenerli fissati perchè scelgo dei parametri di riferimento per la validazione
    #? ci sono dei valori specifici di questi parametri che è utile settare?

    rb_acq = rb_acquisition(params, platform, target)
    rb = rb_fit(rb_acq)
    #qua ci va qualcosa in mezzo perchè rb_fit restituisce un tipo specifico di dati che non è quello che viene preson in 
        # input dalle funzioni di minimizzazione di scipy
    fidelity = 
    res = minimize(fidelity, init_guess, method = method, options={'xatol': 1e-8, 'disp': True})
    #? la funzione da minimizzare è l'RB?
    #? l'initial guess potrebbero essere i prametri teorici che ottimizzano il drag
    
    #non sono affatto sicura di aver capito come siano collegati rb e drag
    return res

executor = Executor(
    "myexec",
    path="test_rb",
    platform=platform,
    targets=[target],
    update=True,
    force=True,
) 
  
executor.platform.settings.nshots = 2000

