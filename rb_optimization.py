'''IDEA: costruire un ciclo di ottimizzazione della RB al variare dei
parametri dell'impulso di DRAG'''

from dataclasses import dataclass, field
from typing import Optional
import json
import re

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
from qibocal.auto.execute import Executor
from qibocal.cli.report import report

def test_rb_optimization(
        executor, 
        platform,
        target,
        method,
        init_guess #forse potrebbe essere utile fissarla all'interno su un valore teorico
    ):
    
    #inizializzo l'esecutore
    with Executor.open(
        "myexec",
        path="test_rb", #eventualmente inserire il path come argomento della funzione
        platform=platform,
        targets=[target],
        update=True,
        force=True,
    ) as e:
  
        e.platform.settings.nshots = 2000 
        #? c'è un valore "migliore"? affidabile ma non troppo grande per risparmiare tempo?

        '''INSERIRE DRAG - non penso di poter eseguire l'intera routine in questo caso'''
        drag_output = e.drag_tuning(beta_start=-4, beta_end=4, beta_step=0.5) #questa riga è da modificare
        #eseguo darg con certi paramteri e poi trovo la ondevice_rb

        #check parameters value (sono definiti interanmente perchè vario la rb e DRAG rimane fisso)
        rb_output = e.rb_ondevice(
            num_of_sequences = 1000,
            max_circuit_depth = 500,
            delta_clifford = 10,
            n_avg = 1,
            save_sequences = True,
            apply_inverse = True
        ) 
        #esegue tutta la routine, a me in realtà non serve

    infidelity = re.search(r"Gate infidelity.*?(\d+\.\d+)", fitting_report)
    
    
    res = minimize(infidelity, init_guess, method = method, options={'xatol': 1e-8, 'disp': True})
    #? la funzione da minimizzare è l'RB?
    #? l'initial guess potrebbero essere i prametri teorici che ottimizzano il drag
    
    #non sono affatto sicura di aver capito come siano collegati rb e drag


    return res

target = "D1"
platform = "qw11q"
method = 'nelder-mead'
#definire qual è la tolleranza in maniera intlligente



