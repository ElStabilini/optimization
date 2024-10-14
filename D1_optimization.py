import numpy as np
import os
import time
import pickle
from qibocal.auto.execute import Executor
from qibocal.cli.report import report
from rb_optimization import rb_optimization, scale_params


start_time = time.time()

target = "D1"
platform = "qw11q"
method = 'nelder-mead' 

executor_path = f'./optimization_data/{target}_{method}'
opt_history_path = f'./opt_analysis/{target}_{method}'
calibration_path = f'/home/elisa/Desktop/Qibo/Qibocal/cal_qw11q/runcard_cal/output_D1'
    #aggiungere cartella di ramsey o simili

with Executor.open(
    "myexec",
    path=executor_path,
    platform=platform,
    targets=[target],
    update=True,
    force=True,
) as e:
 
    e.platform.settings.nshots = 2000
    drag_output = e.drag_tuning(
         beta_start = -4,
         beta_end = 4,
         beta_step = 0.5
    )

    #per ora in questo step faccio tutto manualmente ma meglio sistemare diversamente
    #chiedere se c'è una cartella comune in cui viene salvata la migliore calibrazione

    beta_best = drag_output.results.betas[target]
    ampl_RX = 4.1570229140026074e-2
    freq_RX = 4.958263653e9
    
    scale_factors = np.array([1e-2, 1e-9, 1])
    init_guess = np.array([ampl_RX, freq_RX, beta_best])
    scaled_init_guess = scale_params(init_guess, scale_factors)

    lower_bounds = np.array([-0.5, freq_RX-4e6, beta_best-0.25])
    upper_bounds = np.array([0.5, freq_RX+4e6, beta_best+0.25])
    scaled_bounds = list(zip(scale_params(lower_bounds, scale_factors),
                         scale_params(upper_bounds, scale_factors)))

    opt_results, optimization_history = rb_optimization(e, target, method, scaled_init_guess, scale_factors, scaled_bounds)

report(e.path, e.history)

#save optimization_history as .npz
iterations = np.array([step.iteration for step in optimization_history])
parameters = np.array([step.parameters for step in optimization_history])
objective_values = np.array([step.objective_value for step in optimization_history])
objective_value_error = np.array([step.objective_value_error for step in optimization_history])

os.makedirs(opt_history_path, exist_ok=True)
np.savez(os.path.join(opt_history_path,'optimization_history.npz'), iterations=iterations, parameters=parameters, objective_values=objective_values)


with open(os.path.join(opt_history_path,'optimization_result.pkl'), 'wb') as f:
    pickle.dump(opt_results, f)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

"""TO DO:
    y controllare dove uso scale_factors
    y capire come gestire l'errore    
    y passare 1.875 (avg_gate) come costante all'inizio
    y maxiter
    y xatol: normalizzazione + definizione
    * leggere l'initial guess dalla cartella platform precedente + automatizzare riscalamento
    y nshot
    y modificare delta_clifford
    y provare a variare beta in un piccolo intervallo intorno a quello suggerito da drag
    y spostare report ?
    y vedere quali di questi parametri potrebbe essere interessante variare

"RX": (D1) {
                    "duration": 40,
                    "amplitude": 0.05,
                    "shape": "Gaussian(5)",
                    "frequency": 4900000000,
                    "relative_start": 0,
                    "phase": 0,
                    "type": "qd"
""" 