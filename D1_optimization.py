import numpy as np
import os
import time
import pickle
from qibocal.auto.execute import Executor
from qibocal.cli.report import report
from rb_optimization import rb_optimization
from scipy.optimize import Bounds


start_time = time.time()

target = "D1"
platform = "qw11q"
method = 'SLSQP' 

executor_path = f'../optimization_data/{target}_{method}_post_ft_true'
opt_history_path = f'./opt_analysis/{target}_{method}_post_ft_true'

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


    beta_best = drag_output.results.betas[target]
    ampl_RX = e.platform.qubits[target].native_gates.RX.amplitude
    freq_RX = e.platform.qubits[target].native_gates.RX.frequency
    
    init_guess = np.array([ampl_RX, freq_RX, beta_best])

    lower_bounds = np.array([-0.5, freq_RX-4e6, beta_best-0.25])  
    upper_bounds = np.array([0.5, freq_RX+4e6, beta_best+0.25])   
    bounds = Bounds(lower_bounds, upper_bounds)

    opt_results, optimization_history = rb_optimization(e, target, method, init_guess, bounds)

report(e.path, e.history)

#save optimization_history as .npz
iterations = np.array([step.iteration for step in optimization_history])
parameters = np.array([step.parameters for step in optimization_history])
#capire come salvare parameters errors
objective_values = np.array([step.objective_value for step in optimization_history])
objective_value_error = np.array([step.objective_value_error for step in optimization_history])

os.makedirs(opt_history_path, exist_ok=True)
np.savez(os.path.join(opt_history_path,'optimization_history.npz'), 
         iterations=iterations, 
         parameters=parameters, 
         objective_values=objective_values, 
         objective_value_errors=objective_value_error)

with open(os.path.join(opt_history_path,'optimization_result.pkl'), 'wb') as f:
    pickle.dump(opt_results, f)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")
