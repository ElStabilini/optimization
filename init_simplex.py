import numpy as np
import os
import time
import pickle
import datetime
from qibocal.auto.execute import Executor
from qibocal.cli.report import report
from rb_init_simplex import rb_optimization
from scipy.optimize import Bounds

start_time = time.time()

target = "D1"
platform = "qw11q"
method = 'Nelder-Mead'
now = datetime.datetime.now()
formatted_time = now.strftime("%Y%m%d_%H%M%S") 

executor_path = f'./optimization_data/{target}_init_simplex_{formatted_time}'
opt_history_path = f'./opt_analysis/{target}_init_simplex_{formatted_time}'

with Executor.open(
    "myexec",
    path=executor_path,
    platform=platform,
    targets=[target],
    update=True,
    force=True,
) as e:
 
    #Frequency fine tuning using ramsey
    e.platform.settings.nshots = 1024
    ramsey_output = e.ramsey(
        delay_between_pulses_end = 1000,
        delay_between_pulses_start = 10,
        delay_between_pulses_step = 20,
        detuning = 3_000_000,
        relaxation_time = 200000,
    )

    freq_RX = ramsey_output.results.frequency[target][0]
    sigma_freq = 1.5*ramsey_output.results.frequency[target][1]

    #Amplitude fine tuning using flipping
    e.platform.settings.nshots = 1024
    flipping_output = e.flipping(
        nflips_max = 20,
        nflips_step = 1,
    )

    ampl_RX = flipping_output.results.amplitude[target][0]
    sigma_ampl = 1.5*flipping_output.results.amplitude[target][1]

    #I want to change from initial guess to initial simplex
    init_simplex = np.array(
        [ampl_RX+sigma_ampl, freq_RX+sigma_freq],
        [ampl_RX, freq_RX-sigma_freq],
        [ampl_RX-sigma_ampl, freq_RX])
    lower_bounds = np.array([-0.5, freq_RX-4e6])  
    upper_bounds = np.array([0.5, freq_RX+4e6])   
    bounds = Bounds(lower_bounds, upper_bounds)

    opt_results, optimization_history = rb_optimization(e, target, method, init_simplex, bounds)

report(e.path, e.history)

end_time = time.time()
elapsed_time = end_time - start_time

#save optimization_history as .npz
iterations = np.array([step.iteration for step in optimization_history])
parameters = np.array([step.parameters for step in optimization_history])
objective_values = np.array([step.objective_value for step in optimization_history])
objective_value_error = np.array([step.objective_value_error for step in optimization_history])

os.makedirs(opt_history_path, exist_ok=True)
np.savez(os.path.join(opt_history_path,'optimization_history.npz'), 
         iterations=iterations, 
         parameters=parameters, 
         objective_values=objective_values, 
         objective_value_errors=objective_value_error)

result_data = {
    'optimization_result': opt_results,
    'elapsed_time': elapsed_time
}

with open(os.path.join(opt_history_path,'optimization_result.pkl'), 'wb') as f:
    pickle.dump(result_data, f)
