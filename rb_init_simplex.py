import numpy as np
from scipy.optimize import minimize
from qibocal.auto.execute import Executor
from qibolab import pulses
from dataclasses import dataclass

AVG_GATE = 1.875 # 1.875 is the average number of gates in a clifford operation
error_storage = {'error': None}

@dataclass
class OptimizationStep:
    iteration: int
    parameters: np.ndarray
    objective_value: float
    objective_value_error: float

#objective function to minimize
def objective(params, e, target):

    amplitude, frequency = params

    e.platform.qubits[target].native_gates.RX.amplitude = amplitude
    e.platform.qubits[target].native_gates.RX.frequency = frequency
    
    rb_output = e.rb_ondevice(
        num_of_sequences=1000,
        max_circuit_depth=1000,
        delta_clifford=10,
        n_avg=1,
        save_sequences=True,
        apply_inverse=True
    )

    # Calculate infidelity and error
    stdevs = np.sqrt(np.diag(np.reshape(rb_output.results.cov[target], (3, 3))))

    pars = rb_output.results.pars.get(target)
    one_minus_p = 1 - pars[2]
    r_c = one_minus_p * (1 - 1 / 2**1)
    r_g = r_c / AVG_GATE
    r_c_std = stdevs[2] * (1 - 1 / 2**1)
    r_g_std = r_c_std / AVG_GATE
    
    error_storage['error'] = r_g_std

    print('terminating objective call')
    return r_g


def rb_optimization(
        executor : Executor,
        target : str,
        method : str,
        initial_simplex : list[list[float]],
        bounds
    ):
    
    optimization_history = []
    iteration_count = 0

    def callback(x, f=None):
        nonlocal iteration_count
        if f is None:
            # If the optimization method doesn't provide f, need to calculate it
            f = objective(x, executor, target)
        
        step = OptimizationStep(
            iteration=iteration_count,
            parameters=np.copy(x),
            objective_value=f,
            objective_value_error = error_storage['error']
        )
        optimization_history.append(step)
        iteration_count += 1
        print(f"Completed iteration {iteration_count}, objective value: {f}")

    res = minimize(objective, args=(executor, target), method=method, 
                   tol=1e-4, options = {"maxiter" : 40, "initial_simplex": initial_simplex}, 
                   bounds = bounds, callback=callback) 
    
    return res, optimization_history




#RES description: object of type OptimizeResult, among others returns the 
# final values for the optimized parameters and optimized value of objective function
# doesn't store history information, for this reason nedd to be saved and returned separately


#OPTIMIZATION_HISTORY description: array of Optimization step object
