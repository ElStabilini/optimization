import numpy as np
from scipy.optimize import minimize
from qibocal.auto.execute import Executor
from qibolab import pulses
from dataclasses import dataclass

AVG_GATE = 1.875 # 1.875 is the average number of gates in a clifford operation

@dataclass
class OptimizationStep:
    iteration: int
    parameters: np.ndarray
    objective_value: float
    objective_value_error = float

#objective function to minimize
def objective(scaled_params, e, target, scale_factors):

    #unscales params
    params = unscale_params(scaled_params, scale_factors)
    amplitude, frequency, beta = params

    e.platform.qubits[target].native_gates.RX.amplitude = amplitude
    e.platform.qubits[target].native_gates.RX.frequency = frequency
    
    pulse = e.platform.qubits[target].native_gates.RX.pulse(start=0)
    rel_sigma = pulse.shape.rel_sigma
    drag_pulse = pulses.Drag(rel_sigma=rel_sigma, beta=beta)
    e.platform.qubits[target].native_gates.RX.shape = repr(drag_pulse)

    rb_output = e.rb_ondevice(
        num_of_sequences=1000,
        max_circuit_depth=1000,
        delta_clifford=10,
        n_avg=1,
        save_sequences=True,
        apply_inverse=True
    )

    # Calculate infidelity
    pars = rb_output.results.pars.get(target)
    one_minus_p = 1 - pars[2]
    r_c = one_minus_p * (1 - 1 / 2**1)
    r_g = r_c / AVG_GATE

    print('terminating objective call')
    return r_g


def rb_optimization(
        executor : Executor,
        target : str,
        method : str,
        init_guess : list[float],
        scale,
        bounds
    ):
    
    optimization_history = []
    iteration_count = 0

    def callback(x, f=None):
        nonlocal iteration_count
        if f is None:
            # If the optimization method doesn't provide f, need to calculate it
            f = objective(x, executor, target, scale)
        
        step = OptimizationStep(
            iteration=iteration_count,
            parameters=np.copy(x),
            objective_value=f
        )
        optimization_history.append(step)
        iteration_count += 1
        print(f"Completed iteration {iteration_count}, objective value: {f}")

    res = minimize(objective, init_guess, args=(executor, target, scale), method=method, 
                   tol=1e-8, options = {"maxiter" : 5}, bounds = bounds, callback=callback)
    
    return res, optimization_history

#RES description: object of type OptimizeResult, among others returns the 
# final values for the optimized parameters and optimized value of objective function
# doesn't store history information, for this reason nedd to be saved and returned separately


#OPTIMIZATION_HISTORY description: array of Optimization step object


def scale_params(params, scale_factors):
    return params / scale_factors

def unscale_params(scaled_params, scale_factors):
    return scaled_params * scale_factors
