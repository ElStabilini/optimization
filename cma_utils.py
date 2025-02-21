import numpy as np
import cma
from qibocal.auto.execute import Executor
from qibolab import pulses
from dataclasses import dataclass

DELTA = 10
MAX_DEPTH = 1000
AVG_GATE = 1.875  # 1.875 is the average number of gates in a Clifford operation
SEQUENCES = 1000
INIT_STD = 0.25

error_storage = {"error": None}


@dataclass
class OptimizationStep:
    iteration: int
    parameters: np.ndarray
    objective_value: float
    objective_value_error: float


# Objective function to minimize
def objective(params, e, target):
    amplitude, frequency, beta = params

    e.platform.qubits[target].native_gates.RX.amplitude = amplitude
    e.platform.qubits[target].native_gates.RX.frequency = frequency

    pulse = e.platform.qubits[target].native_gates.RX.pulse(start=0)
    rel_sigma = pulse.shape.rel_sigma
    drag_pulse = pulses.Drag(rel_sigma=rel_sigma, beta=beta)
    e.platform.qubits[target].native_gates.RX.shape = repr(drag_pulse)

    rb_output = e.rb_ondevice(
        num_of_sequences=1000,
        max_circuit_depth=MAX_DEPTH,
        delta_clifford=DELTA,
        n_avg=1,
        save_sequences=True,
        apply_inverse=True,
    )

    # Calculate infidelity and error
    stdevs = np.sqrt(np.diag(np.reshape(rb_output.results.cov[target], (3, 3))))
    pars = rb_output.results.pars.get(target)
    one_minus_p = 1 - pars[2]
    r_c = one_minus_p * (1 - 1 / 2**1)
    r_g = r_c / AVG_GATE
    r_c_std = stdevs[2] * (1 - 1 / 2**1)
    r_g_std = r_c_std / AVG_GATE

    error_storage["error"] = r_g_std

    print("terminating objective call")
    return r_g


def rb_optimization(executor: Executor, target: str, init_guess: list[float], bounds):

    optimization_history = []
    iteration_count = 0

    def record_history(x, f):
        nonlocal iteration_count
        if f is None:
            # If the optimization method doesn't provide f, need to calculate it
            f = objective(x, executor, target)

        step = OptimizationStep(
            iteration=iteration_count,
            parameters=np.copy(x),
            objective_value=f,
            objective_value_error=error_storage["error"],
        )
        optimization_history.append(step)
        iteration_count += 1
        print(f"Completed iteration {iteration_count}, objective value: {f}")

    sigma = INIT_STD  # Standard deviation for initial search
    lower_bounds, upper_bounds = zip(*bounds)

    # Create a CMA-ES optimizer instance
    es = cma.CMAEvolutionStrategy(
        init_guess, sigma, {"bounds": [lower_bounds, upper_bounds], "maxiter": 3}
    )

    # Optimization loop (testing this instead of es.optimize)
    while not es.stop():
        solutions = es.ask()

        # Evaluate the objective function for each solution
        function_values = [objective(sol, executor, target) for sol in solutions]
        es.tell(solutions, function_values)

        # Record history for the best solution of the current iteration
        best_idx = np.argmin(function_values)
        record_history(solutions[best_idx], function_values[best_idx])

    # Retrieve the final result - not strictly necessary but useful to keep track of the history similarly to scipy optimize
    res = {
        "x": es.result.xbest,  # Best solution found
        "fun": es.result.fbest,  # Objective value at the best solution
        "nfev": es.result.evaluations,  # Number of function evaluations
        "nit": es.result.iterations,  # Number of iterations
        "success": es.result.stop,  # Stopping condition
    }

    return res, optimization_history
