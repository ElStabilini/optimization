import numpy as np
from scipy.optimize import minimize, Bounds
from qibocal.auto.execute import Executor
from qibolab import pulses
from dataclasses import dataclass

# Constants
DELTA = 10
MAX_DEPTH = 1000
AVG_GATE = 1.875
SEQUENCES = 1000
INIT_STD = 0.25

error_storage = {"error": None}

@dataclass
class OptimizationStep:
    iteration: int
    parameters: np.ndarray
    objective_value: float
    objective_value_error: float

def objective(params, e: Executor, target: str):
    amplitude, frequency, beta = params

    e.platform.qubits[target].native_gates.RX.amplitude = amplitude
    e.platform.qubits[target].native_gates.RX.frequency = frequency

    # Apply DRAG pulse
    pulse = e.platform.qubits[target].native_gates.RX.pulse(start=0)
    rel_sigma = pulse.shape.rel_sigma
    drag_pulse = pulses.Drag(rel_sigma=rel_sigma, beta=beta)
    e.platform.qubits[target].native_gates.RX.shape = repr(drag_pulse)

    rb_output = e.rb_ondevice(
        num_of_sequences=SEQUENCES,
        max_circuit_depth=MAX_DEPTH,
        delta_clifford=DELTA,
        n_avg=1,
        save_sequences=True,
        apply_inverse=True,
    )

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

def rb_optimization(
    executor: Executor,
    target: str,
    method: str,
    init_guess: list[float],
    bounds: Bounds,
    initial_simplex: list[list[float]] = None,
):
    optimization_history = []
    iteration_count = 0

    def callback(x, f=None):
        nonlocal iteration_count
        if f is None:
            f = objective(x, executor, target)

        step = OptimizationStep(
            iteration=iteration_count,
            parameters=np.copy(x),
            objective_value=f,
            objective_value_error=error_storage["error"],
        )
        optimization_history.append(step)
        print(f"Completed iteration {iteration_count}, objective value: {f}")
        iteration_count += 1

    # Construct simplex if not provided
    if initial_simplex is None:
        identity = np.eye(len(init_guess))
        initial_simplex = [init_guess] + [init_guess + 0.01 * row for row in identity]

    res = minimize(
        objective,
        init_guess,
        args=(executor, target),
        method=method,
        tol=1e-4,
        options={"maxiter": 40, "initial_simplex": initial_simplex},
        bounds=bounds,
        callback=callback,
    )

    return res, optimization_history
