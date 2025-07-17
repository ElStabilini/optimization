import numpy as np
import optuna
from qibocal.auto.execute import Executor
from qibolab import pulses
from dataclasses import dataclass

# Constants
DELTA = 20
MAX_DEPTH = 1000
AVG_GATE = 1.875  # average number of gates in a Clifford operation
SEQUENCES = 1000
INIT_STD = 0.25

# Error container
error_storage = {"error": None}

@dataclass
class OptimizationStep:
    iteration: int
    parameters: np.ndarray
    objective_value: float
    objective_value_error: float

# Objective function to minimize
def objective(trial, e: Executor, target: str, bounds: list[list[float]]):
    amplitude = trial.suggest_float("amplitude", bounds[0][0], bounds[0][1])
    frequency = trial.suggest_float("frequency", bounds[1][0], bounds[1][1])
    beta = trial.suggest_float("beta", bounds[2][0], bounds[2][1])

    e.platform.qubits[target].native_gates.RX.amplitude = amplitude
    e.platform.qubits[target].native_gates.RX.frequency = frequency

    # Apply DRAG pulse with current beta
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
    trial.set_user_attr("error", r_g_std)

    print("terminating objective call")
    return r_g

def rb_optimization(
    executor: Executor,
    target: str,
    init_guess: dict,
    bounds: list[list[float]],
    study_name: str,
    storage: str,
):
    def wrapped_objective(trial):
        return objective(trial, executor, target, bounds)

    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        storage=storage,
        load_if_exists=False,
    )

    # Enqueue initial guess (must match all three parameters)
    study.enqueue_trial(init_guess)
    study.optimize(wrapped_objective, n_trials=1000, show_progress_bar=False)

    # Convert to optimization history
    optimization_history = []
    for i, trial in enumerate(study.trials):
        try:
            amplitude = trial.params["amplitude"]
            frequency = trial.params["frequency"]
            beta = trial.params["beta"]
            error = trial.user_attrs.get("error", np.nan)
            params = np.array([amplitude, frequency, beta])
            step = OptimizationStep(i, params, trial.value, error)
            optimization_history.append(step)
        except KeyError:
            continue  # Skip failed/incomplete trials

    return study, optimization_history

def log_optimization(process_name: str, duration: float, filename: str):
    with open(filename, "a") as file:
        file.write(f"{process_name}\t{duration}\n")
