import numpy as np
from qibocal.auto.execute import Executor
from qibolab import pulses
from dataclasses import dataclass
import optuna

DELTA = 20
MAX_DEPTH = 1000
AVG_GATE = 1.875  # 1.875 is the average number of gates in a clifford operation
SEQUENCES = 1000
INIT_STD = 0.25


# objective function to minimize
def objective(trial, e, target, bounds):

    amplitude = trial.suggest_float("amplitude", bounds[0][0], bounds[0][1])
    frequency = trial.suggest_float("frequency", bounds[1][0], bounds[1][1])

    e.platform.qubits[target].native_gates.RX.amplitude = amplitude
    e.platform.qubits[target].native_gates.RX.frequency = frequency

    # eventually add for DRAG pulse optimization
    # pulse = e.platform.qubits[target].native_gates.RX.pulse(start=0)
    # rel_sigma = pulse.shape.rel_sigma
    # drag_pulse = pulses.Drag(rel_sigma=rel_sigma, beta=beta)
    # e.platform.qubits[target].native_gates.RX.shape = repr(drag_pulse)

    rb_output = e.rb_ondevice(
        num_of_sequences=SEQUENCES,
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
    # simulate initial guess (as I do in scipy optimization)
    study.enqueue_trial(init_guess)
    study.optimize(wrapped_objective, n_trials=1000, show_progress_bar=False)

    return study


# 1e-5 va bene come tolleranza? L'errore se non sbaglio dovrebbe essere la deviazione standard
# e doverbbe essere attorno a 1e-4 nei report di Hisham


def log_optimization(process_name, duration, filename):

    with open(filename, "a") as file:
        file.write(f"{process_name}\t{duration}\n")
