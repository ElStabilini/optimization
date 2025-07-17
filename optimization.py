import numpy as np
import os
import time
import datetime
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from qibocal.auto.execute import Executor
from qibocal import update
from qibocal.cli.report import report
from scipy.optimize import Bounds

# Optimization utilities per method
from cma_utils import rb_optimization as cma_rb_optimization
from optunaopt_utils import rb_optimization as optuna_rb_optimization, log_optimization
from scipyopt_utils import rb_optimization as scipy_rb_optimization

NSHOTS = 2000

def parse() -> Namespace:
    parser = ArgumentParser(description="Fine tuning calibration with optimization backends")
    parser.add_argument("--platform", type=str, required=True, help="Platform identifier")
    parser.add_argument("--target", type=str, required=True, help="Target qubit to be calibrated")
    parser.add_argument("--platform_update", action="store_true", help="Enable platform update")
    parser.add_argument("--method", type=str, required=True, choices=["cma", "optuna", "scipy"],
                        help="Optimization method to use")
    return parser.parse_args()

def update_platform(args: Namespace, params: list[float]):
    amplitude, frequency, beta = params
    update.drive_amplitude(amplitude, args.platform, args.target)
    update.drive_frequency(frequency, args.platform, args.target)
    update.drag_pulse_beta(beta, args.platform, args.target)

def save_optimization_history(opt_history_path, optimization_history):
    iterations = np.array([step.iteration for step in optimization_history])
    parameters = np.array([step.parameters for step in optimization_history])
    objective_values = np.array([step.objective_value for step in optimization_history])
    objective_value_error = np.array([step.objective_value_error for step in optimization_history])

    Path(opt_history_path).mkdir(parents=True, exist_ok=True)
    np.savez(
        Path(opt_history_path) / "optimization_history.npz",
        iterations=iterations,
        parameters=parameters,
        objective_values=objective_values,
        objective_value_errors=objective_value_error,
    )

    return parameters, objective_values, objective_value_error

def execute(args: Namespace):
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%d_%H%M%S")

    target = args.target
    method = args.method
    platform_update = args.platform_update

    executor_path = Path.cwd().parent / "optimization_data" / f"{target}_{method}_{formatted_time}"
    opt_history_path = Path.cwd().parent / "opt_analysis" / f"{target}_{method}_{formatted_time}"

    if method == "optuna":
        study_name = formatted_time
        study_path = Path.cwd().parent / "optuna_data" / f"{target}_{study_name}"
        os.makedirs(study_path.parent, exist_ok=True)

    start_time = time.time()

    with Executor.open(
        "myexec",
        path=executor_path,
        platform=args.platform,
        targets=[target],
        update=platform_update,
        force=True,
    ) as e:

        e.platform.settings.nshots = NSHOTS
        drag_output = e.drag_tuning(beta_start=-4, beta_end=4, beta_step=0.5)

        beta_best = drag_output.results.betas[target]
        ampl_RX = e.platform.qubits[target].native_gates.RX.amplitude
        freq_RX = e.platform.qubits[target].native_gates.RX.frequency

        if method in ["cma", "scipy"]:
            init_guess = np.array([ampl_RX, freq_RX, beta_best])
            lower_bounds = np.array([-0.5, freq_RX - 4e6, beta_best - 0.25])
            upper_bounds = np.array([0.5, freq_RX + 4e6, beta_best + 0.25])
            bounds = Bounds(lower_bounds, upper_bounds) if method == "scipy" else zip(lower_bounds, upper_bounds)
        elif method == "optuna":
            init_guess = {"amplitude": ampl_RX, "frequency": freq_RX}
            bounds = [[-0.5, 0.5], [freq_RX - 4e6, freq_RX + 4e6]]

        if method == "cma":
            opt_results, optimization_history = cma_rb_optimization(e, target, init_guess, bounds)
        elif method == "scipy":
            opt_results, optimization_history = scipy_rb_optimization(e, target, method, init_guess, bounds)
        elif method == "optuna":
            opt_results = optuna_rb_optimization(
                e, target, init_guess, bounds,
                study_name=study_name,
                storage=f"sqlite:///{study_path}.db",
            )
            optimization_history = opt_results.optimization_history  # assuming consistent format

    report(e.path, e.history)
    elapsed_time = time.time() - start_time

    if method != "optuna":  # Only Optuna handles logs differently
        parameters, objective_values, objective_value_error = save_optimization_history(opt_history_path, optimization_history)
        data_stored = {"opt_results": opt_results, "elapsed_time": elapsed_time}
        with open(opt_history_path / "optimization_result.pkl", "wb") as f:
            pickle.dump(data_stored, f)

        fidelities = 1 - objective_values
        sorted_indices_desc = np.argsort(fidelities)[::-1]
        try:
            idx = np.flatnonzero(fidelities[sorted_indices_desc] + objective_value_error[sorted_indices_desc] > 1)[0]
            update_platform(args, parameters[sorted_indices_desc][idx])
        except IndexError:
            pass
    else:
        log_optimization(study_name, elapsed_time, "../optuna_data/time_log.txt")

def main():
    execute(parse())

if __name__ == "__main__":
    main()
