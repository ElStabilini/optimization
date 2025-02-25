import numpy as np
import os
import time
import datetime
import pickle

from scipyopt_utils import rb_optimization_scipy, OptimizationStep
from optunaopt_utils import rb_optimization_optuna, log_optimization
from cma_utils import rb_optimization_cma
from argparse import ArgumentParser, Namespace
from pathlib import Path
from qibocal.auto.execute import Executor
from qibocal import update
from qibocal.cli.report import report
from scipy.optimize import Bounds

# from optunaopt_utils import rb_optimization, log_optimization

NSHOTS = 2000
"number of measurements"
ALLOWED_METHODS = ["Nelder-Mad", "SLSQP", "Newton-CG", "cma", "optuna"]
"list of methods for which the correct functioning of the code is guaranteed"


def parse() -> Namespace:
    parser = ArgumentParser(description="Fine tuning calibration using cma algorithm")
    parser.add_argument(
        "--platform", type=str, required=True, help="Platform identifier"
    )
    parser.add_argument(
        "--target", type=str, required=True, help="Target qubit to be calibrated"
    )
    parser.add_argument(
        "--platform_update", action="store_true", help="Enable platform update"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=ALLOWED_METHODS,
        required=True,
        help=f"Optimization method to use. Allowed values: {', '.join(ALLOWED_METHODS)}",
    )
    parser.add_argument(
        "--drag",
        type=str,
        required=False,
        help="Perform optimization also on the beta parameter of the DRAG pulse",
    )
    parser.add_argument(
        "--init",
        action="store_true",
        help="Perform Nelder-Mead optimization with initial_symplex",
    )

    args, _ = parser.parse_known_args()

    if args.method == "Nelder-Mead" and args.init is None:
        parser.error("--init is required when --method is 'Nelder-Mead'")

    return parser.parse_args()


def update_platform(
    args: Namespace,
    params: list[float],
):
    platform = args.platform
    target = args.target
    amplitude, frequency, beta = params
    update.drive_amplitude(amplitude, platform, target)
    update.drive_frequency(frequency, platform, target)

    if args.drag is not None:
        update.drag_pulse_beta(beta, platform, target)


def history_to_lists(optimization_history: list[OptimizationStep]):

    iterations = np.array([step.iteration for step in optimization_history])
    parameters = np.array([step.parameters for step in optimization_history])
    objective_values = np.array([step.objective_value for step in optimization_history])
    objective_value_error = np.array(
        [step.objective_value_error for step in optimization_history]
    )

    return iterations, parameters, objective_values, objective_value_error


def update(
    objective_values: list[float],
    objective_value_error: list[float],
    parameters: list[np.ndarray],
):

    fidelities = 1 - objective_values
    sorted_indices_desc = np.argsort(fidelities)[::-1]

    sorted_fidelities = fidelities[sorted_indices_desc]
    sorted_errors = objective_value_error[sorted_indices_desc]
    sorted_parameters = parameters[sorted_indices_desc]

    try:
        idx = np.flatnonzero(sorted_fidelities + sorted_errors > 1)[0]
        update_platform(Namespace, sorted_parameters[idx])
    except IndexError:
        pass


# save optimization_history as .npz
def save_optimization_history(
    iterations: list[int],
    parameters: list[np.ndarray],
    objective_values: list[float],
    objective_value_error: list[float],
    path: str,
):

    Path(path).mkdir(parents=True, exist_ok=True)
    np.savez(
        Path(path) / "optimization_history.npz",
        iterations=iterations,
        parameters=parameters,
        objective_values=objective_values,
        objective_value_errors=objective_value_error,
    )


def execute(args: Namespace):

    platform = args.platform
    target = args.target
    platform_update = args.platform_update
    method = args.method

    start_time = time.time()
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%d_%H%M%S")

    executor_path = (
        Path.cwd().parent / "test_data" / f"{target}_{method}_{formatted_time}"
    )
    opt_history_path = (
        Path.cwd().parent / "test_analysis" / f"{target}_{method}_{formatted_time}"
    )

    with Executor.open(
        "myexec",
        path=executor_path,
        platform=platform,
        targets=[target],
        update=platform_update,
        force=True,
    ) as e:

        e.platform.settings.nshots = NSHOTS

        # define values for initial conditions (will have a ddiferent format depending on optimization method)
        if args.drag is not None:
            drag = True
            drag_output = e.drag_tuning(beta_start=-4, beta_end=4, beta_step=0.5)
            beta_best = drag_output.results.betas[target]
            beta_low = beta_best - 0.25
            beta_high = beta_best + 0.25
            # TODO: add control to guarantee DRAG routine worked correctly

        ampl_RX = e.platform.qubits[target].native_gates.RX.amplitude
        freq_RX = e.platform.qubits[target].native_gates.RX.frequency

        # define values for bounds (will have different format depending on optimization method)
        ampl_low = -0.5
        ampl_high = 0.5
        freq_low = freq_RX - 4e6
        freq_high = freq_RX + 4e6
        lower_bounds = np.array([ampl_low, freq_low])
        upper_bounds = np.array([ampl_high, freq_high])
        # TODO: add option for DRAG parameter optimization

        if method == "optuna":
            study_name = f"{formatted_time}"
            study_path = (
                Path.cwd().parent / "optuna_data" / f"{target}_{formatted_time}"
            )
            os.makedirs(os.path.dirname(study_path), exist_ok=True)
            init_guess = {"amplitude": ampl_RX, "frequency": freq_RX}
            bounds = [[ampl_low, ampl_high], [freq_low, freq_high]]

            opt_results = rb_optimization_optuna(
                e,
                target,
                init_guess,
                bounds,
                study_name,
                storage=f"sqlite:///{study_path}.db",
            )

        if method == "cma":
            bounds = zip(lower_bounds, upper_bounds)
            opt_results, optimization_history = rb_optimization_cma(
                e, target, init_guess, bounds
            )

        else:
            bounds = Bounds(lower_bounds, upper_bounds)
            opt_results, optimization_history = rb_optimization_scipy(
                e, target, method, init_guess, bounds
            )

    report(e.path, e.history)

    end_time = time.time()
    elapsed_time = end_time - start_time

    if method == "optuna":
        log_optimization(
            study_name,
            elapsed_time,
            Path.cwd().parent / "optuna_data" / "time_log.txt",
        )

    else:
        data_stored = {"opt_results": opt_results, "elapsed_time": elapsed_time}

        with open(os.path.join(opt_history_path, "optimization_result.pkl"), "wb") as f:
            pickle.dump(data_stored, f)

        iterations, parameters, objective_values, objective_value_error = (
            history_to_lists(optimization_history)
        )

        save_optimization_history(
            iterations,
            parameters,
            objective_values,
            objective_value_error,
            opt_history_path,
        )

        update(objective_values, objective_value_error, parameters)


def main():
    execute(parse())


if __name__ == "__main__":
    main()
