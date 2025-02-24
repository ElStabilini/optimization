import numpy as np
import os
import time
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from qibocal.auto.execute import Executor
from qibocal import update
from qibocal.cli.report import report
from rb_init_simplex import rb_optimization
from scipy.optimize import Bounds


NSHOTS = 2000


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
    parser.add_argument("--method", type=str, required=True, help="Optimization method")
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
    update.drag_pulse_beta(beta, platform, target)


def execute(args: Namespace):

    platform = args.platform
    target = args.target
    platform_update = args.platform_update
    method = args.method

    executor_path = (
        Path.cwd().parent / "optimization_data" / f"{target}_{method}_post_ft_true"
    )
    opt_history_path = (
        Path.cwd().parent / "opt_analysis" / f"{target}_{method}_post_ft_true"
    )

    start_time = time.time()

    with Executor.open(
        "myexec",
        path=executor_path,
        platform=platform,
        targets=[target],
        update=platform_update,
        force=True,
    ) as e:

        # check for missing init_symplex
        e.platform.settings.nshots = NSHOTS
        drag_output = e.drag_tuning(beta_start=-4, beta_end=4, beta_step=0.5)

        beta_best = drag_output.results.betas[target]
        ampl_RX = e.platform.qubits[target].native_gates.RX.amplitude
        freq_RX = e.platform.qubits[target].native_gates.RX.frequency

        init_guess = np.array([ampl_RX, freq_RX, beta_best])

        lower_bounds = np.array([-0.5, freq_RX - 4e6, beta_best - 0.25])
        upper_bounds = np.array([0.5, freq_RX + 4e6, beta_best + 0.25])
        bounds = Bounds(lower_bounds, upper_bounds)

        opt_results, optimization_history = rb_optimization(
            e,
            target,
            method,
            init_guess,
            bounds,
        )

    report(e.path, e.history)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # save optimization_history as .npz
    iterations = np.array([step.iteration for step in optimization_history])
    parameters = np.array([step.parameters for step in optimization_history])
    objective_values = np.array([step.objective_value for step in optimization_history])
    objective_value_error = np.array(
        [step.objective_value_error for step in optimization_history]
    )

    Path(opt_history_path).mkdir(parents=True, exist_ok=True)
    np.savez(
        Path(opt_history_path) / "optimization_history.npz",
        iterations=iterations,
        parameters=parameters,
        objective_values=objective_values,
        objective_value_errors=objective_value_error,
    )

    data_stored = {"opt_results": opt_results, "elapsed_time": elapsed_time}

    with open(os.path.join(opt_history_path, "optimization_result.pkl"), "wb") as f:
        pickle.dump(data_stored, f)

    # Update platform to best parameters result
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


def main():
    execute(parse())


if __name__ == "__main__":
    main()
