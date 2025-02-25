import numpy as np
import os
import time
import datetime
from scipyopt_utils import rb_optimization_scipy
from optunaopt_utils import rb_optimization_optuna
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


def main():
    execute(parse())


if __name__ == "__main__":
    main()
