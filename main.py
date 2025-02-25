import numpy as np
import os
import time
import datetime
import utils
from argparse import ArgumentParser, Namespace
from pathlib import Path
from qibocal.auto.execute import Executor
from qibocal import update
from qibocal.cli.report import report
from optunaopt_utils import rb_optimization, log_optimization

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
    drag = args.drag

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

        if args.drag is not None:
            drag_output = e.drag_tuning(beta_start=-4, beta_end=4, beta_step=0.5)
            # TODO: add control to guarantee DRAG routine worked correctly

    end_time = time.time()
    elapsed_time = end_time - start_time


def main():
    execute(parse())


if __name__ == "__main__":
    main()
