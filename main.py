import numpy as np
import os
import time
import datetime
from argparse import ArgumentParser, Namespace
from pathlib import Path
from qibocal.auto.execute import Executor
from qibocal import update
from qibocal.cli.report import report
from optunaopt_utils import rb_optimization, log_optimization

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
    parser.add_argument(
        "--drag",
        type=str,
        required=False,
        help="Perform optimization also on the beta parameter of the DRAG pulse",
    )
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

        e.platform.settings.nshots = NSHOTS


def main():
    execute(parse())


if __name__ == "__main__":
    main()
