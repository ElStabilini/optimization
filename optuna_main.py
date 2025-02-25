import numpy as np
import os
import time
import datetime
from argparse import ArgumentParser, Namespace
from pathlib import Path
from qibocal.auto.execute import Executor
from qibocal import update
from qibocal.cli.report import report
from optunaopt_utils import rb_optimization_optuna, log_optimization

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

    start_time = time.time()
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%d_%H%M%S")

    executor_path = (
        Path.cwd().parent / "optimization_data" / f"{target}_{formatted_time}"
    )
    study_name = f"{formatted_time}"
    study_path = Path.cwd().parent / "optuna_data" / f"{target}_{study_name}"
    os.makedirs(os.path.dirname(study_path), exist_ok=True)

    with Executor.open(
        "myexec",
        path=executor_path,
        platform=platform,
        targets=[target],
        update=platform_update,
        force=True,
    ) as e:

        e.platform.settings.nshots = NSHOTS
        ampl_RX = e.platform.qubits[target].native_gates.RX.amplitude
        freq_RX = e.platform.qubits[target].native_gates.RX.frequency
        # eventually add drag parameter

        init_guess = {"amplitude": ampl_RX, "frequency": freq_RX}

        bounds = [[-0.5, 0.5], [freq_RX - 4e6, freq_RX + 4e6]]
        # eventually add bounds for drag parameter

        opt_result = rb_optimization_optuna(
            e,
            target,
            init_guess,
            bounds,
            study_name=study_name,
            storage=f"sqlite:///{study_path}.db",
        )

    report(e.path, e.history)

    end_time = time.time()
    elapsed_time = end_time - start_time

    log_optimization(study_name, elapsed_time, "../optuna_data/time_log.txt")


def main():
    execute(parse())


if __name__ == "__main__":
    main()
