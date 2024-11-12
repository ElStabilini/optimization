import argparse
import datetime
from qibocal.auto.execute import Executor
from qibocal.cli.report import report
from pathlib import Path

AVG_GATE = 1.875
# ramsey, flipping, drag, randomized benchmarking


def main():
    parser = argparse.ArgumentParser(
        description="Fine tuning calibration iteratively running routines"
    )
    parser.add_argument(
        "--platform", type=str, required=True, help="Platform identifier"
    )
    parser.add_argument(
        "--target", type=str, required=True, help="Target qubit to be calibrated"
    )
    parser.add_argument(
        "--platform_update", action="store_true", help="Enable platform update"
    )

    args = parser.parse_args()
    platform = args.platform
    target = args.target
    platform_update = args.platform_update

    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%d_%H%M%S")

    path = Path().parent / "optimization_data" / "sequence" / f"{formatted_time}"

    with Executor.open(
        "myexec",
        path=path,
        platform=platform,
        targets=[target],
        update=platform_update,
        force=True,
    ) as e:

        e.platform.settings.nshots = 1024

        ramsey_output = e.ramsey(
            delay_between_pulses_end=1000,
            delay_between_pulses_start=10,
            delay_between_pulses_step=20,
            detuning=3000000,
            relaxation_time=200000,
        )

        if ramsey_output.results.delta_phys[target][0] > 1e5:
            if ramsey_output.results.chi2[target][0] > 2:
                raise RuntimeError(
                    f"Ramsey fit has chi2 {ramsey_output.results.chi2[target][0]} greater than 2. Stopping."
                )
            else:
                ramsey_output.update_platform(e.platform)

        for i in range(10):
            if i == 0:
                flipping_output = e.flipping(
                    nflips_max=20,
                    nflips_step=1,
                )
            else:
                flipping_output = e.flipping(
                    nflips_max=20,
                    delta_amplitude=3e-4,  # avarage correction for first flipping tests
                    nflips_step=1,
                )
            if flipping_output.results.chi2[target][0] > 2:
                raise RuntimeError(
                    f"Ramsey fit has chi2 {ramsey_output.results.chi2[target][0]} greater than 2. Stopping."
                )
            else:
                flipping_output.update_platform(e.platform)
                err = flipping_output.results.amplitude[target][1]

    report(e.path, e.history)


if __name__ == "__main()__":
    main()
