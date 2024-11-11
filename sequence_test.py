import numpy as np
from qibocal.auto.execute import Executor
from qibocal.cli.report import report
from pathlib import Path

AVG_GATE = 1.875
# ramsey, flipping, drag, randomized benchmarking

target = "D1"
platform = "qw11q"
path = Path().parent / "optimization_data" / "sequence"

with Executor.open(
    "myexec",
    path=path,
    platform=platform,
    targets=[target],
    update=False,
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
