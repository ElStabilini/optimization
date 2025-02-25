import numpy as np
import cma
from qibocal.auto.execute import Executor
from qibolab import pulses
from dataclasses import dataclass

DELTA = 10
MAX_DEPTH = 1000
AVG_GATE = 1.875  # 1.875 is the average number of gates in a Clifford operation
SEQUENCES = 1000
INIT_STD = 0.25

error_storage = {"error": None}


@dataclass
class OptimizationStep:
    iteration: int
    parameters: np.ndarray
    objective_value: float
    objective_value_error: float


def objective(params, e, target):
    amplitude, frequency, beta = params

    e.platform.qubits[target].native_gates.RX.amplitude = amplitude
    e.platform.qubits[target].native_gates.RX.frequency = frequency

    pulse = e.platform.qubits[target].native_gates.RX.pulse(start=0)
    rel_sigma = pulse.shape.rel_sigma
    drag_pulse = pulses.Drag(rel_sigma=rel_sigma, beta=beta)
    e.platform.qubits[target].native_gates.RX.shape = repr(drag_pulse)

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

    error_storage["error"] = r_g_std

    print("terminating objective call")
    return r_g
