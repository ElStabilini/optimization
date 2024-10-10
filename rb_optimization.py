import numpy as np
import matplotlib.pyplot as plt
from qibolab.qubits import QubitId
from scipy.optimize import minimize
from qibocal.auto.execute import Executor
from qibocal.cli.report import report
from qibolab import pulses

AVG_GATE = 1.875 # 1.875 is the average number of gates in a clifford operation

#objective function to minimize
def objective(scaled_params, e, target, scale_factors):

    #unscales params
    params = unscale_params(scaled_params, scale_factors)
    amplitude, frequency, beta = params

    e.platform.qubits[target].native_gates.RX.amplitude = amplitude
    e.platform.qubits[target].native_gates.RX.frequency = frequency
    
    pulse = e.platform.qubits[target].native_gates.RX.pulse(start=0)
    rel_sigma = pulse.shape.rel_sigma
    drag_pulse = pulses.Drag(rel_sigma=rel_sigma, beta=beta)
    e.platform.qubits[target].native_gates.RX.shape = repr(drag_pulse)

    rb_output = e.rb_ondevice(
        num_of_sequences=1000,
        max_circuit_depth=1000,
        delta_clifford=10,
        n_avg=1,
        save_sequences=True,
        apply_inverse=True
    )

    # Calculate infidelity
    pars = rb_output.results.pars.get(target)
    one_minus_p = 1 - pars[2]
    r_c = one_minus_p * (1 - 1 / 2**1)
    r_g = r_c / AVG_GATE

    #l'errore in teoria viene salvato in automatico dalla procedura di RB  
    #cov = rb_output.results.cov
    #stdevs = np.sqrt(np.diag(np.reshape(cov[target], (3, 3))))
    #r_c_std = stdevs[2] * (1 - 1 / 2**1)
    #r_g_std = r_c_std / AVG_GATE

    return r_g


def rb_optimization(
        executor : Executor,
        target : str,
        method : str,
        init_guess : list[float],
        scale,
        bounds
    ):
    
    obj_values = []

    def callback(x):
        obj_val = objective(x, executor, target, scale)
        obj_values.append(obj_val)

    #wrapped_callback = lambda x: callback(x, executor, target, scale, obj_values)

    res = minimize(objective, init_guess, args=(executor, target, scale), method=method, 
                   tol=1e-8, options = {"maxiter" : 100}, bounds = bounds, callback=callback)
    
    return res, obj_values



def scale_params(params, scale_factors):
    return params / scale_factors

def unscale_params(scaled_params, scale_factors):
    return scaled_params * scale_factors



#Esecuzione della rountine, magari spostare in un altro script
target = "D1"
platform = "qw11q"
method = 'nelder-mead' #forse non la migliore? Non ho idea del landscape

with Executor.open(
    "myexec",
    path="rb_opt",
    platform=platform,
    targets=[target],
    update=True,
    force=True,
) as e:
 
    e.platform.settings.nshots = 2000
    drag_output = e.drag_tuning(
         beta_start = -4,
         beta_end = 4,
         beta_step = 0.5
    )

    #per ora in questo step faccio tutto manualmente ma meglio sistemare diversamente

    beta_best = drag_output.results.betas[target]
    ampl_RX = 4.1570229140026074e-2
    freq_RX = 4.958263653e9
    
    scale_factors = np.array([1e-2, 1e-9, 1])
    init_guess = np.array([ampl_RX, freq_RX, beta_best])
    scaled_init_guess = scale_params(init_guess, scale_factors)

    lower_bounds = np.array([-0.5, freq_RX-4e6, beta_best-0.25])
    upper_bounds = np.array([0.5, freq_RX+4e6, beta_best+0.25])
    scaled_bounds = list(zip(scale_params(lower_bounds, scale_factors),
                         scale_params(upper_bounds, scale_factors)))

    optimization, optimization_history = rb_optimization(e, target, method, scaled_init_guess, scale_factors, scaled_bounds)

report(e.path, e.history)

plt.plot(optimization_history, label="Objective Function Value")
plt.xlabel("Iteration")
plt.ylabel("Objective Function Value")
plt.title("Objective Function Value vs. Iteration")
plt.legend()
plt.grid(True)
plt.show()


"""TO DO:
    y controllare dove uso scale_factors
    y capire come gestire l'errore    
    y passare 1.875 (avg_gate) come costante all'inizio
    y maxiter
    y xatol: normalizzazione + definizione
    * leggere l'initial guess dalla cartella platform precedente + automatizzare riscalamento
    y nshot
    y modificare delta_clifford
    y provare a variare beta in un piccolo intervallo intorno a quello suggerito da drag
    y spostare report ?
    * sistemare plot con plotly e analisi dati
    y vedere quali di questi parametri potrebbe essere interessante variare

"RX": (D1) {
                    "duration": 40,
                    "amplitude": 0.05,
                    "shape": "Gaussian(5)",
                    "frequency": 4900000000,
                    "relative_start": 0,
                    "phase": 0,
                    "type": "qd"
""" 