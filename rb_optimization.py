import numpy as np
from qibolab.qubits import QubitId
from scipy.optimize import minimize
from qibocal.auto.execute import Executor

#objective function to minimize
def objective(params, e, target, scales):

    #unscales params
    params = params * scales     
    amplitude, frequency, beta = params

    e.platform.qubits[target].native_gates.RX.amplitude = amplitude
    e.platform.qubits[target].native_gates.RX.frequency = frequency

    rb_output = e.rb_ondevice(
        num_of_sequences=1000,
        max_circuit_depth=1000,
        delta_clifford=10,
        n_avg=1,
        save_sequences=True,
        apply_inverse=True
    )

    # Calculate infidelity
    cov = rb_output.results.cov
    pars = rb_output.results.pars.get(target)
    stdevs = np.sqrt(np.diag(np.reshape(cov[target], (3, 3))))
    one_minus_p = 1 - pars[2]
    r_c = one_minus_p * (1 - 1 / 2**1)
    r_g = r_c / 1.875  # 1.875 is the average number of gates in a clifford operation
    r_c_std = stdevs[2] * (1 - 1 / 2**1)
    r_g_std = r_c_std / 1.875

    return r_g, r_g_std


def test_rb_optimization(
        executor : Executor,
        target : str,
        method : str,
        init_guess : list[float],
        scale,
        bounds
    ):
    
    res = minimize(objective, init_guess, args=(executor, target, scale), method=method, tol=1e-3, maxiter = 1, bounds = bounds)
    
    return res

def scale_params(params : list[float]):

    return params


#Esecuzione della rountine, magari spostare in un altro script

target = "D1"
platform = "qw11q"
method = 'nelder-mead' #forse non la migliore? Non ho idea del landscape

with Executor.open(
    "myexec",
    path="test_rb",
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

    beta_best = drag_output.results.betas[target]
    
    #per ora in questo step faccio tutto manualmente ma meglio sistemare diversamente
    init_guess = [4.1570229140026074, 4.958263653, beta_best] 
    bounds = [(None,None),(None,None), (beta_best-0.5, beta_best+0.5)]
    scale = np.array([100, 1e-9, 1])
    

    test_rb_optimization(e, target, method, init_guess, scale, bounds)


"""TO DO: 
    y maxiter
    y xatol: normalizzazione + definizione
    * leggere l'initial guess dalla cartella platform precedente + automatizzare riscalamento
    y nshot
    y modificare delta_clifford
    y provare a variare beta in un piccolo intervallo intorno a quello suggerito da drag
    y spostare report ?
    * vedere quali di questi parametri potrebbe essere interessante variare    

"RX": (D1) {
                    "duration": 40,
                    "amplitude": 0.05,
                    "shape": "Gaussian(5)",
                    "frequency": 4900000000,
                    "relative_start": 0,
                    "phase": 0,
                    "type": "qd"
"""



