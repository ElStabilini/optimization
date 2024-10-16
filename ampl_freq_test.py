from qibocal.auto.execute import Executor


target = "D1"
platform = "qw11q"
method = 'BFGS' 

executor_path = f'./optimization_data/{target}_{method}'
opt_history_path = f'./opt_analysis/{target}_{method}'
 
with Executor.open(
    "myexec",
    path=executor_path,
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
    ampl_RX = e.platform.qubits[target].native_gates.RX.amplitude #4.1570229140026074e-2
    freq_RX = e.platform.qubits[target].native_gates.RX.frequence

    print(ampl_RX)
    print(freq_RX)