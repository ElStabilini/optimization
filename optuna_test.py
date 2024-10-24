import numpy as np
import os
import time
import pickle
import optuna
from qibocal.auto.execute import Executor
from qibocal.cli.report import report

start_time = time.time()

target = "D1"
platform = "qw11q"

executor_path = f'./optimization_data/{target}_Optuna_post_ft'
opt_history_path = f'./opt_analysis/{target}_Optuna_post_ft'

# Define the optimization function
def objective(trial):
    with Executor.open(
        "myexec",
        path=executor_path,
        platform=platform,
        targets=[target],
        update=True,
        force=True,
    ) as e:
        e.platform.settings.nshots = 2000

        # Drag tuning
        drag_output = e.drag_tuning(
             beta_start=-4,
             beta_end=4,
             beta_step=0.5
        )
        beta_best = drag_output.results.betas[target]
        ampl_RX = e.platform.qubits[target].native_gates.RX.amplitude
        freq_RX = e.platform.qubits[target].native_gates.RX.frequency

        # Suggest parameters using Optuna
        ampl_suggested = trial.suggest_uniform('ampl_RX', -0.5, 0.5)
        freq_suggested = trial.suggest_uniform('freq_RX', freq_RX - 4e6, freq_RX + 4e6)
        beta_suggested = trial.suggest_uniform('beta_best', beta_best - 0.25, beta_best + 0.25)

        # Perform RB optimization (replace rb_optimization with the actual function)
        results, history = rb_optimization(
            e, 
            target, 
            init_guess=np.array([ampl_suggested, freq_suggested, beta_suggested]),
        )

        # Return the objective value to be minimized
        return results.objective_value

# Optuna study setup
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Save study results
os.makedirs(opt_history_path, exist_ok=True)
with open(os.path.join(opt_history_path, 'optuna_study.pkl'), 'wb') as f:
    pickle.dump(study, f)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")

# Report generation
report(executor_path, e.history)
