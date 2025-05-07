import os
import optuna
import numpy as np
import pandas as pd


def process_opt(folders):
    rows = []

    for folder in folders:
        path = os.path.join("opt_analysis", folder, "optimization_history.npz")
        data = np.load(path)

        iterations = data["iterations"]
        parameters = data["parameters"]
        objective_values = data["objective_values"]

        best_idx = np.argmin(objective_values)

        # Cost values
        cost_initial = objective_values[0]
        cost_best = objective_values[best_idx]
        cost_final = objective_values[-1]

        # Cost improvements
        improvement_best = 100 * (cost_initial - cost_best) / cost_initial
        improvement_final = 100 * (cost_initial - cost_final) / cost_initial

        # Fidelity values
        fidelity_initial = 1 - cost_initial
        fidelity_best = 1 - cost_best
        fidelity_final = 1 - cost_final

        # Fidelity improvements
        fidelity_improvement_best = (
            100 * (fidelity_best - fidelity_initial) / fidelity_initial
        )
        fidelity_improvement_final = (
            100 * (fidelity_final - fidelity_initial) / fidelity_initial
        )

        # Parameters
        A_best = parameters[best_idx, 0]
        f_best = parameters[best_idx, 1]
        A_final = parameters[-1, 0]
        f_final = parameters[-1, 1]

        B_best = B_final = None
        if parameters.shape[1] > 2:
            B_best = parameters[best_idx, 2]
            B_final = parameters[-1, 2]

        row = {
            "Analysis Name": folder,
            "cost_initial": cost_initial,
            "cost_best": cost_best,
            "index_best": best_idx,
            "A best [a.u.]": A_best,
            "f best [Hz]": f_best,
            "B best": B_best,
            "cost_final": cost_final,
            "A final [a.u.]": A_final,
            "f final [Hz]": f_final,
            "B final": B_final,
            "improvement_best [%]": improvement_best,
            "improvement_final [%]": improvement_final,
            "fidelity_initial": fidelity_initial,
            "fidelity_best": fidelity_best,
            "fidelity_final": fidelity_final,
            "fidelity_improvement_best [%]": fidelity_improvement_best,
            "fidelity_improvement_final [%]": fidelity_improvement_final,
        }

        rows.append(row)

    # Build DataFrame
    df = pd.DataFrame(rows)
    df.to_csv("summary_with_improvement.csv", index=False)


def process_optuna_study(db_paths, db_filename="optuna.sb"):

    rows = []

    for db_file in db_paths:
        filename = os.path.basename(db_file)

        # Extract study name from filename (after the first underscore)
        study_name = filename.split("_", 1)[1].replace(".db", "")

        db_path = f"sqlite:///{os.path.abspath(db_file)}"

        study = optuna.load_study(study_name=study_name, storage=db_path)

        trials_df = study.trials_dataframe(attrs=("number", "value", "params", "state"))

        # Filter only completed trials
        trials_df = trials_df[trials_df["state"] == "COMPLETE"]

        if trials_df.empty:
            return None

        # Sort by trial number (to get initial and final)
        trials_df = trials_df.sort_values("number")

        cost_initial = trials_df.iloc[0]["value"]
        cost_final = trials_df.iloc[-1]["value"]
        best_trial = study.best_trial
        cost_best = best_trial.value

        improvement_best = 100 * (cost_initial - cost_best) / cost_initial
        improvement_final = 100 * (cost_initial - cost_final) / cost_initial

        # Fidelity calculations
        fidelity_initial = 1 - cost_initial
        fidelity_best = 1 - cost_best
        fidelity_final = 1 - cost_final

        fidelity_improvement_best = (
            100 * (fidelity_best - fidelity_initial) / fidelity_initial
        )
        fidelity_improvement_final = (
            100 * (fidelity_final - fidelity_initial) / fidelity_initial
        )

        # Parameters
        A_best = best_trial.params.get("amplitude")
        f_best = best_trial.params.get("frequency")
        B_best = best_trial.params.get("beta", None)

        A_final = trials_df.iloc[-1]["params_amplitude"]
        f_final = trials_df.iloc[-1]["params_frequency"]
        B_final = trials_df.iloc[-1].get("params_beta", None)

        row = {
            "Analysis Name": study,
            "cost_initial": cost_initial,
            "cost_best": cost_best,
            "index_best": best_trial.number,
            "A best [a.u.]": A_best,
            "f best [Hz]": f_best,
            "B best": B_best,
            "cost_final": cost_final,
            "A final [a.u.]": A_final,
            "f final [Hz]": f_final,
            "B final": B_final,
            "improvement_best [%]": improvement_best,
            "improvement_final [%]": improvement_final,
            "fidelity_initial": fidelity_initial,
            "fidelity_best": fidelity_best,
            "fidelity_final": fidelity_final,
            "fidelity_improvement_best [%]": fidelity_improvement_best,
            "fidelity_improvement_final [%]": fidelity_improvement_final,
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv("optuna_summary.csv", index=False)


if __name__ == "__main__":

    print(os.getcwd())

    folders = [
        "D1_cma_post_ft_true",
        "D1_init_simplex_20241110_211211",
        "D1_nelder-mead_post_ft_true",
        "D1_init_simplex_20241113_181711",
        "D1_init_simplex_20241113_200745",
        "D1_SLSQP_post_ft_true",
    ]

    db_paths = [
        "../optuna_data/D1_20241110_074214.db",
        "../optuna_data/D1_20241118_151919.db",
        "../optuna_data/D1_20241109_114242.db",
        "../optuna_data/D1_20241121_192626.db",
    ]

    process_opt(folders)
    process_optuna_study(db_paths)
