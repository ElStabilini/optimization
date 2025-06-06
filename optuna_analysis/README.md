# Optimization log 

This file contains the log for optimization using optuna:
* D1_20241106_082639.db: testing optuna code (only 3 iterations)
* D1_20241106_204312.db: testing optuna code with [platform](https://github.com/ElStabilini/qw11q_calibration/blob/main/runcard_cal/recalD1_061124/firsttry/classification/new_platform/parameters.json), report was deleted by accident
* D1_20241109_114242.db : optimization using optuna performed with [platform](https://github.com/ElStabilini/qw11q_calibration/blob/main/runcard_cal/ft_091124/classification/new_platform/parameters.json)
* D1_20241110_074214.db : optimization using optuna performed with [platform](https://github.com/ElStabilini/qw11q_calibration/blob/main/runcard_cal/ft_091124/classification/new_platform/parameters.json), 100 iterations
* <u>drag was performed also separately before running the optimization</u>

* optuna with verified correct DRAG executed before 
* optuna without optimizing the $\beta$ parameter on the DRAG (only frequency and amplitude were optimized)

NB: `study_name` for `optuna` studies is always `formatted_time = now.strftime("%Y%m%d_%H%M%S")`
