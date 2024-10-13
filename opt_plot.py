import numpy as np
import matplotlib.pyplot as plt
import os
import json


WIDTH  = 0.5
FONT = 24

BLUE = "#788ae3"
RED = "#d1545e"
DARKBLUE = "#075cb3"
DARKRED = "#8c1822"



"""UNZIP ROUTINE"""
def unzip_data_npz(root_dir, destination_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if 'data.npz' in filenames:
            npz_path = os.path.join(dirpath, 'data.npz')
            relative_path = os.path.relpath(dirpath, root_dir)
            data_dir = os.path.join(destination_dir, relative_path, 'data')

            if not os.path.exists(data_dir):
                try:
                    with np.load(npz_path, allow_pickle=True) as data:
                        os.makedirs(data_dir, exist_ok=True)
                        for key, value in data.items():
                            np.save(os.path.join(data_dir, f"{key}.npy"), value)
                except Exception as e:
                    print(f"Error unzipping {npz_path}: {str(e)}")
#NB: here calling current working directory

target = 'D1'
method = 'nelder-mead'

root_directory = f'optimization_{target}_{method}'
unzip_directory = f'../rb_opt_unzipped/{target}_{method}/'
unzip_data_npz(root_directory, unzip_directory)





"""CODICE PER PLOT da ROUTINES"""
#Access unzipped files
#NB: da modificare perchè qui sto accedendo a tutti i dati non solo a quelli dello step di iterazione

path = os.path.join(unzip_directory, 'data/')

rb_paths = [item for item in os.listdir(path) if item.startswith('rb_ondevice') 
            and os.path.isdir(os.path.join(path, item))]
sorted_rb_items = sorted(rb_paths, key=lambda x: int(x.split('-')[1]))
print(len(sorted_rb_items))

all_decay_params, all_decay_errs = [], []
for rbpath in sorted_rb_items:
    with open(f"{path}{rbpath}/results.json") as f:
        data_dict = json.load(f)
    popt = data_dict['"pars"']['"D1"']
    pcov = np.array(data_dict['"cov"']['"D1"']).reshape(3,3)
    perr = np.sqrt(np.diag(pcov))
    all_decay_params.append(popt[2])
    all_decay_errs.append(perr[2])

all_decay_params, all_decay_errs = np.array(all_decay_params), np.array(all_decay_errs)
print(all_decay_params, all_decay_errs)

infidelity = (1 - all_decay_params) / 2
pulse_fidelities = 1 - infidelity / 1.875

pulse_fidelities_err = all_decay_errs / (1.875 * 2)


x = np.arange(1, len(pulse_fidelities) + 1)

fig, ax1 = plt.subplots(1, figsize=(WIDTH * 10 , WIDTH * 10 * 6/8))
ax1.errorbar(x, pulse_fidelities, yerr = pulse_fidelities_err, fmt='o',linestyle='none', markersize=4, color="black", markerfacecolor=RED, ecolor="black", markeredgewidth=0.7, elinewidth=0.7, alpha=1, label=r"qubit 1", capsize=3)
ax1.plot(x, pulse_fidelities, '-', color='black', linewidth=0.7)
ax1.set_xlabel('# optimization steps',  fontsize=12)
#ax1.set_ylabel(r'$\pi/2$ fidelity',  fontsize=12)
plt.savefig(f'plot', bbox_inches="tight", dpi=600)



""" CODICE PER PLOT DA OPTIMIZATION HISTORY """