import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
RESULTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\results'
PLOTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\plots'
lon_file = os.path.join(RESULTS_DIR, 'structure_functions_longitude_2005.csv')
lat_file = os.path.join(RESULTS_DIR, 'structure_functions_latitude_2005.csv')
lon = pd.read_csv(lon_file)
lat = pd.read_csv(lat_file)
Q_VALUES = [1, 2, 3, 4, 5]
plt.figure(figsize=(10, 7))
for q in Q_VALUES:
    plt.plot(np.log10(lon['distance_km']), np.log10(lon[f'S{q}']), marker='o', markersize=3, label=f'q = {q}')
plt.xlabel('$\\log_{10}(r)$')
plt.ylabel('$\\log_{10}[S_q(r)]$')
plt.title('Longitudinal Structure Functions — Log-Log View')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'structure_functions_longitude_loglog_2005.png'), dpi=300)
plt.close()
plt.figure(figsize=(10, 7))
for q in Q_VALUES:
    plt.plot(np.log10(lat['distance_km']), np.log10(lat[f'S{q}']), marker='o', markersize=3, label=f'q = {q}')
plt.xlabel('$\\log_{10}(r)$')
plt.ylabel('$\\log_{10}[S_q(r)]$')
plt.title('Latitudinal Structure Functions — Log-Log View')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'structure_functions_latitude_loglog_2005.png'), dpi=300)
plt.close()
