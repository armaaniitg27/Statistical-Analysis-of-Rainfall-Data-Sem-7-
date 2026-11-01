import os
import pandas as pd
import matplotlib.pyplot as plt
RESULTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\results'
PLOTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\plots'
os.makedirs(PLOTS_DIR, exist_ok=True)
Q_VALUES = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8]

def plot_structure_functions(direction):
    input_file = os.path.join(RESULTS_DIR, f'structure_functions_{direction}_2005_q_extended.csv')
    df = pd.read_csv(input_file)
    plt.figure(figsize=(10, 7))
    for q in Q_VALUES:
        column = f'S{q:g}'
        if column not in df.columns:
            continue
        plt.loglog(df['distance_km'], df[column], marker='o', markersize=3, linewidth=1, label=f'q = {q:g}')
    plt.xlabel('Spatial separation (km)')
    plt.ylabel('$S_q(r)$')
    plt.title(f'Spatial Structure Functions - {direction.capitalize()}, 2005')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    output_file = os.path.join(PLOTS_DIR, f'structure_functions_{direction}_2005_q_extended.png')
    plt.savefig(output_file, dpi=300)
    plt.close()
plot_structure_functions('longitude')
plot_structure_functions('latitude')
