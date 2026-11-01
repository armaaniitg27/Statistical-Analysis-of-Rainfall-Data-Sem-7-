import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
RESULTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\results'
PLOTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\plots'
os.makedirs(PLOTS_DIR, exist_ok=True)
LON_FILE = os.path.join(RESULTS_DIR, 'structure_functions_longitude_2005.csv')
LAT_FILE = os.path.join(RESULTS_DIR, 'structure_functions_latitude_2005.csv')
Q_VALUES = [1, 2, 3, 4, 5]
LON_WINDOWS = {'longitude_80_135': (80, 135), 'longitude_90_155': (90, 155), 'longitude_155_207': (155, 207), 'longitude_373_425': (373, 425), 'longitude_383_435': (383, 435), 'longitude_393_445': (393, 445)}
LAT_WINDOWS = {'latitude_67_122': (67, 122), 'latitude_78_134': (78, 134), 'latitude_111_167': (111, 167), 'latitude_223_278': (223, 278), 'latitude_234_289': (234, 289), 'latitude_245_301': (245, 301)}
lon_df = pd.read_csv(LON_FILE)
lat_df = pd.read_csv(LAT_FILE)

def fit_window(df, start_distance, end_distance, direction):
    mask = (df['distance_km'] >= start_distance) & (df['distance_km'] <= end_distance)
    subset = df.loc[mask].copy()
    if len(subset) < 4:
        return None
    x = np.log10(subset['distance_km'].to_numpy())
    results = {'direction': direction, 'start_distance_km': subset['distance_km'].iloc[0], 'end_distance_km': subset['distance_km'].iloc[-1], 'n_points': len(subset)}
    zetas = []
    for q in Q_VALUES:
        column = f'S{q}'
        if column not in subset.columns:
            raise ValueError(f"Missing column '{column}' in structure-function file.")
        y_values = subset[column].to_numpy()
        valid = y_values > 0
        x_valid = x[valid]
        y_valid = np.log10(y_values[valid])
        slope, intercept = np.polyfit(x_valid, y_valid, 1)
        y_pred = slope * x_valid + intercept
        ss_res = np.sum((y_valid - y_pred) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        r2 = 1 - ss_res / ss_tot
        results[f'zeta_q{q}'] = slope
        results[f'R2_q{q}'] = r2
        zetas.append(slope)
    results['zeta_mean'] = np.mean(zetas)
    results['zeta_std'] = np.std(zetas)
    return results
lon_results = []
for name, (start, end) in LON_WINDOWS.items():
    result = fit_window(lon_df, start, end, 'longitude')
    if result is not None:
        result['window'] = name
        lon_results.append(result)
lon_results = pd.DataFrame(lon_results)
lat_results = []
for name, (start, end) in LAT_WINDOWS.items():
    result = fit_window(lat_df, start, end, 'latitude')
    if result is not None:
        result['window'] = name
        lat_results.append(result)
lat_results = pd.DataFrame(lat_results)
lon_output = os.path.join(RESULTS_DIR, 'scaling_exponent_robustness_longitude_2005.csv')
lat_output = os.path.join(RESULTS_DIR, 'scaling_exponent_robustness_latitude_2005.csv')
lon_results.to_csv(lon_output, index=False)
lat_results.to_csv(lat_output, index=False)
plt.figure(figsize=(9, 6))
for _, row in lon_results.iterrows():
    zetas = [row[f'zeta_q{q}'] for q in Q_VALUES]
    plt.plot(Q_VALUES, zetas, marker='o', label=row['window'])
plt.xlabel('Order q')
plt.ylabel('$\\zeta(q)$')
plt.title('Longitudinal Spatial Scaling Exponents — 2005')
plt.xticks(Q_VALUES)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
lon_plot = os.path.join(PLOTS_DIR, 'zeta_q_robustness_longitude_2005.png')
plt.savefig(lon_plot, dpi=300, bbox_inches='tight')
plt.close()
plt.figure(figsize=(9, 6))
for _, row in lat_results.iterrows():
    zetas = [row[f'zeta_q{q}'] for q in Q_VALUES]
    plt.plot(Q_VALUES, zetas, marker='o', label=row['window'])
plt.xlabel('Order q')
plt.ylabel('$\\zeta(q)$')
plt.title('Latitudinal Spatial Scaling Exponents — 2005')
plt.xticks(Q_VALUES)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
lat_plot = os.path.join(PLOTS_DIR, 'zeta_q_robustness_latitude_2005.png')
plt.savefig(lat_plot, dpi=300, bbox_inches='tight')
plt.close()
