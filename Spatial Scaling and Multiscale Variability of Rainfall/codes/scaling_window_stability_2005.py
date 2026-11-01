import os
import numpy as np
import pandas as pd
RESULTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\results'
LON_FILE = os.path.join(RESULTS_DIR, 'structure_functions_longitude_2005.csv')
LAT_FILE = os.path.join(RESULTS_DIR, 'structure_functions_latitude_2005.csv')
ORDERS = [1, 2, 3, 4, 5]
MIN_WINDOW = 6
MAX_WINDOW = 20
lon_df = pd.read_csv(LON_FILE)
lat_df = pd.read_csv(LAT_FILE)
for column in ['distance_km'] + [f'S{i}' for i in ORDERS]:
    if column not in lon_df.columns:
        raise ValueError(f"Missing column '{column}' in longitude structure-function file.\nAvailable columns: {list(lon_df.columns)}")
    if column not in lat_df.columns:
        raise ValueError(f"Missing column '{column}' in latitude structure-function file.\nAvailable columns: {list(lat_df.columns)}")

def analyze_windows(df, direction):
    distances = df['distance_km'].to_numpy(dtype=float)
    results = []
    n = len(df)
    max_window = min(MAX_WINDOW, n)
    for window_size in range(MIN_WINDOW, max_window + 1):
        for start in range(0, n - window_size + 1):
            end = start + window_size
            x = np.log10(distances[start:end])
            zetas = []
            r2_values = []
            for q in ORDERS:
                Sq = df[f'S{q}'].to_numpy(dtype=float)
                y = np.log10(Sq[start:end])
                slope, intercept = np.polyfit(x, y, 1)
                y_pred = slope * x + intercept
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                if ss_tot == 0:
                    r2 = np.nan
                else:
                    r2 = 1 - ss_res / ss_tot
                zetas.append(slope)
                r2_values.append(r2)
            zetas = np.asarray(zetas)
            r2_values = np.asarray(r2_values)
            results.append({'direction': direction, 'window_size': window_size, 'start_distance_km': distances[start], 'end_distance_km': distances[end - 1], 'mean_R2': np.mean(r2_values), 'minimum_R2': np.min(r2_values), 'slope_std': np.std(zetas), 'zeta_q1': zetas[0], 'zeta_q2': zetas[1], 'zeta_q3': zetas[2], 'zeta_q4': zetas[3], 'zeta_q5': zetas[4], 'R2_q1': r2_values[0], 'R2_q2': r2_values[1], 'R2_q3': r2_values[2], 'R2_q4': r2_values[3], 'R2_q5': r2_values[4]})
    return pd.DataFrame(results)
lon_results = analyze_windows(lon_df, 'longitude')
lat_results = analyze_windows(lat_df, 'latitude')
lon_best = lon_results.sort_values(['mean_R2', 'minimum_R2'], ascending=False).head(15)
lat_best = lat_results.sort_values(['mean_R2', 'minimum_R2'], ascending=False).head(15)
lon_consistent = lon_results[lon_results['mean_R2'] >= 0.9].sort_values(['slope_std', 'mean_R2'], ascending=[True, False]).head(15)
lat_consistent = lat_results[lat_results['mean_R2'] >= 0.9].sort_values(['slope_std', 'mean_R2'], ascending=[True, False]).head(15)
lon_output = os.path.join(RESULTS_DIR, 'scaling_windows_longitude_2005.csv')
lat_output = os.path.join(RESULTS_DIR, 'scaling_windows_latitude_2005.csv')
lon_results.to_csv(lon_output, index=False)
lat_results.to_csv(lat_output, index=False)
