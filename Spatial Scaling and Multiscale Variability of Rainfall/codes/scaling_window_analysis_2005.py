import os
import numpy as np
import pandas as pd
RESULTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\results'
lon_file = os.path.join(RESULTS_DIR, 'structure_functions_longitude_2005.csv')
lat_file = os.path.join(RESULTS_DIR, 'structure_functions_latitude_2005.csv')
lon = pd.read_csv(lon_file)
lat = pd.read_csv(lat_file)
Q_VALUES = [1, 2, 3, 4, 5]
MIN_POINTS = 6
MAX_POINTS = 15

def evaluate_window(df, start, end):
    x = np.log(df['distance_km'].iloc[start:end].to_numpy())
    results = []
    for q in Q_VALUES:
        y = np.log(df[f'S{q}'].iloc[start:end].to_numpy())
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        if ss_tot > 0:
            r_squared = 1 - ss_res / ss_tot
        else:
            r_squared = np.nan
        results.append({'q': q, 'slope': slope, 'r_squared': r_squared})
    return results

def scan_windows(df, direction):
    all_results = []
    n = len(df)
    for window_size in range(MIN_POINTS, min(MAX_POINTS, n) + 1):
        for start in range(0, n - window_size + 1):
            end = start + window_size
            fit_results = evaluate_window(df, start, end)
            slopes = [item['slope'] for item in fit_results]
            r2_values = [item['r_squared'] for item in fit_results]
            all_results.append({'direction': direction, 'window_size': window_size, 'start_distance_km': df['distance_km'].iloc[start], 'end_distance_km': df['distance_km'].iloc[end - 1], 'mean_R2': np.mean(r2_values), 'minimum_R2': np.min(r2_values), 'slope_std': np.std(slopes), 'zeta_q1': slopes[0], 'zeta_q2': slopes[1], 'zeta_q3': slopes[2], 'zeta_q4': slopes[3], 'zeta_q5': slopes[4], 'R2_q1': r2_values[0], 'R2_q2': r2_values[1], 'R2_q3': r2_values[2], 'R2_q4': r2_values[3], 'R2_q5': r2_values[4]})
    return pd.DataFrame(all_results)
lon_windows = scan_windows(lon, 'longitude')
lat_windows = scan_windows(lat, 'latitude')
lon_output = os.path.join(RESULTS_DIR, 'scaling_windows_longitude_2005.csv')
lat_output = os.path.join(RESULTS_DIR, 'scaling_windows_latitude_2005.csv')
lon_windows.to_csv(lon_output, index=False)
lat_windows.to_csv(lat_output, index=False)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', '{:.5f}'.format)
best_lon = lon_windows.sort_values(['mean_R2', 'minimum_R2'], ascending=False).head(15)
best_lat = lat_windows.sort_values(['mean_R2', 'minimum_R2'], ascending=False).head(15)
consistent_lon = lon_windows.sort_values(['slope_std', 'mean_R2'], ascending=[True, False]).head(15)
consistent_lat = lat_windows.sort_values(['slope_std', 'mean_R2'], ascending=[True, False]).head(15)
