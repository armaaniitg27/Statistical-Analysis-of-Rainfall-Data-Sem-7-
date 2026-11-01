import os
import numpy as np
import pandas as pd
RESULTS_FOLDER = 'D:\\SBS project\\sem 7\\chapter2\\results'
N_BLOCKS = 25
Q = 2
WINDOWS = {'W1': (70, 120), 'W2': (80, 130), 'W3': (80, 135), 'W4': (90, 140), 'W5': (90, 150)}
MIN_LAGS = 5

def fit_scaling(df, q, min_km, max_km):
    subset = df[(df['distance_km'] >= min_km) & (df['distance_km'] <= max_km)].copy()
    subset = subset[(subset[f'S{q}'] > 0) & np.isfinite(subset[f'S{q}'])]
    if len(subset) < MIN_LAGS:
        return None
    x = np.log10(subset['distance_km'].to_numpy())
    y = np.log10(subset[f'S{q}'].to_numpy())
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        r2 = np.nan
    else:
        r2 = 1 - ss_res / ss_tot
    return (slope, r2, len(subset))
records = []
for block_number in range(1, N_BLOCKS + 1):
    block_name = f'B{block_number:02d}'
    lon_file = os.path.join(RESULTS_FOLDER, f'block_{block_name}_structure_longitude_2005.csv')
    lat_file = os.path.join(RESULTS_FOLDER, f'block_{block_name}_structure_latitude_2005.csv')
    lon_df = pd.read_csv(lon_file)
    lat_df = pd.read_csv(lat_file)
    for window_name, (low, high) in WINDOWS.items():
        lon_fit = fit_scaling(lon_df, Q, low, high)
        if lon_fit is None:
            lon_zeta = np.nan
            lon_r2 = np.nan
            lon_n = 0
        else:
            lon_zeta = lon_fit[0]
            lon_r2 = lon_fit[1]
            lon_n = lon_fit[2]
        lat_fit = fit_scaling(lat_df, Q, low, high)
        if lat_fit is None:
            lat_zeta = np.nan
            lat_r2 = np.nan
            lat_n = 0
        else:
            lat_zeta = lat_fit[0]
            lat_r2 = lat_fit[1]
            lat_n = lat_fit[2]
        records.append({'block': block_name, 'window': window_name, 'min_km': low, 'max_km': high, 'lon_zeta_q2': lon_zeta, 'lon_R2_q2': lon_r2, 'lon_n_lags': lon_n, 'lat_zeta_q2': lat_zeta, 'lat_R2_q2': lat_r2, 'lat_n_lags': lat_n})
results = pd.DataFrame(records)
output_file = os.path.join(RESULTS_FOLDER, 'local_scaling_window_stability_2005.csv')
results.to_csv(output_file, index=False)
for direction in ['lon', 'lat']:
    summary_rows = []
    for window_name in WINDOWS:
        subset = results[results['window'] == window_name]
        zeta = subset[f'{direction}_zeta_q2'].dropna()
        r2 = subset[f'{direction}_R2_q2'].dropna()
        summary_rows.append({'window': window_name, 'zeta_mean': zeta.mean(), 'zeta_median': zeta.median(), 'zeta_std': zeta.std(), 'zeta_min': zeta.min(), 'zeta_max': zeta.max(), 'mean_R2': r2.mean(), 'median_R2': r2.median(), 'blocks_R2_ge_0.90': np.sum(r2 >= 0.9), 'blocks_R2_ge_0.95': np.sum(r2 >= 0.95), 'blocks_R2_ge_0.98': np.sum(r2 >= 0.98)})
    summary = pd.DataFrame(summary_rows)
for direction in ['lon', 'lat']:
    for block in [f'B{i:02d}' for i in range(1, N_BLOCKS + 1)]:
        subset = results[results['block'] == block]
        values = subset[f'{direction}_zeta_q2'].dropna()
        if len(values) == 0:
            continue
