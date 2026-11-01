import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
DATA_FOLDER = 'D:\\SBS project\\vikas\\kcsv2005'
RESULTS_FOLDER = 'D:\\SBS project\\sem 7\\chapter2\\results'
PLOTS_FOLDER = 'D:\\SBS project\\sem 7\\chapter2\\plots'
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(PLOTS_FOLDER, exist_ok=True)
YEAR = 2005
N_LAT = 70
N_LON = 140
N_BLOCK_LAT = 5
N_BLOCK_LON = 5
Q_VALUES = [1, 2, 3, 4]
MAX_LAT_LAG = 13
MAX_LON_LAG = 20
LON_MIN_KM = 80
LON_MAX_KM = 135
LAT_MIN_KM = 78
LAT_MAX_KM = 134
MIN_LAGS = 5
files = [f'ktest{i}.csv' for i in range(2, 8834)]
rainfall_list = []
for idx, filename in enumerate(files, start=1):
    path = os.path.join(DATA_FOLDER, filename)
    df = pd.read_csv(path)
    rainfall_list.append(df['precipitationCal'].to_numpy())
rainfall = np.asarray(rainfall_list)
rainfall = rainfall.reshape(rainfall.shape[0], N_LAT, N_LON)
lat_spacing_deg = 0.1
lon_spacing_deg = 0.1
mean_latitude = np.nanmean(np.arange(18.05, 25.0, 0.1))
lat_km_per_cell = 111.32 * lat_spacing_deg
lon_km_per_cell = 111.32 * lon_spacing_deg * np.cos(np.deg2rad(mean_latitude))
lat_edges = np.linspace(0, N_LAT, N_BLOCK_LAT + 1, dtype=int)
lon_edges = np.linspace(0, N_LON, N_BLOCK_LON + 1, dtype=int)

def calculate_structure_functions(block, direction, max_lag, q_values):
    results = []
    if direction == 'longitude':
        max_possible_lag = block.shape[2] - 1
        max_lag = min(max_lag, max_possible_lag)
        for lag in range(1, max_lag + 1):
            increments = block[:, :, lag:] - block[:, :, :-lag]
            increments = np.abs(increments).ravel()
            increments = increments[np.isfinite(increments)]
            if len(increments) == 0:
                continue
            row = {'lag': lag, 'distance_km': lag * lon_km_per_cell, 'number_of_increments': len(increments)}
            for q in q_values:
                row[f'S{q}'] = np.mean(increments ** q)
            results.append(row)
    elif direction == 'latitude':
        max_possible_lag = block.shape[1] - 1
        max_lag = min(max_lag, max_possible_lag)
        for lag in range(1, max_lag + 1):
            increments = block[:, lag:, :] - block[:, :-lag, :]
            increments = np.abs(increments).ravel()
            increments = increments[np.isfinite(increments)]
            if len(increments) == 0:
                continue
            row = {'lag': lag, 'distance_km': lag * lat_km_per_cell, 'number_of_increments': len(increments)}
            for q in q_values:
                row[f'S{q}'] = np.mean(increments ** q)
            results.append(row)
    return pd.DataFrame(results)

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
    return {'zeta': slope, 'intercept': intercept, 'R2': r2, 'n_lags': len(subset), 'min_distance_km': subset['distance_km'].min(), 'max_distance_km': subset['distance_km'].max()}
all_results = []
block_counter = 0
for bi in range(N_BLOCK_LAT):
    lat_start = lat_edges[bi]
    lat_end = lat_edges[bi + 1]
    for bj in range(N_BLOCK_LON):
        lon_start = lon_edges[bj]
        lon_end = lon_edges[bj + 1]
        block_counter += 1
        block = rainfall[:, lat_start:lat_end, lon_start:lon_end]
        sf_lon = calculate_structure_functions(block, 'longitude', MAX_LON_LAG, Q_VALUES)
        sf_lat = calculate_structure_functions(block, 'latitude', MAX_LAT_LAG, Q_VALUES)
        sf_lon.to_csv(os.path.join(RESULTS_FOLDER, f'block_B{block_counter:02d}_structure_longitude_2005.csv'), index=False)
        sf_lat.to_csv(os.path.join(RESULTS_FOLDER, f'block_B{block_counter:02d}_structure_latitude_2005.csv'), index=False)
        row = {'block': f'B{block_counter:02d}', 'block_row': bi + 1, 'block_column': bj + 1, 'lat_start_index': lat_start, 'lat_end_index': lat_end - 1, 'lon_start_index': lon_start, 'lon_end_index': lon_end - 1, 'block_n_lat': lat_end - lat_start, 'block_n_lon': lon_end - lon_start}
        for q in Q_VALUES:
            fit = fit_scaling(sf_lon, q, LON_MIN_KM, LON_MAX_KM)
            if fit is None:
                row[f'lon_zeta_q{q}'] = np.nan
                row[f'lon_R2_q{q}'] = np.nan
                row[f'lon_n_lags_q{q}'] = 0
            else:
                row[f'lon_zeta_q{q}'] = fit['zeta']
                row[f'lon_R2_q{q}'] = fit['R2']
                row[f'lon_n_lags_q{q}'] = fit['n_lags']
        for q in Q_VALUES:
            fit = fit_scaling(sf_lat, q, LAT_MIN_KM, LAT_MAX_KM)
            if fit is None:
                row[f'lat_zeta_q{q}'] = np.nan
                row[f'lat_R2_q{q}'] = np.nan
                row[f'lat_n_lags_q{q}'] = 0
            else:
                row[f'lat_zeta_q{q}'] = fit['zeta']
                row[f'lat_R2_q{q}'] = fit['R2']
                row[f'lat_n_lags_q{q}'] = fit['n_lags']
        all_results.append(row)
results = pd.DataFrame(all_results)
output_file = os.path.join(RESULTS_FOLDER, 'local_scaling_heterogeneity_2005.csv')
results.to_csv(output_file, index=False)
for direction in ['lon', 'lat']:
    for q in Q_VALUES:
        column = f'{direction}_zeta_q{q}'
        values = results[column].dropna()
for direction in ['lon', 'lat']:
    matrix = results.pivot(index='block_row', columns='block_column', values=f'{direction}_zeta_q2')
    plt.figure(figsize=(8, 6))
    plt.imshow(matrix.to_numpy(), origin='upper', aspect='auto')
    plt.colorbar(label='$\\zeta(2)$')
    plt.xlabel('Block column')
    plt.ylabel('Block row')
    plt.title(f'Local spatial scaling exponent $\\zeta(2)$ — {direction} direction — 2005')
    plt.xticks(range(N_BLOCK_LON), range(1, N_BLOCK_LON + 1))
    plt.yticks(range(N_BLOCK_LAT), range(1, N_BLOCK_LAT + 1))
    plt.tight_layout()
    filename = os.path.join(PLOTS_FOLDER, f'local_zeta2_heatmap_{direction}_2005.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
for q in Q_VALUES:
    data = [results[f'lon_zeta_q{q}'].dropna(), results[f'lat_zeta_q{q}'].dropna()]
    plt.figure(figsize=(7, 5))
    plt.boxplot(data, labels=['Longitude', 'Latitude'])
    plt.ylabel('$\\zeta(q)$')
    plt.title(f'Spatial heterogeneity of local scaling exponent $\\zeta({q})$ — 2005')
    plt.tight_layout()
    filename = os.path.join(PLOTS_FOLDER, f'local_zeta_q{q}_boxplot_2005.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
