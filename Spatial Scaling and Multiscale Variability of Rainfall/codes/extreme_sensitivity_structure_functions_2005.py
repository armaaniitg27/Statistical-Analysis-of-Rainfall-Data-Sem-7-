import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
DATA_DIR = 'D:\\SBS project\\vikas'
RESULTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\results'
PLOTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\plots'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
YEAR = 2005
N_LAT = 70
N_LON = 140
MAX_LAG = 50
Q_VALUES = [1, 2, 3, 4, 5]
PERCENTILES = [95, 99]
LON_SCALING_RANGE = (80, 135)
LAT_SCALING_RANGE = (78, 134)
year_dir = os.path.join(DATA_DIR, f'kcsv{YEAR}')
files = [os.path.join(year_dir, f'ktest{i}.csv') for i in range(2, 8834)]
rainfall = []
for idx, file_path in enumerate(files):
    df = pd.read_csv(file_path)
    if 'precipitationCal' not in df.columns:
        raise ValueError(f'Missing precipitationCal in {file_path}')
    values = df['precipitationCal'].to_numpy()
    if len(values) != N_LAT * N_LON:
        raise ValueError(f'Unexpected number of grid cells in {file_path}: {len(values)}')
    rainfall.append(values)
rainfall = np.asarray(rainfall, dtype=np.float32)
rainfall = rainfall.reshape(rainfall.shape[0], N_LAT, N_LON)
LAT_SPACING_DEG = 0.1
LON_SPACING_DEG = 0.1
LAT_KM_PER_DEG = 111.32
mean_latitude = (18.05 + 24.95) / 2.0
LON_KM_PER_DEG = LAT_KM_PER_DEG * np.cos(np.radians(mean_latitude))
LON_STEP_KM = LON_SPACING_DEG * LON_KM_PER_DEG
LAT_STEP_KM = LAT_SPACING_DEG * LAT_KM_PER_DEG

def collect_increments(rainfall_array, direction, lag):
    if direction == 'longitude':
        left = rainfall_array[:, :, :-lag]
        right = rainfall_array[:, :, lag:]
    elif direction == 'latitude':
        left = rainfall_array[:, :-lag, :]
        right = rainfall_array[:, lag:, :]
    else:
        raise ValueError('Direction must be longitude or latitude.')
    increments = right - left
    increments = increments.ravel()
    increments = increments[np.isfinite(increments)]
    return increments
lon_lags_for_threshold = [k for k in range(1, MAX_LAG + 1) if LON_SCALING_RANGE[0] <= k * LON_STEP_KM <= LON_SCALING_RANGE[1]]
lat_lags_for_threshold = [k for k in range(1, MAX_LAG + 1) if LAT_SCALING_RANGE[0] <= k * LAT_STEP_KM <= LAT_SCALING_RANGE[1]]
lon_threshold_sample = []
for lag in lon_lags_for_threshold:
    increments = collect_increments(rainfall, 'longitude', lag)
    abs_inc = np.abs(increments)
    if len(abs_inc) > 2000000:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(abs_inc), size=2000000, replace=False)
        abs_inc = abs_inc[indices]
    lon_threshold_sample.append(abs_inc)
lon_threshold_sample = np.concatenate(lon_threshold_sample)
lat_threshold_sample = []
for lag in lat_lags_for_threshold:
    increments = collect_increments(rainfall, 'latitude', lag)
    abs_inc = np.abs(increments)
    if len(abs_inc) > 2000000:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(abs_inc), size=2000000, replace=False)
        abs_inc = abs_inc[indices]
    lat_threshold_sample.append(abs_inc)
lat_threshold_sample = np.concatenate(lat_threshold_sample)
lon_thresholds = {p: np.percentile(lon_threshold_sample, p) for p in PERCENTILES}
lat_thresholds = {p: np.percentile(lat_threshold_sample, p) for p in PERCENTILES}

def calculate_structure_functions(rainfall_array, direction, mode, percentile=None):
    results = []
    if direction == 'longitude':
        thresholds = lon_thresholds
        step_km = LON_STEP_KM
    else:
        thresholds = lat_thresholds
        step_km = LAT_STEP_KM
    for lag in range(1, MAX_LAG + 1):
        increments = collect_increments(rainfall_array, direction, lag)
        abs_increments = np.abs(increments)
        if mode == 'original':
            processed = increments
        elif mode == 'filtered':
            threshold = thresholds[percentile]
            processed = increments[abs_increments <= threshold]
        elif mode == 'winsorized':
            threshold = thresholds[percentile]
            processed = np.clip(increments, -threshold, threshold)
        else:
            raise ValueError('Unknown mode.')
        result = {'lag': lag, 'distance_km': lag * step_km, 'number_of_increments': len(processed), 'mode': mode}
        if percentile is not None:
            result['percentile'] = percentile
        else:
            result['percentile'] = 100
        for q in Q_VALUES:
            result[f'S{q}'] = np.mean(np.abs(processed) ** q)
        results.append(result)
    return pd.DataFrame(results)
all_results = []
modes = [('original', None), ('filtered', 95), ('filtered', 99), ('winsorized', 95), ('winsorized', 99)]
for direction in ['longitude', 'latitude']:
    for mode, percentile in modes:
        df_result = calculate_structure_functions(rainfall, direction, mode, percentile)
        df_result['direction'] = direction
        all_results.append(df_result)
all_results = pd.concat(all_results, ignore_index=True)
output_file = os.path.join(RESULTS_DIR, 'extreme_sensitivity_structure_functions_2005.csv')
all_results.to_csv(output_file, index=False)

def estimate_zeta(df, start_distance, end_distance):
    subset = df[(df['distance_km'] >= start_distance) & (df['distance_km'] <= end_distance)].copy()
    x = np.log10(subset['distance_km'].to_numpy())
    result = {'direction': df['direction'].iloc[0], 'mode': df['mode'].iloc[0], 'percentile': df['percentile'].iloc[0], 'start_distance_km': subset['distance_km'].iloc[0], 'end_distance_km': subset['distance_km'].iloc[-1]}
    for q in Q_VALUES:
        y = np.log10(subset[f'S{q}'].to_numpy())
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot
        result[f'zeta_q{q}'] = slope
        result[f'R2_q{q}'] = r2
    return result
zeta_results = []
for direction in ['longitude', 'latitude']:
    if direction == 'longitude':
        start, end = LON_SCALING_RANGE
    else:
        start, end = LAT_SCALING_RANGE
    direction_data = all_results[all_results['direction'] == direction]
    for (mode, percentile), group in direction_data.groupby(['mode', 'percentile']):
        result = estimate_zeta(group, start, end)
        zeta_results.append(result)
zeta_results = pd.DataFrame(zeta_results)
zeta_output = os.path.join(RESULTS_DIR, 'extreme_sensitivity_zeta_2005.csv')
zeta_results.to_csv(zeta_output, index=False)
lon_plot_data = zeta_results[zeta_results['direction'] == 'longitude']
plt.figure(figsize=(9, 6))
for _, row in lon_plot_data.iterrows():
    label = row['mode']
    if row['percentile'] != 100:
        label += f" {int(row['percentile'])}%"
    zetas = [row[f'zeta_q{q}'] for q in Q_VALUES]
    plt.plot(Q_VALUES, zetas, marker='o', label=label)
plt.xlabel('Order q')
plt.ylabel('$\\zeta(q)$')
plt.title('Extreme Sensitivity of Longitudinal Scaling Exponents — 2005')
plt.xticks(Q_VALUES)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
lon_plot = os.path.join(PLOTS_DIR, 'extreme_sensitivity_zeta_longitude_2005.png')
plt.savefig(lon_plot, dpi=300, bbox_inches='tight')
plt.close()
lat_plot_data = zeta_results[zeta_results['direction'] == 'latitude']
plt.figure(figsize=(9, 6))
for _, row in lat_plot_data.iterrows():
    label = row['mode']
    if row['percentile'] != 100:
        label += f" {int(row['percentile'])}%"
    zetas = [row[f'zeta_q{q}'] for q in Q_VALUES]
    plt.plot(Q_VALUES, zetas, marker='o', label=label)
plt.xlabel('Order q')
plt.ylabel('$\\zeta(q)$')
plt.title('Extreme Sensitivity of Latitudinal Scaling Exponents — 2005')
plt.xticks(Q_VALUES)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
lat_plot = os.path.join(PLOTS_DIR, 'extreme_sensitivity_zeta_latitude_2005.png')
plt.savefig(lat_plot, dpi=300, bbox_inches='tight')
plt.close()
