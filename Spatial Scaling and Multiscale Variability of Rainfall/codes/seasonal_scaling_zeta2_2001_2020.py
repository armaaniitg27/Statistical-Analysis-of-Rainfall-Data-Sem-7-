import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import linregress, ttest_rel
BASE = Path('D:\\SBS project\\vikas')
OUT_RESULTS = Path('D:\\SBS project\\sem 7\\chapter2\\results')
OUT_PLOTS = Path('D:\\SBS project\\sem 7\\chapter2\\plots')
OUT_RESULTS.mkdir(parents=True, exist_ok=True)
OUT_PLOTS.mkdir(parents=True, exist_ok=True)
YEARS = range(2001, 2021)
LAT_MIN = 18.05
LAT_MAX = 24.95
LON_MIN = 78.05
LON_MAX = 91.95
N_LAT = 70
N_LON = 140
LON_LAGS = np.arange(8, 14)
LAT_LAGS = np.arange(7, 13)
EXTREME_PERCENTILE = 95
MEAN_LAT = (LAT_MIN + LAT_MAX) / 2
KM_PER_DEG_LAT = 111.32
LON_KM_PER_DEG = KM_PER_DEG_LAT * np.cos(np.deg2rad(MEAN_LAT))

def lon_distance_km(k):
    return k * 0.1 * LON_KM_PER_DEG

def lat_distance_km(k):
    return k * 0.1 * KM_PER_DEG_LAT
EXPECTED_LAT = np.round(np.arange(LAT_MIN, LAT_MAX + 0.001, 0.1), 2)
EXPECTED_LON = np.round(np.arange(LON_MIN, LON_MAX + 0.001, 0.1), 2)

def load_year(year):
    folder = BASE / f'kcsv{year}'
    snapshots = []
    times = []
    files = sorted(folder.glob('ktest*.csv'), key=lambda p: int(p.stem.replace('ktest', '')))
    for count, file in enumerate(files, start=1):
        df = pd.read_csv(file)
        df['lat'] = df['lat'].round(2)
        df['lon'] = df['lon'].round(2)
        df = df[(df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX) & (df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX)].copy()
        if len(df) != N_LAT * N_LON:
            raise ValueError(f'Year {year}, file {file.name}: expected {N_LAT * N_LON} cells, found {len(df)}')
        if df[['lat', 'lon']].duplicated().any():
            raise ValueError(f'Year {year}, file {file.name}: duplicate spatial coordinates found.')
        df = df.sort_values(['lat', 'lon'])
        actual_lat = np.sort(df['lat'].unique())
        actual_lon = np.sort(df['lon'].unique())
        if not np.array_equal(actual_lat, EXPECTED_LAT):
            raise ValueError(f'Year {year}, file {file.name}: latitude grid mismatch.')
        if not np.array_equal(actual_lon, EXPECTED_LON):
            raise ValueError(f'Year {year}, file {file.name}: longitude grid mismatch.')
        rain = df['precipitationCal'].to_numpy().reshape(N_LAT, N_LON)
        snapshots.append(rain)
        times.append(df['time'].iloc[0])
    snapshots = np.asarray(snapshots, dtype=float)
    return (snapshots, times)

def structure_function_q2(data, lags, direction):
    values = []
    for k in lags:
        if direction == 'longitude':
            diff = data[:, :, k:] - data[:, :, :-k]
        elif direction == 'latitude':
            diff = data[:, k:, :] - data[:, :-k, :]
        else:
            raise ValueError('direction must be longitude or latitude')
        s2 = np.mean(np.abs(diff) ** 2, axis=(1, 2))
        values.append(np.mean(s2))
    return np.asarray(values)

def estimate_zeta2(data, lags, direction):
    S2 = structure_function_q2(data, lags, direction)
    if direction == 'longitude':
        distances = np.array([lon_distance_km(k) for k in lags])
    else:
        distances = np.array([lat_distance_km(k) for k in lags])
    slope, intercept, r_value, p_value, stderr = linregress(np.log(distances), np.log(S2))
    return (slope, r_value ** 2, distances, S2)
results = []
for year in YEARS:
    data, times = load_year(year)
    domain_mean = np.nanmean(data, axis=(1, 2))
    threshold = np.percentile(domain_mean, EXTREME_PERCENTILE)
    extreme_mask = domain_mean >= threshold
    extreme_data = data[extreme_mask]
    n_extreme = len(extreme_data)
    expected_extreme = int(np.ceil(len(data) * 0.05))
    zeta_lon, r2_lon, _, _ = estimate_zeta2(extreme_data, LON_LAGS, 'longitude')
    zeta_lat, r2_lat, _, _ = estimate_zeta2(extreme_data, LAT_LAGS, 'latitude')
    results.append({'year': year, 'n_total': len(data), 'n_extreme': n_extreme, 'extreme_threshold': threshold, 'zeta2_longitude_extreme': zeta_lon, 'zeta2_latitude_extreme': zeta_lat, 'r2_longitude_extreme': r2_lon, 'r2_latitude_extreme': r2_lat, 'zeta2_difference_extreme': zeta_lon - zeta_lat, 'zeta2_ratio_extreme': zeta_lon / zeta_lat})
results_df = pd.DataFrame(results)
output_file = OUT_RESULTS / 'extreme_spatial_scaling_zeta2_2001_2020.csv'
results_df.to_csv(output_file, index=False)
summary = {'mean_zeta2_longitude_extreme': results_df['zeta2_longitude_extreme'].mean(), 'std_zeta2_longitude_extreme': results_df['zeta2_longitude_extreme'].std(), 'mean_zeta2_latitude_extreme': results_df['zeta2_latitude_extreme'].mean(), 'std_zeta2_latitude_extreme': results_df['zeta2_latitude_extreme'].std(), 'mean_difference_extreme': results_df['zeta2_difference_extreme'].mean(), 'std_difference_extreme': results_df['zeta2_difference_extreme'].std(), 'mean_ratio_extreme': results_df['zeta2_ratio_extreme'].mean(), 'n_years_lon_gt_lat': np.sum(results_df['zeta2_longitude_extreme'] > results_df['zeta2_latitude_extreme'])}
summary_df = pd.DataFrame([summary])
summary_file = OUT_RESULTS / 'extreme_spatial_scaling_zeta2_summary_2001_2020.csv'
summary_df.to_csv(summary_file, index=False)
