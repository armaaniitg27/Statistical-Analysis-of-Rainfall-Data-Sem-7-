import os
import glob
import numpy as np
import pandas as pd
DATA_ROOT = 'D:\\SBS project\\vikas'
RESULTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\results'
os.makedirs(RESULTS_DIR, exist_ok=True)
YEARS = [2018, 2019, 2020]
N_LAT = 70
N_LON = 140
EXPECTED_CELLS = N_LAT * N_LON
LAT_MIN = 18.05
LAT_MAX = 24.95
LON_MIN = 78.05
LON_MAX = 91.95
LON_SPACING_KM = 10.357408
LAT_SPACING_KM = 11.132
LON_MIN_KM = 80
LON_MAX_KM = 135
LAT_MIN_KM = 78
LAT_MAX_KM = 134

def load_year(year):
    year_folder = os.path.join(DATA_ROOT, f'kcsv{year}')
    files = sorted(glob.glob(os.path.join(year_folder, 'ktest*.csv')))
    rainfall = []
    for i, file in enumerate(files):
        df = pd.read_csv(file)
        df['lat'] = df['lat'].round(2)
        df['lon'] = df['lon'].round(2)
        df = df[(df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX) & (df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX)].copy()
        if len(df) != EXPECTED_CELLS:
            raise ValueError(f'\n{os.path.basename(file)} has {len(df)} cells after common-domain filtering. Expected {EXPECTED_CELLS}.')
        duplicates = df.duplicated(subset=['lat', 'lon']).sum()
        if duplicates > 0:
            raise ValueError(f'\n{os.path.basename(file)} contains {duplicates} duplicate (lat, lon) cells.')
        n_lat = df['lat'].nunique()
        n_lon = df['lon'].nunique()
        if n_lat != N_LAT:
            raise ValueError(f'\n{os.path.basename(file)} has {n_lat} unique latitudes. Expected {N_LAT}.')
        if n_lon != N_LON:
            raise ValueError(f'\n{os.path.basename(file)} has {n_lon} unique longitudes. Expected {N_LON}.')
        df = df.sort_values(['lat', 'lon'])
        rainfall.append(df['precipitationCal'].to_numpy(dtype=float))
    rainfall = np.asarray(rainfall, dtype=float)
    rainfall = rainfall.reshape(len(files), N_LAT, N_LON)
    return rainfall

def calculate_S2(rainfall, direction):
    distances = []
    S2_values = []
    if direction == 'longitude':
        max_lag = 13
        spacing = LON_SPACING_KM
        for lag in range(1, max_lag + 1):
            diff = rainfall[:, :, lag:] - rainfall[:, :, :-lag]
            S2 = np.mean(np.abs(diff) ** 2)
            distances.append(lag * spacing)
            S2_values.append(S2)
    elif direction == 'latitude':
        max_lag = 12
        spacing = LAT_SPACING_KM
        for lag in range(1, max_lag + 1):
            diff = rainfall[:, lag:, :] - rainfall[:, :-lag, :]
            S2 = np.mean(np.abs(diff) ** 2)
            distances.append(lag * spacing)
            S2_values.append(S2)
    else:
        raise ValueError("Direction must be 'longitude' or 'latitude'.")
    return (np.asarray(distances), np.asarray(S2_values))

def fit_scaling(distances, S2, min_km, max_km):
    mask = (distances >= min_km) & (distances <= max_km) & (S2 > 0)
    x = np.log(distances[mask])
    y = np.log(S2[mask])
    if len(x) < 3:
        raise ValueError('Not enough points for scaling fit.')
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    R2 = 1 - ss_res / ss_tot
    return (slope, intercept, R2, len(x))
results = []
for year in YEARS:
    rainfall = load_year(year)
    lon_dist, lon_S2 = calculate_S2(rainfall, 'longitude')
    lon_zeta2, lon_intercept, lon_R2, lon_n = fit_scaling(lon_dist, lon_S2, LON_MIN_KM, LON_MAX_KM)
    lat_dist, lat_S2 = calculate_S2(rainfall, 'latitude')
    lat_zeta2, lat_intercept, lat_R2, lat_n = fit_scaling(lat_dist, lat_S2, LAT_MIN_KM, LAT_MAX_KM)
    results.append({'year': year, 'lon_zeta2': lon_zeta2, 'lon_R2': lon_R2, 'lon_n': lon_n, 'lat_zeta2': lat_zeta2, 'lat_R2': lat_R2, 'lat_n': lat_n, 'zeta2_difference': lon_zeta2 - lat_zeta2, 'zeta2_ratio': lon_zeta2 / lat_zeta2})
results = pd.DataFrame(results)
output_file = os.path.join(RESULTS_DIR, 'interannual_scaling_zeta2_2018_2020.csv')
results.to_csv(output_file, index=False)
