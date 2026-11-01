import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
DATA_ROOT = 'D:\\SBS project\\vikas'
RESULTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\results'
PLOTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\plots'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
YEARS = range(2001, 2021)
N_LAT = 70
N_LON = 140
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
        rainfall.append(df['precipitationCal'].to_numpy())
    rainfall = np.asarray(rainfall)
    rainfall = rainfall.reshape(len(files), N_LAT, N_LON)
    return rainfall

def calculate_S2(rainfall, direction):
    if direction == 'longitude':
        spacing_km = 10.357408
        max_lag = 13
        distances = []
        S2_values = []
        for lag in range(1, max_lag + 1):
            diff = rainfall[:, :, lag:] - rainfall[:, :, :-lag]
            S2 = np.mean(np.abs(diff) ** 2)
            distance = lag * spacing_km
            distances.append(distance)
            S2_values.append(S2)
    elif direction == 'latitude':
        spacing_km = 11.132
        max_lag = 12
        distances = []
        S2_values = []
        for lag in range(1, max_lag + 1):
            diff = rainfall[:, lag:, :] - rainfall[:, :-lag, :]
            S2 = np.mean(np.abs(diff) ** 2)
            distance = lag * spacing_km
            distances.append(distance)
            S2_values.append(S2)
    else:
        raise ValueError('direction must be longitude or latitude')
    return (np.asarray(distances), np.asarray(S2_values))

def fit_scaling(distances, S2, min_km, max_km):
    mask = (distances >= min_km) & (distances <= max_km) & (S2 > 0)
    x = np.log(distances[mask])
    y = np.log(S2[mask])
    if len(x) < 3:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {'zeta2': slope, 'intercept': intercept, 'R2': r2, 'n_points': len(x)}
results = []
for year in YEARS:
    rainfall = load_year(year)
    lon_dist, lon_S2 = calculate_S2(rainfall, 'longitude')
    lon_fit = fit_scaling(lon_dist, lon_S2, LON_MIN_KM, LON_MAX_KM)
    lat_dist, lat_S2 = calculate_S2(rainfall, 'latitude')
    lat_fit = fit_scaling(lat_dist, lat_S2, LAT_MIN_KM, LAT_MAX_KM)
    results.append({'year': year, 'lon_zeta2': lon_fit['zeta2'], 'lon_R2': lon_fit['R2'], 'lon_n': lon_fit['n_points'], 'lat_zeta2': lat_fit['zeta2'], 'lat_R2': lat_fit['R2'], 'lat_n': lat_fit['n_points'], 'zeta2_difference': lon_fit['zeta2'] - lat_fit['zeta2'], 'zeta2_ratio': lon_fit['zeta2'] / lat_fit['zeta2'] if lat_fit['zeta2'] != 0 else np.nan})
results = pd.DataFrame(results)
output_file = os.path.join(RESULTS_DIR, 'interannual_scaling_zeta2_2001_2020.csv')
results.to_csv(output_file, index=False)
for direction in ['lon', 'lat']:
    values = results[f'{direction}_zeta2']
    R2 = results[f'{direction}_R2']
lon_larger = (results['lon_zeta2'] > results['lat_zeta2']).sum()
lat_larger = (results['lat_zeta2'] > results['lon_zeta2']).sum()
equal = (results['lon_zeta2'] == results['lat_zeta2']).sum()
for direction in ['lon', 'lat']:
    column = f'{direction}_zeta2'
    max_row = results.loc[results[column].idxmax()]
    min_row = results.loc[results[column].idxmin()]
trend_results = []
for direction in ['lon', 'lat']:
    y = results[f'{direction}_zeta2'].to_numpy()
    x = results['year'].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    trend_results.append({'direction': direction, 'slope_per_year': slope, 'R2': r2})
plt.figure(figsize=(9, 5))
plt.plot(results['year'], results['lon_zeta2'], marker='o', label='Longitude')
plt.plot(results['year'], results['lat_zeta2'], marker='s', label='Latitude')
plt.xlabel('Year')
plt.ylabel('Spatial scaling exponent $\\zeta(2)$')
plt.title('Interannual variation of spatial scaling exponent $\\zeta(2)$')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plot_file = os.path.join(PLOTS_DIR, 'interannual_zeta2_2001_2020.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()
plt.figure(figsize=(9, 5))
plt.axhline(0, linestyle='--')
plt.plot(results['year'], results['zeta2_difference'], marker='o')
plt.xlabel('Year')
plt.ylabel('$\\zeta_{\\mathrm{lon}}(2)-\\zeta_{\\mathrm{lat}}(2)$')
plt.title('Interannual directional difference in spatial scaling')
plt.grid(alpha=0.3)
plt.tight_layout()
plot_file = os.path.join(PLOTS_DIR, 'interannual_zeta2_difference_2001_2020.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()
plt.figure(figsize=(9, 5))
plt.plot(results['year'], results['lon_R2'], marker='o', label='Longitude')
plt.plot(results['year'], results['lat_R2'], marker='s', label='Latitude')
plt.xlabel('Year')
plt.ylabel('$R^2$')
plt.title('Quality of interannual spatial scaling fits')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plot_file = os.path.join(PLOTS_DIR, 'interannual_scaling_R2_2001_2020.png')
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
plt.close()
