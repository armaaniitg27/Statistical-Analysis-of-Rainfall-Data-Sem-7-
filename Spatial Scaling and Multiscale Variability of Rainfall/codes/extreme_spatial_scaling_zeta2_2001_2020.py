import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
DATA_ROOT = 'D:\\SBS project\\vikas'
RESULTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\results'
PLOTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\plots'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
YEARS = range(2001, 2021)
LAT_MIN = 18.05
LAT_MAX = 24.95
LON_MIN = 78.05
LON_MAX = 91.95
EXPECTED_CELLS = 70 * 140
LON_LAGS = np.arange(8, 14)
LAT_LAGS = np.arange(7, 13)

def load_year(year):
    year_dir = os.path.join(DATA_ROOT, f'kcsv{year}')
    data = []
    first_file = os.path.join(year_dir, 'ktest2.csv')
    first_df = pd.read_csv(first_file)
    first_df['lat'] = first_df['lat'].round(2)
    first_df['lon'] = first_df['lon'].round(2)
    first_df = first_df[(first_df['lat'] >= LAT_MIN) & (first_df['lat'] <= LAT_MAX) & (first_df['lon'] >= LON_MIN) & (first_df['lon'] <= LON_MAX)].copy()
    lat_values = np.sort(first_df['lat'].unique())
    lon_values = np.sort(first_df['lon'].unique())
    if len(lat_values) != 70:
        raise ValueError(f'Year {year}: expected 70 latitude cells, found {len(lat_values)}')
    if len(lon_values) != 140:
        raise ValueError(f'Year {year}: expected 140 longitude cells, found {len(lon_values)}')
    grid_index = pd.MultiIndex.from_product([lat_values, lon_values], names=['lat', 'lon'])
    for file_no in range(2, 8834):
        file_path = os.path.join(year_dir, f'ktest{file_no}.csv')
        df = pd.read_csv(file_path)
        df['lat'] = df['lat'].round(2)
        df['lon'] = df['lon'].round(2)
        df = df[(df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX) & (df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX)].copy()
        df = df.set_index(['lat', 'lon'])
        df = df.reindex(grid_index)
        values = df['precipitationCal'].to_numpy(dtype=float)
        data.append(values)
    data = np.asarray(data)
    missing = np.isnan(data).sum()
    if missing > 0:
        raise ValueError(f'Year {year}: {missing} missing grid values after alignment.')
    data = data.reshape(data.shape[0], 70, 140)
    return data

def get_distances():
    mean_lat = (LAT_MIN + LAT_MAX) / 2.0
    lon_spacing = 0.1 * 111.32 * np.cos(np.radians(mean_lat))
    lat_spacing = 0.1 * 111.32
    return (lon_spacing, lat_spacing)

def calculate_structure_function(data, direction, lags):
    results = []
    lon_spacing, lat_spacing = get_distances()
    for lag in lags:
        if direction == 'longitude':
            increments = data[:, :, lag:] - data[:, :, :-lag]
            distance = lag * lon_spacing
        elif direction == 'latitude':
            increments = data[:, lag:, :] - data[:, :-lag, :]
            distance = lag * lat_spacing
        else:
            raise ValueError('Invalid direction')
        increments = increments.ravel()
        increments = increments[np.isfinite(increments)]
        if len(increments) == 0:
            continue
        S2 = np.mean(np.abs(increments) ** 2)
        results.append({'lag': lag, 'distance_km': distance, 'number_of_increments': len(increments), 'S2': S2})
    return pd.DataFrame(results)

def fit_zeta2(df):
    x = np.log(df['distance_km'].to_numpy())
    y = np.log(df['S2'].to_numpy())
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return (slope, intercept, r2)
all_results = []
for year in YEARS:
    data = load_year(year)
    domain_mean = np.mean(data, axis=(1, 2))
    threshold = np.percentile(domain_mean, 95)
    extreme_mask = domain_mean >= threshold
    n_extreme = np.sum(extreme_mask)
    extreme_data = data[extreme_mask]
    lon_sf = calculate_structure_function(extreme_data, 'longitude', LON_LAGS)
    lon_zeta, lon_intercept, lon_r2 = fit_zeta2(lon_sf)
    lat_sf = calculate_structure_function(extreme_data, 'latitude', LAT_LAGS)
    lat_zeta, lat_intercept, lat_r2 = fit_zeta2(lat_sf)
    difference = lon_zeta - lat_zeta
    ratio = lon_zeta / lat_zeta if lat_zeta != 0 else np.nan
    all_results.append({'year': year, 'extreme_threshold': threshold, 'n_extreme': n_extreme, 'lon_zeta2_extreme': lon_zeta, 'lon_R2_extreme': lon_r2, 'lat_zeta2_extreme': lat_zeta, 'lat_R2_extreme': lat_r2, 'difference_extreme': difference, 'ratio_extreme': ratio})
results = pd.DataFrame(all_results)
complete_file = os.path.join(RESULTS_DIR, 'extreme_scaling_zeta2_2001_2020_COMPLETE.csv')
results.to_csv(complete_file, index=False)
summary_rows = []
for label, column in [('Longitude', 'lon_zeta2_extreme'), ('Latitude', 'lat_zeta2_extreme'), ('Difference', 'difference_extreme'), ('Ratio', 'ratio_extreme')]:
    values = results[column].to_numpy()
    summary_rows.append({'quantity': label, 'mean': np.mean(values), 'std': np.std(values, ddof=1), 'min': np.min(values), 'max': np.max(values)})
summary = pd.DataFrame(summary_rows)
summary_file = os.path.join(RESULTS_DIR, 'extreme_scaling_zeta2_summary_2001_2020.csv')
summary.to_csv(summary_file, index=False)
all_file = os.path.join(RESULTS_DIR, 'interannual_scaling_zeta2_2001_2020_COMPLETE.csv')
if not os.path.exists(all_file):
    raise FileNotFoundError('Could not find:\n' + all_file)
all_rain = pd.read_csv(all_file)
comparison = pd.DataFrame({'year': results['year'], 'lon_zeta2_all': all_rain['lon_zeta2'], 'lon_zeta2_extreme': results['lon_zeta2_extreme'], 'lat_zeta2_all': all_rain['lat_zeta2'], 'lat_zeta2_extreme': results['lat_zeta2_extreme']})
comparison['lon_change'] = comparison['lon_zeta2_extreme'] - comparison['lon_zeta2_all']
comparison['lat_change'] = comparison['lat_zeta2_extreme'] - comparison['lat_zeta2_all']
comparison['lon_percent_change'] = 100 * comparison['lon_change'] / comparison['lon_zeta2_all']
comparison['lat_percent_change'] = 100 * comparison['lat_change'] / comparison['lat_zeta2_all']
comparison['difference_all'] = comparison['lon_zeta2_all'] - comparison['lat_zeta2_all']
comparison['difference_extreme'] = comparison['lon_zeta2_extreme'] - comparison['lat_zeta2_extreme']
comparison['difference_change'] = comparison['difference_extreme'] - comparison['difference_all']
comparison['ratio_all'] = comparison['lon_zeta2_all'] / comparison['lat_zeta2_all']
comparison['ratio_extreme'] = comparison['lon_zeta2_extreme'] / comparison['lat_zeta2_extreme']
comparison_file = os.path.join(RESULTS_DIR, 'extreme_vs_all_scaling_zeta2_2001_2020.csv')
comparison.to_csv(comparison_file, index=False)
lon_t, lon_p = ttest_rel(comparison['lon_zeta2_extreme'], comparison['lon_zeta2_all'])
lat_t, lat_p = ttest_rel(comparison['lat_zeta2_extreme'], comparison['lat_zeta2_all'])
diff_t, diff_p = ttest_rel(comparison['difference_extreme'], comparison['difference_all'])
tests = pd.DataFrame([{'quantity': 'Longitude zeta2', 't_statistic': lon_t, 'p_value': lon_p}, {'quantity': 'Latitude zeta2', 't_statistic': lat_t, 'p_value': lat_p}, {'quantity': 'Directional difference', 't_statistic': diff_t, 'p_value': diff_p}])
tests_file = os.path.join(RESULTS_DIR, 'extreme_vs_all_scaling_paired_tests_2001_2020.csv')
tests.to_csv(tests_file, index=False)
mean_lon_all = comparison['lon_zeta2_all'].mean()
mean_lon_extreme = comparison['lon_zeta2_extreme'].mean()
mean_lat_all = comparison['lat_zeta2_all'].mean()
mean_lat_extreme = comparison['lat_zeta2_extreme'].mean()
lon_positive = np.sum(comparison['lon_change'] > 0)
lat_positive = np.sum(comparison['lat_change'] > 0)
difference_positive = np.sum(comparison['difference_change'] > 0)
plt.figure(figsize=(10, 6))
plt.plot(comparison['year'], comparison['lon_zeta2_all'], marker='o', label='All rainfall')
plt.plot(comparison['year'], comparison['lon_zeta2_extreme'], marker='s', label='Extreme rainfall')
plt.xlabel('Year')
plt.ylabel('$\\zeta_2$')
plt.title('Longitude Spatial Scaling: All vs Extreme Rainfall')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'extreme_vs_all_zeta2_longitude_2001_2020.png'), dpi=300)
plt.close()
plt.figure(figsize=(10, 6))
plt.plot(comparison['year'], comparison['lat_zeta2_all'], marker='o', label='All rainfall')
plt.plot(comparison['year'], comparison['lat_zeta2_extreme'], marker='s', label='Extreme rainfall')
plt.xlabel('Year')
plt.ylabel('$\\zeta_2$')
plt.title('Latitude Spatial Scaling: All vs Extreme Rainfall')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'extreme_vs_all_zeta2_latitude_2001_2020.png'), dpi=300)
plt.close()
plt.figure(figsize=(10, 6))
plt.axhline(0, linestyle='--', linewidth=1)
plt.plot(comparison['year'], comparison['lon_change'], marker='o', label='$\\Delta\\zeta_2$ longitude')
plt.plot(comparison['year'], comparison['lat_change'], marker='s', label='$\\Delta\\zeta_2$ latitude')
plt.xlabel('Year')
plt.ylabel('$\\zeta_{2,\\mathrm{extreme}}-\\zeta_{2,\\mathrm{all}}$')
plt.title('Change in Spatial Scaling Under Extreme Rainfall')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'extreme_vs_all_zeta2_change_2001_2020.png'), dpi=300)
plt.close()
plt.figure(figsize=(10, 6))
plt.axhline(0, linestyle='--', linewidth=1)
plt.plot(comparison['year'], comparison['difference_all'], marker='o', label='All rainfall')
plt.plot(comparison['year'], comparison['difference_extreme'], marker='s', label='Extreme rainfall')
plt.xlabel('Year')
plt.ylabel('$\\zeta_{\\mathrm{lon}}(2)-\\zeta_{\\mathrm{lat}}(2)$')
plt.title('Directional Scaling Difference: All vs Extreme Rainfall')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'extreme_vs_all_directional_difference_2001_2020.png'), dpi=300)
plt.close()
