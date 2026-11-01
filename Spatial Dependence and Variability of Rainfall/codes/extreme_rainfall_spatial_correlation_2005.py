import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_folder = 'D:\\SBS project\\vikas'
plot_folder = 'D:\\SBS project\\sem 7\\chapter1\\plots'
result_folder = 'D:\\SBS project\\sem 7\\chapter1\\results'
os.makedirs(plot_folder, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)
YEAR = 2005
NUM_FILES = 8832
WET_THRESHOLD = 0.1
EXTREME_PERCENTILE = 95
MAX_LAG = 50
LAT_MIN = 18.05
LAT_MAX = 24.95
LON_MIN = 78.05
LON_MAX = 91.95
mean_latitude = (LAT_MIN + LAT_MAX) / 2
LAT_SPACING_KM = 111.32 * 0.1
LON_SPACING_KM = 111.32 * np.cos(np.radians(mean_latitude)) * 0.1
THRESHOLD_1E = 1 / np.e

def load_year(year):
    year_folder = os.path.join(data_folder, f'kcsv{year}')
    first_file = os.path.join(year_folder, 'ktest2.csv')
    first_df = pd.read_csv(first_file)
    first_df['lat'] = first_df['lat'].round(2)
    first_df['lon'] = first_df['lon'].round(2)
    lats = np.sort(first_df['lat'].unique())
    lons = np.sort(first_df['lon'].unique())
    lat_mask = (lats >= LAT_MIN - 1e-06) & (lats <= LAT_MAX + 1e-06)
    lon_mask = (lons >= LON_MIN - 1e-06) & (lons <= LON_MAX + 1e-06)
    common_lats = lats[lat_mask]
    common_lons = lons[lon_mask]
    nlat = len(common_lats)
    nlon = len(common_lons)
    expected_cells = nlat * nlon
    rainfall = np.empty((NUM_FILES, nlat, nlon), dtype=np.float32)
    times = []
    start_load = time.time()
    for i in range(NUM_FILES):
        file_number = i + 2
        filepath = os.path.join(year_folder, f'ktest{file_number}.csv')
        df = pd.read_csv(filepath)
        df['lat'] = df['lat'].round(2)
        df['lon'] = df['lon'].round(2)
        df = df[(df['lat'] >= LAT_MIN - 1e-06) & (df['lat'] <= LAT_MAX + 1e-06) & (df['lon'] >= LON_MIN - 1e-06) & (df['lon'] <= LON_MAX + 1e-06)]
        if len(df) != expected_cells:
            raise ValueError(f'Unexpected number of cells in {filepath}: {len(df)} instead of {expected_cells}')
        rainfall[i] = df['precipitationCal'].to_numpy(dtype=np.float32).reshape(nlat, nlon)
        times.append(df['time'].iloc[0])
        if (i + 1) % 1000 == 0:
            elapsed = time.time() - start_load
    times = pd.to_datetime(times, format='mixed')
    return (rainfall, times, common_lats, common_lons)

def spatial_correlations(data, max_lag):
    """
    Calculate mean spatial Pearson correlation
    for each spatial lag.

    data shape:
        time x latitude x longitude
    """
    ntime, nlat, nlon = data.shape
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, ddof=1, keepdims=True)
    std[std == 0] = np.nan
    standardized = (data - mean) / std
    longitude_results = []
    latitude_results = []
    for lag in range(1, max_lag + 1):
        if lag >= nlon:
            break
        x = standardized[:, :, :-lag]
        y = standardized[:, :, lag:]
        corr = np.nanmean(x * y, axis=0) * ntime / (ntime - 1)
        mean_corr = np.nanmean(corr)
        longitude_results.append({'lag': lag, 'distance_km': lag * LON_SPACING_KM, 'mean_correlation': mean_corr, 'number_of_pairs': nlat * (nlon - lag)})
    for lag in range(1, max_lag + 1):
        if lag >= nlat:
            break
        x = standardized[:, :-lag, :]
        y = standardized[:, lag:, :]
        corr = np.nanmean(x * y, axis=0) * ntime / (ntime - 1)
        mean_corr = np.nanmean(corr)
        latitude_results.append({'lag': lag, 'distance_km': lag * LAT_SPACING_KM, 'mean_correlation': mean_corr, 'number_of_pairs': (nlat - lag) * nlon})
    return (pd.DataFrame(longitude_results), pd.DataFrame(latitude_results))

def correlation_length(df):
    distance = df['distance_km'].to_numpy()
    correlation = df['mean_correlation'].to_numpy()
    threshold = THRESHOLD_1E
    for i in range(1, len(correlation)):
        if correlation[i - 1] >= threshold and correlation[i] <= threshold:
            d1 = distance[i - 1]
            d2 = distance[i]
            r1 = correlation[i - 1]
            r2 = correlation[i]
            L = d1 + (threshold - r1) * (d2 - d1) / (r2 - r1)
            return L
    return np.nan
script_start = time.time()
rainfall, times, lats, lons = load_year(YEAR)
domain_mean = np.mean(rainfall, axis=(1, 2))
extreme_threshold = np.percentile(domain_mean, EXTREME_PERCENTILE)
extreme_mask = domain_mean >= extreme_threshold
extreme_count = np.sum(extreme_mask)
extreme_rainfall = rainfall[extreme_mask]
extreme_times = times[extreme_mask]
start_corr = time.time()
all_lon, all_lat = spatial_correlations(rainfall, MAX_LAG)
all_lon_length = correlation_length(all_lon)
all_lat_length = correlation_length(all_lat)
extreme_lon, extreme_lat = spatial_correlations(extreme_rainfall, MAX_LAG)
extreme_lon_length = correlation_length(extreme_lon)
extreme_lat_length = correlation_length(extreme_lat)
all_ratio = all_lon_length / all_lat_length
extreme_ratio = extreme_lon_length / extreme_lat_length
summary = pd.DataFrame({'year': [YEAR], 'extreme_percentile': [EXTREME_PERCENTILE], 'extreme_threshold': [extreme_threshold], 'extreme_time_steps': [extreme_count], 'extreme_percentage': [100 * extreme_count / NUM_FILES], 'all_L_lon_km': [all_lon_length], 'all_L_lat_km': [all_lat_length], 'all_anisotropy': [all_ratio], 'extreme_L_lon_km': [extreme_lon_length], 'extreme_L_lat_km': [extreme_lat_length], 'extreme_anisotropy': [extreme_ratio], 'extreme_to_all_L_lon': [extreme_lon_length / all_lon_length], 'extreme_to_all_L_lat': [extreme_lat_length / all_lat_length]})
summary_path = os.path.join(result_folder, 'extreme_rainfall_spatial_correlation_2005.csv')
summary.to_csv(summary_path, index=False)
curve_path = os.path.join(result_folder, 'extreme_rainfall_correlation_curves_2005.csv')
all_lon['type'] = 'All'
all_lat['type'] = 'All'
all_lon['direction'] = 'Longitude'
all_lat['direction'] = 'Latitude'
extreme_lon['type'] = 'Extreme'
extreme_lat['type'] = 'Extreme'
extreme_lon['direction'] = 'Longitude'
extreme_lat['direction'] = 'Latitude'
curves = pd.concat([all_lon, all_lat, extreme_lon, extreme_lat], ignore_index=True)
curves.to_csv(curve_path, index=False)
plt.figure(figsize=(9, 6))
plt.plot(all_lon['distance_km'], all_lon['mean_correlation'], marker='o', label='All rainfall')
plt.plot(extreme_lon['distance_km'], extreme_lon['mean_correlation'], marker='s', label='Extreme rainfall')
plt.axhline(THRESHOLD_1E, linestyle='--', label='1/e')
plt.xlabel('Distance (km)')
plt.ylabel('Mean Pearson correlation')
plt.title('Longitudinal Spatial Correlation: All vs Extreme Rainfall - 2005')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'extreme_vs_all_longitudinal_correlation_2005.png'), dpi=300)
plt.close()
plt.figure(figsize=(9, 6))
plt.plot(all_lat['distance_km'], all_lat['mean_correlation'], marker='o', label='All rainfall')
plt.plot(extreme_lat['distance_km'], extreme_lat['mean_correlation'], marker='s', label='Extreme rainfall')
plt.axhline(THRESHOLD_1E, linestyle='--', label='1/e')
plt.xlabel('Distance (km)')
plt.ylabel('Mean Pearson correlation')
plt.title('Latitudinal Spatial Correlation: All vs Extreme Rainfall - 2005')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'extreme_vs_all_latitudinal_correlation_2005.png'), dpi=300)
plt.close()
