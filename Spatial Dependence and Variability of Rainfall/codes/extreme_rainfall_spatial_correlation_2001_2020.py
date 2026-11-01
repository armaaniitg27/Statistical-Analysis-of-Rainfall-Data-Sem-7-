import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data_folder = 'D:\\SBS project\\vikas'
code_folder = 'D:\\SBS project\\sem 7\\chapter1\\codes'
plot_folder = 'D:\\SBS project\\sem 7\\chapter1\\plots'
result_folder = 'D:\\SBS project\\sem 7\\chapter1\\results'
os.makedirs(code_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)
YEARS = range(2001, 2021)
NUM_FILES = 8832
MAX_LAG = 50
EXTREME_PERCENTILE = 95
LAT_MIN = 18.05
LAT_MAX = 24.95
LON_MIN = 78.05
LON_MAX = 91.95
LAT_STEP_KM = 11.132
LON_STEP_KM = 10.357408
CORRELATION_THRESHOLD = 1 / np.e

def load_year(year):
    year_folder = os.path.join(data_folder, f'kcsv{year}')
    rainfall_list = []
    time_list = []
    expected_lat = None
    expected_lon = None
    start_time = time.time()
    for file_num in range(2, NUM_FILES + 2):
        file_path = os.path.join(year_folder, f'ktest{file_num}.csv')
        df = pd.read_csv(file_path)
        df['lat'] = df['lat'].round(2)
        df['lon'] = df['lon'].round(2)
        df = df[(df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX) & (df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX)]
        if expected_lat is None:
            expected_lat = np.sort(df['lat'].unique())
            expected_lon = np.sort(df['lon'].unique())
            nlat = len(expected_lat)
            nlon = len(expected_lon)
            if nlat != 70 or nlon != 140:
                raise ValueError(f'Unexpected common grid for {year}: {nlat} x {nlon}')
        if len(df) != nlat * nlon:
            raise ValueError(f'Unexpected number of rows in {file_path}: {len(df)}')
        df = df.sort_values(['lat', 'lon'])
        rainfall_list.append(df['precipitationCal'].to_numpy(dtype=np.float32))
        time_list.append(df['time'].iloc[0])
    rainfall = np.asarray(rainfall_list, dtype=np.float32)
    times = pd.to_datetime(time_list, format='mixed')
    elapsed = time.time() - start_time
    return (rainfall, times)

def spatial_correlation_lengths(rainfall, max_lag=MAX_LAG):
    """
    Calculate mean Pearson spatial correlation
    for longitudinal and latitudinal separations.

    rainfall shape:
        (time, latitude, longitude)

    Returns:
        L_lon
        L_lat
        longitude correlation curve
        latitude correlation curve
    """
    nt, nlat, nlon = rainfall.shape
    X = rainfall.reshape(nt, -1)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    valid = stds > 0
    X_standardized = np.zeros_like(X)
    X_standardized[:, valid] = (X[:, valid] - means[valid]) / stds[valid]
    X_standardized = X_standardized.reshape(nt, nlat, nlon)
    longitude_correlations = []
    latitude_correlations = []
    longitude_distances = []
    latitude_distances = []
    for lag in range(1, max_lag + 1):
        if lag < nlon:
            A = X_standardized[:, :, :-lag]
            B = X_standardized[:, :, lag:]
            corr = np.mean(A * B, axis=0)
            valid_corr = np.isfinite(corr)
            if np.any(valid_corr):
                mean_corr = np.mean(corr[valid_corr])
                longitude_correlations.append(mean_corr)
                longitude_distances.append(lag * LON_STEP_KM)
        if lag < nlat:
            A = X_standardized[:, :-lag, :]
            B = X_standardized[:, lag:, :]
            corr = np.mean(A * B, axis=0)
            valid_corr = np.isfinite(corr)
            if np.any(valid_corr):
                mean_corr = np.mean(corr[valid_corr])
                latitude_correlations.append(mean_corr)
                latitude_distances.append(lag * LAT_STEP_KM)
    longitude_correlations = np.asarray(longitude_correlations)
    latitude_correlations = np.asarray(latitude_correlations)
    longitude_distances = np.asarray(longitude_distances)
    latitude_distances = np.asarray(latitude_distances)

    def get_correlation_length(distances, correlations):
        threshold = CORRELATION_THRESHOLD
        below = np.where(correlations <= threshold)[0]
        if len(below) == 0:
            return np.nan
        i = below[0]
        if i == 0:
            return distances[0]
        d1 = distances[i - 1]
        d2 = distances[i]
        r1 = correlations[i - 1]
        r2 = correlations[i]
        if r2 == r1:
            return d2
        L = d1 + (threshold - r1) / (r2 - r1) * (d2 - d1)
        return L
    L_lon = get_correlation_length(longitude_distances, longitude_correlations)
    L_lat = get_correlation_length(latitude_distances, latitude_correlations)
    return (L_lon, L_lat, longitude_distances, longitude_correlations, latitude_distances, latitude_correlations)
results = []
all_curve_rows = []
overall_start = time.time()
for year in YEARS:
    year_start = time.time()
    rainfall_flat, times = load_year(year)
    rainfall = rainfall_flat.reshape(NUM_FILES, 70, 140)
    domain_mean = np.mean(rainfall, axis=(1, 2))
    extreme_threshold = np.percentile(domain_mean, EXTREME_PERCENTILE)
    extreme_mask = domain_mean >= extreme_threshold
    extreme_count = int(np.sum(extreme_mask))
    extreme_percentage = 100 * extreme_count / len(domain_mean)
    L_lon_all, L_lat_all, d_lon_all, r_lon_all, d_lat_all, r_lat_all = spatial_correlation_lengths(rainfall, MAX_LAG)
    extreme_rainfall = rainfall[extreme_mask]
    L_lon_extreme, L_lat_extreme, d_lon_extreme, r_lon_extreme, d_lat_extreme, r_lat_extreme = spatial_correlation_lengths(extreme_rainfall, MAX_LAG)
    anisotropy_all = L_lon_all / L_lat_all if np.isfinite(L_lon_all) and np.isfinite(L_lat_all) else np.nan
    anisotropy_extreme = L_lon_extreme / L_lat_extreme if np.isfinite(L_lon_extreme) and np.isfinite(L_lat_extreme) else np.nan
    lon_ratio = L_lon_extreme / L_lon_all if np.isfinite(L_lon_extreme) and np.isfinite(L_lon_all) else np.nan
    lat_ratio = L_lat_extreme / L_lat_all if np.isfinite(L_lat_extreme) and np.isfinite(L_lat_all) else np.nan
    results.append({'year': year, 'extreme_threshold_95pct': extreme_threshold, 'extreme_time_steps': extreme_count, 'extreme_percentage': extreme_percentage, 'L_lon_all_km': L_lon_all, 'L_lat_all_km': L_lat_all, 'anisotropy_all': anisotropy_all, 'L_lon_extreme_km': L_lon_extreme, 'L_lat_extreme_km': L_lat_extreme, 'anisotropy_extreme': anisotropy_extreme, 'extreme_to_all_L_lon': lon_ratio, 'extreme_to_all_L_lat': lat_ratio})
    for d, r in zip(d_lon_all, r_lon_all):
        all_curve_rows.append({'year': year, 'rainfall_type': 'all', 'direction': 'longitude', 'distance_km': d, 'correlation': r})
    for d, r in zip(d_lat_all, r_lat_all):
        all_curve_rows.append({'year': year, 'rainfall_type': 'all', 'direction': 'latitude', 'distance_km': d, 'correlation': r})
    for d, r in zip(d_lon_extreme, r_lon_extreme):
        all_curve_rows.append({'year': year, 'rainfall_type': 'extreme', 'direction': 'longitude', 'distance_km': d, 'correlation': r})
    for d, r in zip(d_lat_extreme, r_lat_extreme):
        all_curve_rows.append({'year': year, 'rainfall_type': 'extreme', 'direction': 'latitude', 'distance_km': d, 'correlation': r})
results_df = pd.DataFrame(results)
curves_df = pd.DataFrame(all_curve_rows)
result_csv = os.path.join(result_folder, 'extreme_rainfall_spatial_correlation_2001_2020.csv')
curves_csv = os.path.join(result_folder, 'extreme_rainfall_correlation_curves_2001_2020.csv')
results_df.to_csv(result_csv, index=False)
curves_df.to_csv(curves_csv, index=False)
plt.figure(figsize=(10, 6))
plt.plot(results_df['year'], results_df['L_lon_all_km'], marker='o', label='All rainfall')
plt.plot(results_df['year'], results_df['L_lon_extreme_km'], marker='o', label='Extreme rainfall')
plt.xlabel('Year')
plt.ylabel('Longitudinal correlation length (km)')
plt.title('Longitudinal Spatial Correlation Length\nAll vs Extreme Rainfall (2001–2020)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'extreme_vs_all_longitudinal_correlation_length_2001_2020.png'), dpi=300)
plt.close()
plt.figure(figsize=(10, 6))
plt.plot(results_df['year'], results_df['L_lat_all_km'], marker='o', label='All rainfall')
plt.plot(results_df['year'], results_df['L_lat_extreme_km'], marker='o', label='Extreme rainfall')
plt.xlabel('Year')
plt.ylabel('Latitudinal correlation length (km)')
plt.title('Latitudinal Spatial Correlation Length\nAll vs Extreme Rainfall (2001–2020)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'extreme_vs_all_latitudinal_correlation_length_2001_2020.png'), dpi=300)
plt.close()
plt.figure(figsize=(10, 6))
plt.plot(results_df['year'], results_df['anisotropy_all'], marker='o', label='All rainfall')
plt.plot(results_df['year'], results_df['anisotropy_extreme'], marker='o', label='Extreme rainfall')
plt.axhline(1.0, linestyle='--', linewidth=1)
plt.xlabel('Year')
plt.ylabel('Anisotropy ratio $L_{\\mathrm{lon}}/L_{\\mathrm{lat}}$')
plt.title('Spatial Anisotropy\nAll vs Extreme Rainfall (2001–2020)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'extreme_vs_all_anisotropy_2001_2020.png'), dpi=300)
plt.close()
plt.figure(figsize=(10, 6))
plt.plot(results_df['year'], results_df['extreme_to_all_L_lon'], marker='o', label='Longitude')
plt.plot(results_df['year'], results_df['extreme_to_all_L_lat'], marker='o', label='Latitude')
plt.axhline(1.0, linestyle='--', linewidth=1)
plt.xlabel('Year')
plt.ylabel('Extreme / All correlation length')
plt.title('Change in Spatial Correlation Length\nDuring Extreme Rainfall (2001–2020)')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(plot_folder, 'extreme_to_all_correlation_length_ratio_2001_2020.png'), dpi=300)
plt.close()
total_time = time.time() - overall_start
