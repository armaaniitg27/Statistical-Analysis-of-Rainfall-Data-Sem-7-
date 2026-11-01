import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft, irfft
DATA_FOLDER = 'D:\\SBS project\\vikas'
PROJECT_FOLDER = 'D:\\SBS project\\sem 7'
CODE_FOLDER = os.path.join(PROJECT_FOLDER, 'chapter1', 'codes')
PLOT_FOLDER = os.path.join(PROJECT_FOLDER, 'chapter1', 'plots')
RESULT_FOLDER = os.path.join(PROJECT_FOLDER, 'chapter1', 'results')
os.makedirs(CODE_FOLDER, exist_ok=True)
os.makedirs(PLOT_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
YEARS = range(2001, 2021)
LAT_MIN = 18.05
LAT_MAX = 24.95
LON_MIN = 78.05
LON_MAX = 91.95
FIRST_FILE = 2
LAST_FILE = 8833
EXPECTED_TIME_POINTS = LAST_FILE - FIRST_FILE + 1
EXPECTED_LAT_POINTS = 70
EXPECTED_LON_POINTS = 140
EXPECTED_GRID_POINTS = EXPECTED_LAT_POINTS * EXPECTED_LON_POINTS
CHUNK_SIZE = 512
TARGET_DISTANCES = [50, 100, 200, 300, 500]
MEAN_LATITUDE = (LAT_MIN + LAT_MAX) / 2.0
KM_PER_DEGREE = 111.32
LAT_GRID_SPACING = 0.1
LON_GRID_SPACING = 0.1
LAT_DISTANCE_PER_LAG = KM_PER_DEGREE * LAT_GRID_SPACING
LON_DISTANCE_PER_LAG = KM_PER_DEGREE * np.cos(np.deg2rad(MEAN_LATITUDE)) * LON_GRID_SPACING

def calculate_e_folding_distance(distance, correlation):
    distance = np.asarray(distance, dtype=float)
    correlation = np.asarray(correlation, dtype=float)
    threshold = 1 / np.e
    below = np.where(correlation <= threshold)[0]
    if len(below) == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return distance[0]
    x1 = distance[i - 1]
    x2 = distance[i]
    y1 = correlation[i - 1]
    y2 = correlation[i]
    L = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
    return L

def load_year(year):
    year_folder = os.path.join(DATA_FOLDER, f'kcsv{year}')
    first_file = os.path.join(year_folder, 'ktest2.csv')
    if not os.path.exists(first_file):
        raise FileNotFoundError(f'Could not find:\n{first_file}')
    first_df = pd.read_csv(first_file, usecols=['lat', 'lon', 'precipitationCal'])
    lat_values = np.round(first_df['lat'].to_numpy(), 2)
    lon_values = np.round(first_df['lon'].to_numpy(), 2)
    common_mask = (lat_values >= LAT_MIN) & (lat_values <= LAT_MAX) & (lon_values >= LON_MIN) & (lon_values <= LON_MAX)
    common_count = np.sum(common_mask)
    if common_count != EXPECTED_GRID_POINTS:
        raise ValueError(f'Expected {EXPECTED_GRID_POINTS} common grid points, but found {common_count} in {year}.')
    common_lat = lat_values[common_mask]
    common_lon = lon_values[common_mask]
    unique_lat = np.unique(common_lat)
    unique_lon = np.unique(common_lon)
    if len(unique_lat) != EXPECTED_LAT_POINTS:
        raise ValueError('Unexpected number of latitude points.')
    if len(unique_lon) != EXPECTED_LON_POINTS:
        raise ValueError('Unexpected number of longitude points.')
    rainfall = np.empty((EXPECTED_TIME_POINTS, EXPECTED_LAT_POINTS, EXPECTED_LON_POINTS), dtype=np.float32)
    first_rainfall = first_df['precipitationCal'].to_numpy(dtype=np.float32)
    rainfall[0] = first_rainfall[common_mask].reshape(EXPECTED_LAT_POINTS, EXPECTED_LON_POINTS)
    for index, file_number in enumerate(range(FIRST_FILE + 1, LAST_FILE + 1), start=1):
        filename = f'ktest{file_number}.csv'
        filepath = os.path.join(year_folder, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f'Missing file:\n{filepath}')
        df = pd.read_csv(filepath, usecols=['precipitationCal'])
        values = df['precipitationCal'].to_numpy(dtype=np.float32)
        if len(values) < EXPECTED_GRID_POINTS:
            raise ValueError(f'Unexpected number of rows in {filepath}')
        rainfall[index] = values[common_mask].reshape(EXPECTED_LAT_POINTS, EXPECTED_LON_POINTS)
        if (index + 1) % 500 == 0 or index == EXPECTED_TIME_POINTS - 1:
            pass
    if not np.isfinite(rainfall).all():
        raise ValueError(f'NaN or infinite rainfall values found in {year}.')
    return rainfall

def standardize_rainfall(rainfall):
    mean = np.mean(rainfall, axis=0, dtype=np.float64)
    std = np.std(rainfall, axis=0, ddof=1, dtype=np.float64)
    zero_std = std <= 0
    zero_count = np.sum(zero_std)
    if zero_count > 0:
        raise ValueError('At least one grid cell has zero temporal variance. Pearson correlation cannot be calculated for that cell.')
    mean = mean.astype(np.float32)
    std = std.astype(np.float32)
    return (mean, std)

def calculate_spatial_correlation(rainfall, mean, std, direction):
    time_points, n_lat, n_lon = rainfall.shape
    if direction == 'longitude':
        spatial_size = n_lon
        fft_axis = 2
        number_of_lines = n_lat
        distance_per_lag = LON_DISTANCE_PER_LAG
    elif direction == 'latitude':
        spatial_size = n_lat
        fft_axis = 1
        number_of_lines = n_lon
        distance_per_lag = LAT_DISTANCE_PER_LAG
    else:
        raise ValueError("Direction must be 'longitude' or 'latitude'.")
    number_of_lags = spatial_size - 1
    autocorrelation_sum = np.zeros(number_of_lags, dtype=np.float64)
    for start in range(0, time_points, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, time_points)
        z = (rainfall[start:end] - mean) / std
        fft_length = 2 * spatial_size
        F = rfft(z, n=fft_length, axis=fft_axis, workers=-1)
        power = F * np.conjugate(F)
        autocorrelation = irfft(power, n=fft_length, axis=fft_axis, workers=-1)
        if direction == 'longitude':
            lag_values = autocorrelation[:, :, 1:spatial_size]
            chunk_sum = np.sum(lag_values, axis=(0, 1), dtype=np.float64)
        else:
            lag_values = autocorrelation[:, 1:spatial_size, :]
            chunk_sum = np.sum(lag_values, axis=(0, 2), dtype=np.float64)
        autocorrelation_sum += chunk_sum
        del z
        del F
        del power
        del autocorrelation
    lags = np.arange(1, spatial_size)
    number_of_pairs = number_of_lines * (spatial_size - lags)
    correlation = autocorrelation_sum / ((time_points - 1) * number_of_pairs)
    distance = lags * distance_per_lag
    return (lags, distance, correlation, number_of_pairs)

def correlation_at_target(distance, correlation, target):
    index = np.argmin(np.abs(distance - target))
    return (distance[index], correlation[index])
summary_rows = []
for year in YEARS:
    rainfall = load_year(year)
    mean, std = standardize_rainfall(rainfall)
    lon_lags, lon_distance, lon_correlation, lon_pairs = calculate_spatial_correlation(rainfall, mean, std, 'longitude')
    lat_lags, lat_distance, lat_correlation, lat_pairs = calculate_spatial_correlation(rainfall, mean, std, 'latitude')
    lon_L = calculate_e_folding_distance(lon_distance, lon_correlation)
    lat_L = calculate_e_folding_distance(lat_distance, lat_correlation)
    if np.isfinite(lon_L) and np.isfinite(lat_L) and (lat_L != 0):
        anisotropy_ratio = lon_L / lat_L
    else:
        anisotropy_ratio = np.nan
    max_lags = max(len(lon_lags), len(lat_lags))
    correlation_table = pd.DataFrame({'longitude_lag': pd.Series(lon_lags), 'longitude_distance_km': pd.Series(lon_distance), 'longitude_mean_correlation': pd.Series(lon_correlation), 'longitude_number_of_pairs': pd.Series(lon_pairs), 'latitude_lag': pd.Series(lat_lags), 'latitude_distance_km': pd.Series(lat_distance), 'latitude_mean_correlation': pd.Series(lat_correlation), 'latitude_number_of_pairs': pd.Series(lat_pairs)})
    correlation_file = os.path.join(RESULT_FOLDER, f'spatial_correlation_{year}.csv')
    correlation_table.to_csv(correlation_file, index=False)
    selected_values = {}
    for target in TARGET_DISTANCES:
        actual_distance, corr = correlation_at_target(lon_distance, lon_correlation, target)
        selected_values[f'longitude_distance_{target}km'] = actual_distance
        selected_values[f'longitude_correlation_{target}km'] = corr
        actual_distance, corr = correlation_at_target(lat_distance, lat_correlation, target)
        selected_values[f'latitude_distance_{target}km'] = actual_distance
        selected_values[f'latitude_correlation_{target}km'] = corr
    row = {'year': year, 'longitude_correlation_length_km': lon_L, 'latitude_correlation_length_km': lat_L, 'longitude_latitude_ratio': anisotropy_ratio}
    row.update(selected_values)
    summary_rows.append(row)
    del rainfall
    del mean
    del std
summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values('year').reset_index(drop=True)
summary_file = os.path.join(RESULT_FOLDER, 'yearly_spatial_correlation_lengths_2001_2020.csv')
summary_df.to_csv(summary_file, index=False)
plt.figure(figsize=(10, 6))
plt.plot(summary_df['year'], summary_df['longitude_correlation_length_km'], marker='o')
plt.xlabel('Year')
plt.ylabel('Longitude Correlation Length (km)')
plt.title('Longitudinal Rainfall Correlation Length — 2001–2020')
plt.grid(True, alpha=0.3)
plt.xticks(summary_df['year'], rotation=45)
plt.tight_layout()
lon_length_plot = os.path.join(PLOT_FOLDER, 'longitude_correlation_length_2001_2020.png')
plt.savefig(lon_length_plot, dpi=300, bbox_inches='tight')
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(summary_df['year'], summary_df['latitude_correlation_length_km'], marker='o')
plt.xlabel('Year')
plt.ylabel('Latitude Correlation Length (km)')
plt.title('Latitudinal Rainfall Correlation Length — 2001–2020')
plt.grid(True, alpha=0.3)
plt.xticks(summary_df['year'], rotation=45)
plt.tight_layout()
lat_length_plot = os.path.join(PLOT_FOLDER, 'latitude_correlation_length_2001_2020.png')
plt.savefig(lat_length_plot, dpi=300, bbox_inches='tight')
plt.show()
plt.figure(figsize=(10, 6))
plt.plot(summary_df['year'], summary_df['longitude_latitude_ratio'], marker='o')
plt.axhline(1.0, linestyle='--', label='Ratio = 1')
plt.xlabel('Year')
plt.ylabel('Longitude / Latitude Correlation Length')
plt.title('Spatial Correlation Anisotropy — 2001–2020')
plt.grid(True, alpha=0.3)
plt.xticks(summary_df['year'], rotation=45)
plt.legend()
plt.tight_layout()
ratio_plot = os.path.join(PLOT_FOLDER, 'spatial_correlation_anisotropy_2001_2020.png')
plt.savefig(ratio_plot, dpi=300, bbox_inches='tight')
plt.show()
