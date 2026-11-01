import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.fft import rfft, irfft
data_folder = 'D:\\SBS project\\vikas'
code_folder = 'D:\\SBS project\\sem 7\\chapter1\\codes'
plot_folder = 'D:\\SBS project\\sem 7\\chapter1\\plots'
result_folder = 'D:\\SBS project\\sem 7\\chapter1\\results'
os.makedirs(code_folder, exist_ok=True)
os.makedirs(plot_folder, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)
years = range(2001, 2021)
files_per_year = 8832
LAT_MIN = 18.05
LAT_MAX = 24.95
LON_MIN = 78.05
LON_MAX = 91.95
MAX_LAG = 50
LAT_STEP_KM = 111.32 * 0.1
MEAN_LATITUDE = (LAT_MIN + LAT_MAX) / 2.0
LON_STEP_KM = 111.32 * np.cos(np.deg2rad(MEAN_LATITUDE)) * 0.1
E_INV = 1.0 / np.e
CHUNK_SIZE = 256
seasons = {'May': [5], 'Monsoon_Jun_Sep': [6, 7, 8, 9], 'October': [10]}

def correlation_length(lags, correlations, distances):
    """
    Determine the correlation length from:

        rho(L) = 1/e

    using linear interpolation between the two points
    surrounding 1/e.
    """
    target = E_INV
    x = np.asarray(distances)
    y = np.asarray(correlations)
    below = np.where(y <= target)[0]
    if len(below) == 0:
        return np.nan
    i = below[0]
    if i == 0:
        return x[0]
    x1 = x[i - 1]
    x2 = x[i]
    y1 = y[i - 1]
    y2 = y[i]
    if y2 == y1:
        return x2
    L = x1 + (target - y1) * (x2 - x1) / (y2 - y1)
    return L

def spatial_correlations_fft(data, max_lag=50):
    """
    Calculate mean Pearson spatial correlation as a function
    of spatial lag.

    Input:
        data = (time, latitude, longitude)

    The time series at every grid cell is standardized over
    the selected seasonal period.

    Correlations are calculated separately along longitude
    and latitude.
    """
    T, H, W = data.shape
    mean = np.mean(data, axis=0, dtype=np.float64)
    std = np.std(data, axis=0, ddof=1, dtype=np.float64)
    std[std == 0] = np.nan
    z = (data.astype(np.float64) - mean) / std
    z = np.nan_to_num(z, nan=0.0)
    n = W
    fft_length = 2 * n
    correlation_sum_lon = np.zeros(n, dtype=np.float64)
    for start in range(0, T, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, T)
        chunk = z[start:end, :, :]
        F = rfft(chunk, n=fft_length, axis=2)
        power = F * np.conj(F)
        corr = irfft(power, n=fft_length, axis=2)
        correlation_sum_lon += np.sum(corr[:, :, :n], axis=(0, 1))
    lon_correlations = []
    max_lon_lag = min(max_lag, W - 1)
    for lag in range(max_lon_lag + 1):
        total_pairs = (T - 1) * H * (W - lag)
        value = correlation_sum_lon[lag] / total_pairs
        lon_correlations.append(value)
    lon_correlations = np.asarray(lon_correlations)
    n = H
    fft_length = 2 * n
    correlation_sum_lat = np.zeros(n, dtype=np.float64)
    for start in range(0, T, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, T)
        chunk = z[start:end, :, :]
        F = rfft(chunk, n=fft_length, axis=1)
        power = F * np.conj(F)
        corr = irfft(power, n=fft_length, axis=1)
        correlation_sum_lat += np.sum(corr[:, :n, :], axis=(0, 2))
    lat_correlations = []
    max_lat_lag = min(max_lag, H - 1)
    for lag in range(max_lat_lag + 1):
        total_pairs = (T - 1) * W * (H - lag)
        value = correlation_sum_lat[lag] / total_pairs
        lat_correlations.append(value)
    lat_correlations = np.asarray(lat_correlations)
    return (lon_correlations, lat_correlations)

def load_year(year):
    """
    Load all 8832 half-hourly rainfall fields for one year.

    Only the common 70 x 140 grid is retained.

    Coordinates are rounded to two decimal places before
    applying the geographic boundaries.
    """
    year_folder = os.path.join(data_folder, f'kcsv{year}')
    first_file = os.path.join(year_folder, 'ktest2.csv')
    first_df = pd.read_csv(first_file)
    first_df['lat'] = first_df['lat'].round(2)
    first_df['lon'] = first_df['lon'].round(2)
    mask = (first_df['lat'] >= LAT_MIN) & (first_df['lat'] <= LAT_MAX) & (first_df['lon'] >= LON_MIN) & (first_df['lon'] <= LON_MAX)
    grid_df = first_df.loc[mask, ['lat', 'lon']].copy()
    grid_lat = np.sort(grid_df['lat'].unique())
    grid_lon = np.sort(grid_df['lon'].unique())
    H = len(grid_lat)
    W = len(grid_lon)
    expected_cells = H * W
    if H != 70 or W != 140:
        raise ValueError(f'Unexpected common grid for {year}: {H} x {W} = {expected_cells}. Expected 70 x 140 = 9800.')
    rainfall = np.zeros((files_per_year, H, W), dtype=np.float32)
    for i in range(files_per_year):
        file_number = i + 2
        file_path = os.path.join(year_folder, f'ktest{file_number}.csv')
        df = pd.read_csv(file_path)
        df['lat'] = df['lat'].round(2)
        df['lon'] = df['lon'].round(2)
        sub = df.loc[(df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX) & (df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX), ['lat', 'lon', 'precipitationCal']]
        if len(sub) != expected_cells:
            raise ValueError(f'{file_path} has {len(sub)} cells; expected {expected_cells}')
        sub = sub.sort_values(['lat', 'lon'])
        rainfall[i] = sub['precipitationCal'].to_numpy(dtype=np.float32).reshape(H, W)
        if (i + 1) % 500 == 0:
            pass
    return rainfall

def analyze_season(rainfall, year, season_name, months, times):
    month_mask = times.month.isin(months)
    season_data = rainfall[month_mask]
    n_time = season_data.shape[0]
    if n_time < 10:
        return None
    lon_corr, lat_corr = spatial_correlations_fft(season_data, max_lag=MAX_LAG)
    lon_lags = np.arange(len(lon_corr))
    lat_lags = np.arange(len(lat_corr))
    lon_distances = lon_lags * LON_STEP_KM
    lat_distances = lat_lags * LAT_STEP_KM
    lon_corr[0] = 1.0
    lat_corr[0] = 1.0
    L_lon = correlation_length(lon_lags, lon_corr, lon_distances)
    L_lat = correlation_length(lat_lags, lat_corr, lat_distances)
    if np.isfinite(L_lon) and np.isfinite(L_lat):
        ratio = L_lon / L_lat
    else:
        ratio = np.nan
    lon_curve_df = pd.DataFrame({'year': year, 'season': season_name, 'longitude_lag': lon_lags, 'longitude_distance_km': lon_distances, 'longitude_correlation': lon_corr})
    lon_curve_path = os.path.join(result_folder, f'spatial_correlation_longitude_{season_name}_{year}.csv')
    lon_curve_df.to_csv(lon_curve_path, index=False)
    lat_curve_df = pd.DataFrame({'year': year, 'season': season_name, 'latitude_lag': lat_lags, 'latitude_distance_km': lat_distances, 'latitude_correlation': lat_corr})
    lat_curve_path = os.path.join(result_folder, f'spatial_correlation_latitude_{season_name}_{year}.csv')
    lat_curve_df.to_csv(lat_curve_path, index=False)
    return {'year': year, 'season': season_name, 'n_time_points': n_time, 'L_longitude_km': L_lon, 'L_latitude_km': L_lat, 'anisotropy_ratio': ratio}
all_results = []
for year in years:
    rainfall = load_year(year)
    times = pd.date_range(start=f'{year}-05-01 00:00:00', periods=files_per_year, freq='30min')
    for season_name, months in seasons.items():
        result = analyze_season(rainfall=rainfall, year=year, season_name=season_name, months=months, times=times)
        if result is not None:
            all_results.append(result)
    del rainfall
results_df = pd.DataFrame(all_results)
results_path = os.path.join(result_folder, 'seasonal_spatial_correlation_lengths_2001_2020.csv')
results_df.to_csv(results_path, index=False)
summary = results_df.groupby('season').agg(mean_L_longitude_km=('L_longitude_km', 'mean'), std_L_longitude_km=('L_longitude_km', 'std'), mean_L_latitude_km=('L_latitude_km', 'mean'), std_L_latitude_km=('L_latitude_km', 'std'), mean_anisotropy_ratio=('anisotropy_ratio', 'mean'), std_anisotropy_ratio=('anisotropy_ratio', 'std')).reset_index()
summary_path = os.path.join(result_folder, 'seasonal_spatial_correlation_summary_2001_2020.csv')
summary.to_csv(summary_path, index=False)
plt.figure(figsize=(14, 8))
for season in seasons.keys():
    temp = results_df[results_df['season'] == season]
    plt.plot(temp['year'], temp['L_longitude_km'], marker='o', label=season)
plt.xlabel('Year')
plt.ylabel('Longitude Correlation Length (km)')
plt.title('Seasonal Longitudinal Rainfall Correlation Length — 2001–2020')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plot_path = os.path.join(plot_folder, 'seasonal_longitudinal_correlation_length_2001_2020.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
plt.figure(figsize=(14, 8))
for season in seasons.keys():
    temp = results_df[results_df['season'] == season]
    plt.plot(temp['year'], temp['L_latitude_km'], marker='o', label=season)
plt.xlabel('Year')
plt.ylabel('Latitude Correlation Length (km)')
plt.title('Seasonal Latitudinal Rainfall Correlation Length — 2001–2020')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plot_path = os.path.join(plot_folder, 'seasonal_latitudinal_correlation_length_2001_2020.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
plt.figure(figsize=(14, 8))
for season in seasons.keys():
    temp = results_df[results_df['season'] == season]
    plt.plot(temp['year'], temp['anisotropy_ratio'], marker='o', label=season)
plt.axhline(1.0, linestyle='--', label='Ratio = 1')
plt.xlabel('Year')
plt.ylabel('Anisotropy Ratio, $L_{\\mathrm{lon}}/L_{\\mathrm{lat}}$')
plt.title('Seasonal Spatial Correlation Anisotropy — 2001–2020')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plot_path = os.path.join(plot_folder, 'seasonal_spatial_correlation_anisotropy_2001_2020.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
season_order = ['May', 'Monsoon_Jun_Sep', 'October']
plot_summary = summary.set_index('season').loc[season_order]
x = np.arange(len(season_order))
width = 0.35
plt.figure(figsize=(11, 7))
plt.bar(x - width / 2, plot_summary['mean_L_longitude_km'], width, label='Longitude')
plt.bar(x + width / 2, plot_summary['mean_L_latitude_km'], width, label='Latitude')
plt.xticks(x, ['May', 'June–September', 'October'])
plt.ylabel('Mean Correlation Length (km)')
plt.xlabel('Season')
plt.title('Mean Seasonal Rainfall Correlation Length — 2001–2020')
plt.grid(axis='y', alpha=0.3)
plt.legend()
plt.tight_layout()
plot_path = os.path.join(plot_folder, 'mean_seasonal_spatial_correlation_length_2001_2020.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
