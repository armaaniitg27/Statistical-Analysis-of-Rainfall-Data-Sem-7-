import os
import numpy as np
import pandas as pd
DATA_DIR = 'D:\\SBS project\\vikas\\kcsv2005'
RESULTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\results'
os.makedirs(RESULTS_DIR, exist_ok=True)
MAX_LAG = 50
files = [os.path.join(DATA_DIR, f'ktest{i}.csv') for i in range(2, 8834)]
rainfall_list = []
latitudes = None
longitudes = None
for idx, file in enumerate(files):
    df = pd.read_csv(file)
    if latitudes is None:
        latitudes = np.sort(df['lat'].unique())
        longitudes = np.sort(df['lon'].unique())
    df = df.sort_values(['lat', 'lon'])
    rainfall = df['precipitationCal'].to_numpy(dtype=np.float32)
    rainfall_list.append(rainfall)
rainfall = np.asarray(rainfall_list, dtype=np.float32)
n_times, n_grid = rainfall.shape
n_lat = len(latitudes)
n_lon = len(longitudes)
assert n_times == 8832
assert n_lat == 70
assert n_lon == 140
assert n_grid == 9800
rainfall = rainfall.reshape(n_times, n_lat, n_lon)
mean_latitude = np.mean(latitudes)
KM_PER_DEGREE = 111.32
lon_spacing_km = 0.1 * KM_PER_DEGREE * np.cos(np.deg2rad(mean_latitude))
lat_spacing_km = 0.1 * KM_PER_DEGREE
longitude_results = []
latitude_results = []
for k in range(1, MAX_LAG + 1):
    delta_lon = rainfall[:, :, k:] - rainfall[:, :, :-k]
    delta_lon_flat = delta_lon.ravel()
    longitude_results.append({'lag': k, 'distance_km': k * lon_spacing_km, 'number_of_increments': len(delta_lon_flat), 'mean_increment': np.mean(delta_lon_flat), 'std_increment': np.std(delta_lon_flat), 'mean_absolute_increment': np.mean(np.abs(delta_lon_flat)), 'median_absolute_increment': np.median(np.abs(delta_lon_flat))})
    delta_lat = rainfall[:, k:, :] - rainfall[:, :-k, :]
    delta_lat_flat = delta_lat.ravel()
    latitude_results.append({'lag': k, 'distance_km': k * lat_spacing_km, 'number_of_increments': len(delta_lat_flat), 'mean_increment': np.mean(delta_lat_flat), 'std_increment': np.std(delta_lat_flat), 'mean_absolute_increment': np.mean(np.abs(delta_lat_flat)), 'median_absolute_increment': np.median(np.abs(delta_lat_flat))})
longitude_results = pd.DataFrame(longitude_results)
latitude_results = pd.DataFrame(latitude_results)
lon_output = os.path.join(RESULTS_DIR, 'spatial_increments_longitude_2005.csv')
lat_output = os.path.join(RESULTS_DIR, 'spatial_increments_latitude_2005.csv')
longitude_results.to_csv(lon_output, index=False)
latitude_results.to_csv(lat_output, index=False)
pd.set_option('display.max_rows', 60)
pd.set_option('display.float_format', '{:.6f}'.format)
