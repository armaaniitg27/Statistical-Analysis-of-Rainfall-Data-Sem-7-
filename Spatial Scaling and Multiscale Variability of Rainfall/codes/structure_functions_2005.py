import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
DATA_DIR = 'D:\\SBS project\\vikas\\kcsv2005'
RESULTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\results'
PLOTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\plots'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
MAX_LAG = 50
Q_VALUES = [1, 2, 3, 4, 5]
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
    abs_delta_lon = np.abs(delta_lon)
    lon_row = {'lag': k, 'distance_km': k * lon_spacing_km, 'number_of_increments': delta_lon.size}
    for q in Q_VALUES:
        S_q = np.mean(abs_delta_lon ** q)
        lon_row[f'S{q}'] = S_q
    longitude_results.append(lon_row)
    del delta_lon
    del abs_delta_lon
    delta_lat = rainfall[:, k:, :] - rainfall[:, :-k, :]
    abs_delta_lat = np.abs(delta_lat)
    lat_row = {'lag': k, 'distance_km': k * lat_spacing_km, 'number_of_increments': delta_lat.size}
    for q in Q_VALUES:
        S_q = np.mean(abs_delta_lat ** q)
        lat_row[f'S{q}'] = S_q
    latitude_results.append(lat_row)
    del delta_lat
    del abs_delta_lat
longitude_results = pd.DataFrame(longitude_results)
latitude_results = pd.DataFrame(latitude_results)
lon_output = os.path.join(RESULTS_DIR, 'structure_functions_longitude_2005.csv')
lat_output = os.path.join(RESULTS_DIR, 'structure_functions_latitude_2005.csv')
longitude_results.to_csv(lon_output, index=False)
latitude_results.to_csv(lat_output, index=False)
pd.set_option('display.max_rows', 60)
pd.set_option('display.float_format', '{:.8e}'.format)
plt.figure(figsize=(9, 6))
for q in Q_VALUES:
    plt.loglog(longitude_results['distance_km'], longitude_results[f'S{q}'], marker='o', markersize=3, label=f'q = {q}')
plt.xlabel('Spatial separation r (km)')
plt.ylabel('$S_q(r)$')
plt.title('Longitudinal Spatial Structure Functions — 2005')
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.tight_layout()
lon_plot = os.path.join(PLOTS_DIR, 'structure_functions_longitude_2005.png')
plt.savefig(lon_plot, dpi=300)
plt.close()
plt.figure(figsize=(9, 6))
for q in Q_VALUES:
    plt.loglog(latitude_results['distance_km'], latitude_results[f'S{q}'], marker='o', markersize=3, label=f'q = {q}')
plt.xlabel('Spatial separation r (km)')
plt.ylabel('$S_q(r)$')
plt.title('Latitudinal Spatial Structure Functions — 2005')
plt.grid(True, which='both', alpha=0.3)
plt.legend()
plt.tight_layout()
lat_plot = os.path.join(PLOTS_DIR, 'structure_functions_latitude_2005.png')
plt.savefig(lat_plot, dpi=300)
plt.close()
