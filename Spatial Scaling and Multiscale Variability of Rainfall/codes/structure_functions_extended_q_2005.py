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
Q_VALUES = np.arange(0.5, 8.5, 0.5)
files = [os.path.join(DATA_DIR, f'ktest{i}.csv') for i in range(2, 8834)]
rainfall_list = []
for idx, file in enumerate(files, start=1):
    df = pd.read_csv(file)
    rainfall_list.append(df['precipitationCal'].values)
rainfall = np.array(rainfall_list)
n_time = rainfall.shape[0]
n_grid = rainfall.shape[1]
n_lat = 70
n_lon = 140
if n_grid != n_lat * n_lon:
    raise ValueError(f'Expected {n_lat * n_lon} grid cells, but found {n_grid}')
rainfall = rainfall.reshape(n_time, n_lat, n_lon)
missing = np.isnan(rainfall).sum()
if missing > 0:
    raise ValueError('Missing rainfall values detected.')
lat_spacing_deg = 0.1
lon_spacing_deg = 0.1
km_per_degree = 111.32
latitudes = np.linspace(18.05, 24.95, n_lat)
mean_latitude = np.mean(latitudes)
lon_spacing_km = lon_spacing_deg * km_per_degree * np.cos(np.radians(mean_latitude))
lat_spacing_km = lat_spacing_deg * km_per_degree

def calculate_structure_functions(rainfall, direction, max_lag, q_values):
    results = []
    for lag in range(1, max_lag + 1):
        if direction == 'longitude':
            differences = rainfall[:, :, lag:] - rainfall[:, :, :-lag]
            distance_km = lag * lon_spacing_km
        elif direction == 'latitude':
            differences = rainfall[:, lag:, :] - rainfall[:, :-lag, :]
            distance_km = lag * lat_spacing_km
        else:
            raise ValueError('Direction must be longitude or latitude.')
        differences = differences.ravel()
        absolute_differences = np.abs(differences)
        result = {'lag': lag, 'distance_km': distance_km, 'number_of_increments': len(differences)}
        for q in q_values:
            Sq = np.mean(absolute_differences ** q)
            result[f'S{q:g}'] = Sq
        results.append(result)
    return pd.DataFrame(results)
longitude_sf = calculate_structure_functions(rainfall, 'longitude', MAX_LAG, Q_VALUES)
longitude_file = os.path.join(RESULTS_DIR, 'structure_functions_longitude_2005_q_extended.csv')
longitude_sf.to_csv(longitude_file, index=False)
latitude_sf = calculate_structure_functions(rainfall, 'latitude', MAX_LAG, Q_VALUES)
latitude_file = os.path.join(RESULTS_DIR, 'structure_functions_latitude_2005_q_extended.csv')
latitude_sf.to_csv(latitude_file, index=False)

def plot_structure_functions(df, direction):
    plt.figure(figsize=(9, 7))
    for q in q_values:
        column = f'S{q:g}'
        plt.loglog(df['distance_km'], df[column], marker='o', markersize=3, linewidth=1, label=f'q = {q:g}')
    plt.xlabel('Spatial separation (km)')
    plt.ylabel('$S_q(r)$')
    plt.title(f'Spatial Structure Functions - {direction.capitalize()}, 2005')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    output = os.path.join(PLOTS_DIR, f'structure_functions_{direction}_2005_q_extended.png')
    plt.savefig(output, dpi=300)
    plt.close()
plot_structure_functions(longitude_sf, 'longitude')
plot_structure_functions(latitude_sf, 'latitude')
