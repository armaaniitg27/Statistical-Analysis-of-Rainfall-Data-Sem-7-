import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
DATA_ROOT = 'D:\\SBS project\\vikas'
OUTPUT_ROOT = 'D:\\SBS project\\sem 7'
YEAR = 'kcsv2005'
N_LAT = 70
N_LON = 140
MAX_LAG = 50

def correlation_at_lag(data, axis, lag):
    """
    Calculate Pearson correlation between grid cells
    separated by 'lag' cells along the specified axis.

    data shape:
        (time, latitude, longitude)

    axis = 2 -> longitude
    axis = 1 -> latitude
    """
    if axis == 2:
        x = data[:, :, :-lag]
        y = data[:, :, lag:]
    elif axis == 1:
        x = data[:, :-lag, :]
        y = data[:, lag:, :]
    else:
        raise ValueError('axis must be 1 or 2')
    x = x.reshape(x.shape[0], -1)
    y = y.reshape(y.shape[0], -1)
    correlations = []
    for i in range(x.shape[1]):
        x_i = x[:, i]
        y_i = y[:, i]
        if np.std(x_i) == 0 or np.std(y_i) == 0:
            continue
        corr = np.corrcoef(x_i, y_i)[0, 1]
        if np.isfinite(corr):
            correlations.append(corr)
    if len(correlations) == 0:
        return (np.nan, 0)
    return (np.mean(correlations), len(correlations))
folder = os.path.join(DATA_ROOT, YEAR)
files = [f for f in os.listdir(folder) if f.lower().endswith('.csv') and f.lower().startswith('ktest')]
files.sort(key=lambda x: int(''.join(filter(str.isdigit, os.path.splitext(x)[0]))))
first_file = os.path.join(folder, files[0])
df_first = pd.read_csv(first_file)
latitudes = np.sort(df_first['lat'].unique())
longitudes = np.sort(df_first['lon'].unique())
n_time = len(files)
rainfall = np.empty((n_time, N_LAT, N_LON), dtype=np.float32)
for t, filename in enumerate(files):
    path = os.path.join(folder, filename)
    df = pd.read_csv(path, usecols=['lat', 'lon', 'precipitationCal'])
    grid = df.pivot(index='lat', columns='lon', values='precipitationCal')
    grid = grid.reindex(index=latitudes, columns=longitudes)
    rainfall[t] = grid.to_numpy(dtype=np.float32)
    if (t + 1) % 500 == 0:
        pass
missing = np.isnan(rainfall).sum()
longitude_results = []
latitude_results = []
for lag in range(1, MAX_LAG + 1):
    mean_corr, n_pairs = correlation_at_lag(rainfall, axis=2, lag=lag)
    mean_latitude = np.mean(latitudes)
    distance_km = 111.32 * np.cos(np.radians(mean_latitude)) * 0.1 * lag
    longitude_results.append({'lag': lag, 'distance_km': distance_km, 'mean_correlation': mean_corr, 'number_of_pairs': n_pairs})
for lag in range(1, MAX_LAG + 1):
    mean_corr, n_pairs = correlation_at_lag(rainfall, axis=1, lag=lag)
    distance_km = 111.32 * 0.1 * lag
    latitude_results.append({'lag': lag, 'distance_km': distance_km, 'mean_correlation': mean_corr, 'number_of_pairs': n_pairs})
longitude_df = pd.DataFrame(longitude_results)
latitude_df = pd.DataFrame(latitude_results)
longitude_output = os.path.join(OUTPUT_ROOT, 'spatial_correlation_longitude_2005.csv')
latitude_output = os.path.join(OUTPUT_ROOT, 'spatial_correlation_latitude_2005.csv')
longitude_df.to_csv(longitude_output, index=False)
latitude_df.to_csv(latitude_output, index=False)
plt.figure(figsize=(9, 6))
plt.plot(longitude_df['distance_km'], longitude_df['mean_correlation'], marker='o', markersize=4)
plt.xlabel('Longitude separation (km)')
plt.ylabel('Mean Pearson correlation')
plt.title('Spatial Correlation vs Longitude Separation — 2005')
plt.grid(True, alpha=0.3)
plt.tight_layout()
longitude_plot = os.path.join(OUTPUT_ROOT, 'spatial_correlation_longitude_2005.png')
plt.savefig(longitude_plot, dpi=300)
plt.show()
plt.figure(figsize=(9, 6))
plt.plot(latitude_df['distance_km'], latitude_df['mean_correlation'], marker='o', markersize=4)
plt.xlabel('Latitude separation (km)')
plt.ylabel('Mean Pearson correlation')
plt.title('Spatial Correlation vs Latitude Separation — 2005')
plt.grid(True, alpha=0.3)
plt.tight_layout()
latitude_plot = os.path.join(OUTPUT_ROOT, 'spatial_correlation_latitude_2005.png')
plt.savefig(latitude_plot, dpi=300)
plt.show()
