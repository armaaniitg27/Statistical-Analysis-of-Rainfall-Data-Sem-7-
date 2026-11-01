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
FILES_PER_YEAR = 8832
LAT_MIN = 18.05
LAT_MAX = 24.95
LON_MIN = 78.05
LON_MAX = 91.95
EXPECTED_NLAT = 70
EXPECTED_NLON = 140
N_LAT_BLOCKS = 5
N_LON_BLOCKS = 5
MAX_LAG = 13
CORR_THRESHOLD = 1 / np.e
LAT_STEP_KM = 11.132
MEAN_LATITUDE = (LAT_MIN + LAT_MAX) / 2
LON_STEP_KM = 111.32 * np.cos(np.radians(MEAN_LATITUDE)) * 0.1

def load_year(year):
    year_folder = os.path.join(data_folder, f'kcsv{year}')
    if not os.path.exists(year_folder):
        raise FileNotFoundError(f'Year folder not found:\n{year_folder}')
    first_file = os.path.join(year_folder, 'ktest2.csv')
    first_df = pd.read_csv(first_file)
    first_df['lat'] = first_df['lat'].round(2)
    first_df['lon'] = first_df['lon'].round(2)
    common_first = first_df[(first_df['lat'] >= LAT_MIN) & (first_df['lat'] <= LAT_MAX) & (first_df['lon'] >= LON_MIN) & (first_df['lon'] <= LON_MAX)].copy()
    lats = np.sort(common_first['lat'].unique())
    lons = np.sort(common_first['lon'].unique())
    nlat = len(lats)
    nlon = len(lons)
    if nlat != EXPECTED_NLAT or nlon != EXPECTED_NLON:
        raise ValueError(f'Unexpected grid: {nlat} x {nlon}')
    reference = common_first.sort_values(['lat', 'lon']).reset_index(drop=True)
    reference_coords = reference[['lat', 'lon']].to_numpy()
    rainfall = np.empty((FILES_PER_YEAR, nlat, nlon), dtype=np.float32)
    load_start = time.perf_counter()
    for t in range(FILES_PER_YEAR):
        file_number = t + 2
        file_path = os.path.join(year_folder, f'ktest{file_number}.csv')
        df = pd.read_csv(file_path)
        df['lat'] = df['lat'].round(2)
        df['lon'] = df['lon'].round(2)
        common = df[(df['lat'] >= LAT_MIN) & (df['lat'] <= LAT_MAX) & (df['lon'] >= LON_MIN) & (df['lon'] <= LON_MAX)]
        common = common.sort_values(['lat', 'lon'])
        if t < 5:
            current_coords = common[['lat', 'lon']].to_numpy()
            if not np.array_equal(current_coords, reference_coords):
                raise ValueError(f'Coordinate ordering differs in ktest{file_number}.csv')
        values = common['precipitationCal'].to_numpy(dtype=np.float32)
        if len(values) != nlat * nlon:
            raise ValueError(f'Unexpected cell count in ktest{file_number}.csv')
        rainfall[t] = values.reshape(nlat, nlon)
        if (t + 1) % 1000 == 0:
            elapsed = time.perf_counter() - load_start
    load_time = time.perf_counter() - load_start
    return (rainfall, lats, lons)

def calculate_local_correlations(block, max_lag=10):
    n_time, nlat, nlon = block.shape
    mean = np.mean(block, axis=0, dtype=np.float64)
    std = np.std(block, axis=0, ddof=1, dtype=np.float64)
    std[std == 0] = np.nan
    z = (block.astype(np.float64) - mean[None, :, :]) / std[None, :, :]
    lon_correlations = []
    for lag in range(1, min(max_lag, nlon - 1) + 1):
        x = z[:, :, :-lag]
        y = z[:, :, lag:]
        corr = np.nanmean(x * y) * (n_time / (n_time - 1))
        lon_correlations.append(corr)
    lat_correlations = []
    for lag in range(1, min(max_lag, nlat - 1) + 1):
        x = z[:, :-lag, :]
        y = z[:, lag:, :]
        corr = np.nanmean(x * y) * (n_time / (n_time - 1))
        lat_correlations.append(corr)
    lon_correlations = np.array(lon_correlations)
    lat_correlations = np.array(lat_correlations)
    lon_distance = np.arange(1, len(lon_correlations) + 1) * LON_STEP_KM
    lat_distance = np.arange(1, len(lat_correlations) + 1) * LAT_STEP_KM
    return (lon_distance, lon_correlations, lat_distance, lat_correlations)

def correlation_length(distance, correlation):
    distance = np.asarray(distance)
    correlation = np.asarray(correlation)
    below = np.where(correlation <= CORR_THRESHOLD)[0]
    if len(below) == 0:
        return np.nan
    idx = below[0]
    if idx == 0:
        x1 = 0.0
        y1 = 1.0
        x2 = distance[0]
        y2 = correlation[0]
    else:
        x1 = distance[idx - 1]
        y1 = correlation[idx - 1]
        x2 = distance[idx]
        y2 = correlation[idx]
    if y2 == y1:
        return x2
    L = x1 + (CORR_THRESHOLD - y1) * (x2 - x1) / (y2 - y1)
    return L
total_start = time.perf_counter()
rainfall, lats, lons = load_year(YEAR)
lat_edges = np.linspace(0, EXPECTED_NLAT, N_LAT_BLOCKS + 1, dtype=int)
lon_edges = np.linspace(0, EXPECTED_NLON, N_LON_BLOCKS + 1, dtype=int)
results = []
curve_results = []
calculation_start = time.perf_counter()
block_number = 0
for bi in range(N_LAT_BLOCKS):
    for bj in range(N_LON_BLOCKS):
        block_number += 1
        lat_start = lat_edges[bi]
        lat_end = lat_edges[bi + 1]
        lon_start = lon_edges[bj]
        lon_end = lon_edges[bj + 1]
        block = rainfall[:, lat_start:lat_end, lon_start:lon_end]
        lon_distance, lon_corr, lat_distance, lat_corr = calculate_local_correlations(block, MAX_LAG)
        L_lon = correlation_length(lon_distance, lon_corr)
        L_lat = correlation_length(lat_distance, lat_corr)
        if np.isfinite(L_lon) and np.isfinite(L_lat) and (L_lat != 0):
            ratio = L_lon / L_lat
        else:
            ratio = np.nan
        block_id = f'B{bi + 1}{bj + 1}'
        lat_block = lats[lat_start:lat_end]
        lon_block = lons[lon_start:lon_end]
        results.append({'block_id': block_id, 'block_row': bi + 1, 'block_col': bj + 1, 'lat_min': lat_block[0], 'lat_max': lat_block[-1], 'lon_min': lon_block[0], 'lon_max': lon_block[-1], 'L_longitude_km': L_lon, 'L_latitude_km': L_lat, 'anisotropy_ratio': ratio})
        for k in range(len(lon_corr)):
            curve_results.append({'block_id': block_id, 'block_row': bi + 1, 'block_col': bj + 1, 'direction': 'longitude', 'lag': k + 1, 'distance_km': lon_distance[k], 'correlation': lon_corr[k]})
        for k in range(len(lat_corr)):
            curve_results.append({'block_id': block_id, 'block_row': bi + 1, 'block_col': bj + 1, 'direction': 'latitude', 'lag': k + 1, 'distance_km': lat_distance[k], 'correlation': lat_corr[k]})
calculation_time = time.perf_counter() - calculation_start
results_df = pd.DataFrame(results)
curves_df = pd.DataFrame(curve_results)
results_path = os.path.join(result_folder, 'spatial_heterogeneity_2005_fast.csv')
curves_path = os.path.join(result_folder, 'spatial_heterogeneity_curves_2005.csv')
results_df.to_csv(results_path, index=False)
curves_df.to_csv(curves_path, index=False)
for column, label in [('L_longitude_km', 'Longitude correlation length'), ('L_latitude_km', 'Latitude correlation length'), ('anisotropy_ratio', 'Anisotropy ratio')]:
    values = results_df[column].dropna()
fig, ax = plt.subplots(figsize=(10, 7))
for block_id in curves_df['block_id'].unique():
    subset = curves_df[(curves_df['block_id'] == block_id) & (curves_df['direction'] == 'longitude')]
    ax.plot(subset['distance_km'], subset['correlation'], marker='o', markersize=3, linewidth=1, label=block_id)
ax.axhline(CORR_THRESHOLD, linestyle='--', linewidth=1.5, label='1/e')
ax.set_xlabel('Longitude separation (km)')
ax.set_ylabel('Pearson correlation')
ax.set_title('Local Longitudinal Spatial Correlation Curves - 2005')
ax.grid(True, alpha=0.3)
ax.legend(ncol=5, fontsize=8)
plt.tight_layout()
lon_curve_path = os.path.join(plot_folder, 'local_longitudinal_correlation_curves_2005.png')
plt.savefig(lon_curve_path, dpi=300, bbox_inches='tight')
plt.close()
fig, ax = plt.subplots(figsize=(10, 7))
for block_id in curves_df['block_id'].unique():
    subset = curves_df[(curves_df['block_id'] == block_id) & (curves_df['direction'] == 'latitude')]
    ax.plot(subset['distance_km'], subset['correlation'], marker='o', markersize=3, linewidth=1, label=block_id)
ax.axhline(CORR_THRESHOLD, linestyle='--', linewidth=1.5, label='1/e')
ax.set_xlabel('Latitude separation (km)')
ax.set_ylabel('Pearson correlation')
ax.set_title('Local Latitudinal Spatial Correlation Curves - 2005')
ax.grid(True, alpha=0.3)
ax.legend(ncol=5, fontsize=8)
plt.tight_layout()
lat_curve_path = os.path.join(plot_folder, 'local_latitudinal_correlation_curves_2005.png')
plt.savefig(lat_curve_path, dpi=300, bbox_inches='tight')
plt.close()

def make_heatmap(values, title, colorbar_label, filename, fmt='.1f'):
    matrix = values.reshape(N_LAT_BLOCKS, N_LON_BLOCKS)
    fig, ax = plt.subplots(figsize=(9, 6))
    image = ax.imshow(matrix, origin='lower', aspect='auto')
    cbar = plt.colorbar(image, ax=ax)
    cbar.set_label(colorbar_label)
    ax.set_xlabel('Longitude block')
    ax.set_ylabel('Latitude block')
    ax.set_title(title)
    for i in range(N_LAT_BLOCKS):
        for j in range(N_LON_BLOCKS):
            value = matrix[i, j]
            if np.isfinite(value):
                ax.text(j, i, format(value, fmt), ha='center', va='center', fontsize=9)
    plt.tight_layout()
    path = os.path.join(plot_folder, filename)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
make_heatmap(results_df['L_longitude_km'].values, 'Spatial Heterogeneity of Longitudinal Correlation Length - 2005', 'Longitudinal correlation length (km)', 'spatial_heterogeneity_longitude_2005_fast.png')
make_heatmap(results_df['L_latitude_km'].values, 'Spatial Heterogeneity of Latitudinal Correlation Length - 2005', 'Latitudinal correlation length (km)', 'spatial_heterogeneity_latitude_2005_fast.png')
make_heatmap(results_df['anisotropy_ratio'].values, 'Spatial Heterogeneity of Anisotropy - 2005', 'Anisotropy ratio', 'spatial_heterogeneity_anisotropy_2005_fast.png', fmt='.3f')
total_time = time.perf_counter() - total_start
