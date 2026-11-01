import os
import time
import numpy as np
import pandas as pd
data_folder = 'D:\\SBS project\\vikas'
YEAR = 2005
FILES_PER_YEAR = 8832
LAT_MIN = 18.05
LAT_MAX = 24.95
LON_MIN = 78.05
LON_MAX = 91.95
EXPECTED_NLAT = 70
EXPECTED_NLON = 140
total_start = time.perf_counter()
year_folder = os.path.join(data_folder, f'kcsv{YEAR}')
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
first_sorted = common_first.sort_values(['lat', 'lon']).reset_index(drop=True)
if np.array_equal(first_sorted[['lat', 'lon']].to_numpy(), reference_coords):
    pass
rainfall = np.empty((FILES_PER_YEAR, nlat, nlon), dtype=np.float32)
load_start = time.perf_counter()
ordering_verified = False
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
            raise ValueError(f'Coordinate ordering differs in file ktest{file_number}.csv')
        if t == 4:
            ordering_verified = True
    values = common['precipitationCal'].to_numpy(dtype=np.float32)
    if len(values) != nlat * nlon:
        raise ValueError(f'Unexpected number of cells in ktest{file_number}.csv: {len(values)}')
    rainfall[t] = values.reshape(nlat, nlon)
    if (t + 1) % 500 == 0:
        elapsed = time.perf_counter() - load_start
load_time = time.perf_counter() - load_start
total_time = time.perf_counter() - total_start
