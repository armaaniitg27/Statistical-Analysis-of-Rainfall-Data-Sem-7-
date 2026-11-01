import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
DATA_ROOT = 'D:\\SBS project\\vikas'
OUTPUT_ROOT = 'D:\\SBS project\\sem 7'
YEAR = 'kcsv2005'
FILE = 'ktest2.csv'
file_path = os.path.join(DATA_ROOT, YEAR, FILE)
df = pd.read_csv(file_path, usecols=['lat', 'lon', 'precipitationCal'])
lat = np.sort(df['lat'].unique())
lon = np.sort(df['lon'].unique())
n_lat = len(lat)
n_lon = len(lon)
lat_diff = np.diff(lat)
lon_diff = np.diff(lon)
duplicates = df.duplicated(subset=['lat', 'lon']).sum()
expected_points = set(((la, lo) for la in lat for lo in lon))
actual_points = set(zip(df['lat'], df['lon']))
missing_points = expected_points - actual_points
if len(missing_points) > 0:
    for point in list(missing_points)[:10]:
        pass
plt.figure(figsize=(10, 6))
plt.scatter(df['lon'], df['lat'], s=2)
plt.xlabel('Longitude (°E)')
plt.ylabel('Latitude (°N)')
plt.title(f'Rainfall Spatial Grid — {YEAR}')
plt.grid(True, alpha=0.3)
plt.tight_layout()
output_file = os.path.join(OUTPUT_ROOT, 'spatial_grid_2005.png')
plt.savefig(output_file, dpi=300)
plt.show()
