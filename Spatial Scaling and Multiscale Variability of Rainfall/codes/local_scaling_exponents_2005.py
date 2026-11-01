import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
RESULTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\results'
PLOTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\plots'
lon_file = os.path.join(RESULTS_DIR, 'structure_functions_longitude_2005.csv')
lat_file = os.path.join(RESULTS_DIR, 'structure_functions_latitude_2005.csv')
lon = pd.read_csv(lon_file)
lat = pd.read_csv(lat_file)
Q_VALUES = [1, 2, 3, 4, 5]

def calculate_local_slopes(df, q):
    log_r = np.log(df['distance_km'].to_numpy())
    log_S = np.log(df[f'S{q}'].to_numpy())
    local_slope = np.diff(log_S) / np.diff(log_r)
    r_mid = np.sqrt(df['distance_km'].to_numpy()[:-1] * df['distance_km'].to_numpy()[1:])
    return (r_mid, local_slope)
lon_slopes = pd.DataFrame()
lat_slopes = pd.DataFrame()
lon_slopes['distance_km'] = np.sqrt(lon['distance_km'].iloc[:-1].to_numpy() * lon['distance_km'].iloc[1:].to_numpy())
lat_slopes['distance_km'] = np.sqrt(lat['distance_km'].iloc[:-1].to_numpy() * lat['distance_km'].iloc[1:].to_numpy())
for q in Q_VALUES:
    _, slope_lon = calculate_local_slopes(lon, q)
    _, slope_lat = calculate_local_slopes(lat, q)
    lon_slopes[f'zeta_local_q{q}'] = slope_lon
    lat_slopes[f'zeta_local_q{q}'] = slope_lat
lon_output = os.path.join(RESULTS_DIR, 'local_scaling_exponents_longitude_2005.csv')
lat_output = os.path.join(RESULTS_DIR, 'local_scaling_exponents_latitude_2005.csv')
lon_slopes.to_csv(lon_output, index=False)
lat_slopes.to_csv(lat_output, index=False)
pd.set_option('display.max_rows', 60)
pd.set_option('display.float_format', '{:.6f}'.format)
plt.figure(figsize=(10, 7))
for q in Q_VALUES:
    plt.plot(lon_slopes['distance_km'], lon_slopes[f'zeta_local_q{q}'], marker='o', markersize=3, label=f'q = {q}')
plt.axhline(0, linewidth=1)
plt.xlabel('Spatial separation r (km)')
plt.ylabel('Local slope $d\\log S_q / d\\log r$')
plt.title('Local Scaling Exponents — Longitude, 2005')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'local_scaling_exponents_longitude_2005.png'), dpi=300)
plt.close()
plt.figure(figsize=(10, 7))
for q in Q_VALUES:
    plt.plot(lat_slopes['distance_km'], lat_slopes[f'zeta_local_q{q}'], marker='o', markersize=3, label=f'q = {q}')
plt.axhline(0, linewidth=1)
plt.xlabel('Spatial separation r (km)')
plt.ylabel('Local slope $d\\log S_q / d\\log r$')
plt.title('Local Scaling Exponents — Latitude, 2005')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'local_scaling_exponents_latitude_2005.png'), dpi=300)
plt.close()
