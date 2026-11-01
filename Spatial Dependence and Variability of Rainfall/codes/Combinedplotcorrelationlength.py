import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
year = 2005
longitude_file = 'D:\\SBS project\\sem 7\\spatial_correlation_longitude_2005.csv'
latitude_file = 'D:\\SBS project\\sem 7\\spatial_correlation_latitude_2005.csv'
e_threshold = 1 / np.e
lon_df = pd.read_csv(longitude_file)
lat_df = pd.read_csv(latitude_file)

def calculate_correlation_length(distance, correlation, threshold=1 / np.e):
    distance = np.asarray(distance)
    correlation = np.asarray(correlation)
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
    correlation_length = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
    return correlation_length
lon_length = calculate_correlation_length(lon_df['distance_km'], lon_df['mean_correlation'])
lat_length = calculate_correlation_length(lat_df['distance_km'], lat_df['mean_correlation'])
plt.figure(figsize=(10, 6))
plt.plot(lon_df['distance_km'], lon_df['mean_correlation'], marker='o', markersize=3, label='Longitude separation')
plt.plot(lat_df['distance_km'], lat_df['mean_correlation'], marker='o', markersize=3, label='Latitude separation')
plt.axhline(e_threshold, linestyle='--', label='$1/e \\approx 0.368$')
plt.axvline(lon_length, linestyle=':', label=f'Longitude L = {lon_length:.1f} km')
plt.axvline(lat_length, linestyle=':', label=f'Latitude L = {lat_length:.1f} km')
plt.xlabel('Spatial Separation (km)')
plt.ylabel('Mean Pearson Correlation')
plt.title(f'Spatial Correlation of Rainfall — {year}')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
output_plot = f'D:\\SBS project\\sem 7\\spatial_correlation_comparison_{year}.png'
plt.savefig(output_plot, dpi=300)
plt.show()
summary = pd.DataFrame({'year': [year], 'longitude_correlation_length_km': [lon_length], 'latitude_correlation_length_km': [lat_length]})
output_csv = f'D:\\SBS project\\sem 7\\correlation_length_{year}.csv'
summary.to_csv(output_csv, index=False)
