import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_squared_error
year = 2005
longitude_file = 'D:\\SBS project\\sem 7\\spatial_correlation_longitude_2005.csv'
latitude_file = 'D:\\SBS project\\sem 7\\spatial_correlation_latitude_2005.csv'
plot_folder = 'D:\\SBS project\\sem 7\\chapter1\\plots'
os.makedirs(plot_folder, exist_ok=True)
SHORT_RANGE_MAX = 200.0
E_THRESHOLD = 1 / np.e
lon_df = pd.read_csv(longitude_file)
lat_df = pd.read_csv(latitude_file)

def exponential_fixed_A(distance, L):
    return np.exp(-distance / L)

def direct_e_folding_distance(distance, correlation):
    distance = np.asarray(distance, dtype=float)
    correlation = np.asarray(correlation, dtype=float)
    threshold = 1 / np.e
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
    L = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
    return L

def fit_fixed_A_exponential(distance, correlation):
    distance = np.asarray(distance, dtype=float)
    correlation = np.asarray(correlation, dtype=float)
    initial_L = 75.0
    parameters, covariance = curve_fit(exponential_fixed_A, distance, correlation, p0=[initial_L], bounds=([0], [np.inf]), maxfev=10000)
    L_fit = parameters[0]
    predicted = exponential_fixed_A(distance, L_fit)
    r2 = r2_score(correlation, predicted)
    rmse = np.sqrt(mean_squared_error(correlation, predicted))
    return (L_fit, predicted, r2, rmse)

def analyze_direction(df, direction_name):
    distance = df['distance_km'].values
    correlation = df['mean_correlation'].values
    direct_L = direct_e_folding_distance(distance, correlation)
    full_L, full_predicted, full_r2, full_rmse = fit_fixed_A_exponential(distance, correlation)
    mask = distance <= SHORT_RANGE_MAX
    short_distance = distance[mask]
    short_correlation = correlation[mask]
    short_L, short_predicted, short_r2, short_rmse = fit_fixed_A_exponential(short_distance, short_correlation)
    return {'direction': direction_name, 'distance': distance, 'correlation': correlation, 'direct_L': direct_L, 'full_L': full_L, 'full_predicted': full_predicted, 'full_r2': full_r2, 'full_rmse': full_rmse, 'short_distance': short_distance, 'short_correlation': short_correlation, 'short_L': short_L, 'short_predicted': short_predicted, 'short_r2': short_r2, 'short_rmse': short_rmse}
lon_results = analyze_direction(lon_df, 'Longitude')
lat_results = analyze_direction(lat_df, 'Latitude')
comparison = pd.DataFrame({'Direction': ['Longitude', 'Latitude'], 'Direct_1e_L_km': [lon_results['direct_L'], lat_results['direct_L']], 'Full_Exponential_L_km': [lon_results['full_L'], lat_results['full_L']], 'Full_Exponential_R2': [lon_results['full_r2'], lat_results['full_r2']], 'Full_Exponential_RMSE': [lon_results['full_rmse'], lat_results['full_rmse']], 'Short_Medium_L_km': [lon_results['short_L'], lat_results['short_L']], 'Short_Medium_R2': [lon_results['short_r2'], lat_results['short_r2']], 'Short_Medium_RMSE': [lon_results['short_rmse'], lat_results['short_rmse']]})
lon_distance = lon_results['distance']
lon_correlation = lon_results['correlation']
lon_smooth = np.linspace(0, lon_distance.max(), 500)
lon_full_curve = exponential_fixed_A(lon_smooth, lon_results['full_L'])
lon_short_curve = exponential_fixed_A(lon_smooth, lon_results['short_L'])
plt.figure(figsize=(10, 6))
plt.scatter(lon_distance, lon_correlation, label='Observed mean correlation')
plt.plot(lon_smooth, lon_full_curve, linewidth=2, label=f"Full exponential (L = {lon_results['full_L']:.2f} km)")
plt.plot(lon_smooth, lon_short_curve, linewidth=2, linestyle='--', label=f"≤200 km exponential (L = {lon_results['short_L']:.2f} km)")
plt.axhline(E_THRESHOLD, linestyle=':', label='1/e')
plt.axvline(lon_results['direct_L'], linestyle='-.', label=f"Direct 1/e = {lon_results['direct_L']:.2f} km")
plt.xlabel('Spatial Separation (km)')
plt.ylabel('Mean Pearson Correlation')
plt.title(f'Longitudinal Correlation — Methodology Check ({year})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
lon_plot = os.path.join(plot_folder, f'methodology_check_longitude_{year}.png')
plt.savefig(lon_plot, dpi=300, bbox_inches='tight')
plt.show()
lat_distance = lat_results['distance']
lat_correlation = lat_results['correlation']
lat_smooth = np.linspace(0, lat_distance.max(), 500)
lat_full_curve = exponential_fixed_A(lat_smooth, lat_results['full_L'])
lat_short_curve = exponential_fixed_A(lat_smooth, lat_results['short_L'])
plt.figure(figsize=(10, 6))
plt.scatter(lat_distance, lat_correlation, label='Observed mean correlation')
plt.plot(lat_smooth, lat_full_curve, linewidth=2, label=f"Full exponential (L = {lat_results['full_L']:.2f} km)")
plt.plot(lat_smooth, lat_short_curve, linewidth=2, linestyle='--', label=f"≤200 km exponential (L = {lat_results['short_L']:.2f} km)")
plt.axhline(E_THRESHOLD, linestyle=':', label='1/e')
plt.axvline(lat_results['direct_L'], linestyle='-.', label=f"Direct 1/e = {lat_results['direct_L']:.2f} km")
plt.xlabel('Spatial Separation (km)')
plt.ylabel('Mean Pearson Correlation')
plt.title(f'Latitudinal Correlation — Methodology Check ({year})')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
lat_plot = os.path.join(plot_folder, f'methodology_check_latitude_{year}.png')
plt.savefig(lat_plot, dpi=300, bbox_inches='tight')
plt.show()
results_file = os.path.join(plot_folder, f'methodology_comparison_{year}.csv')
comparison.to_csv(results_file, index=False)
