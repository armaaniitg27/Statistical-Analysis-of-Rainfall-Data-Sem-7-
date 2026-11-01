import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
CHAPTER1_RESULTS = 'D:\\SBS project\\sem 7\\chapter1\\results'
CHAPTER2_RESULTS = 'D:\\SBS project\\sem 7\\chapter2\\results'
CHAPTER2_PLOTS = 'D:\\SBS project\\sem 7\\chapter2\\plots'
os.makedirs(CHAPTER2_PLOTS, exist_ok=True)
YEAR = 2005
SCALING_WINDOW = 'W3'
SCALING_MIN_KM = 80
SCALING_MAX_KM = 135
MIN_R2 = 0.9
scaling_file = os.path.join(CHAPTER2_RESULTS, 'local_scaling_window_stability_2005.csv')
scaling = pd.read_csv(scaling_file)
scaling = scaling[scaling['window'] == SCALING_WINDOW].copy()
scaling['block_row'] = (scaling['block'].str.extract('(\\d+)')[0].astype(int) - 1) // 5 + 1
scaling['block_col'] = (scaling['block'].str.extract('(\\d+)')[0].astype(int) - 1) % 5 + 1
corr_file = os.path.join(CHAPTER1_RESULTS, 'spatial_heterogeneity_2005.csv')
corr = pd.read_csv(corr_file)
corr = corr[['block_row', 'block_col', 'L_longitude_km', 'L_latitude_km', 'anisotropy_ratio']].copy()
merged = pd.merge(scaling, corr, on=['block_row', 'block_col'], how='inner')

def correlation_analysis(df, x_col, y_col):
    data = df[[x_col, y_col]].dropna()
    x = data[x_col].to_numpy()
    y = data[y_col].to_numpy()
    if len(data) < 3:
        return None
    pearson_r = np.corrcoef(x, y)[0, 1]
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return {'n': len(data), 'pearson_r': pearson_r, 'slope': slope, 'intercept': intercept, 'R2': r2}
lon_reliable = merged[merged['lon_R2_q2'] >= MIN_R2].copy()
lat_reliable = merged[merged['lat_R2_q2'] >= MIN_R2].copy()
lon_raw = correlation_analysis(merged, 'L_longitude_km', 'lon_zeta_q2')
lon_filtered = correlation_analysis(lon_reliable, 'L_longitude_km', 'lon_zeta_q2')
lat_raw = correlation_analysis(merged, 'L_latitude_km', 'lat_zeta_q2')
lat_filtered = correlation_analysis(lat_reliable, 'L_latitude_km', 'lat_zeta_q2')
lon_aniso = correlation_analysis(merged, 'anisotropy_ratio', 'lon_zeta_q2')
lat_aniso = correlation_analysis(merged, 'anisotropy_ratio', 'lat_zeta_q2')
output_file = os.path.join(CHAPTER2_RESULTS, 'correlation_length_vs_scaling_2005.csv')
merged.to_csv(output_file, index=False)

def make_scatter_plot(df, x_col, y_col, xlabel, ylabel, title, stats, filename):
    x = df[x_col]
    y = df[y_col]
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y)
    slope = stats['slope']
    intercept = stats['intercept']
    x_line = np.linspace(x.min(), x.max(), 100)
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.text(0.05, 0.95, f"Pearson r = {stats['pearson_r']:.3f}\n$R^2$ = {stats['R2']:.3f}\nN = {stats['n']}", transform=plt.gca().transAxes, verticalalignment='top')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
make_scatter_plot(lon_reliable, 'L_longitude_km', 'lon_zeta_q2', 'Local longitudinal correlation length (km)', 'Local $\\zeta_{\\mathrm{lon}}(2)$', 'Correlation length vs. local scaling exponent — longitude, 2005', lon_filtered, os.path.join(CHAPTER2_PLOTS, 'correlation_length_vs_zeta_lon_2005.png'))
make_scatter_plot(lat_reliable, 'L_latitude_km', 'lat_zeta_q2', 'Local latitudinal correlation length (km)', 'Local $\\zeta_{\\mathrm{lat}}(2)$', 'Correlation length vs. local scaling exponent — latitude, 2005', lat_filtered, os.path.join(CHAPTER2_PLOTS, 'correlation_length_vs_zeta_lat_2005.png'))
make_scatter_plot(merged, 'anisotropy_ratio', 'lon_zeta_q2', 'Local anisotropy ratio', 'Local $\\zeta_{\\mathrm{lon}}(2)$', 'Anisotropy vs. longitudinal scaling exponent — 2005', lon_aniso, os.path.join(CHAPTER2_PLOTS, 'anisotropy_vs_zeta_lon_2005.png'))
make_scatter_plot(merged, 'anisotropy_ratio', 'lat_zeta_q2', 'Local anisotropy ratio', 'Local $\\zeta_{\\mathrm{lat}}(2)$', 'Anisotropy vs. latitudinal scaling exponent — 2005', lat_aniso, os.path.join(CHAPTER2_PLOTS, 'anisotropy_vs_zeta_lat_2005.png'))
