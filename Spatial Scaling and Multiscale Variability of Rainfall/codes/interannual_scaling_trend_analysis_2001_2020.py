import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress
RESULTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\results'
PLOTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\plots'
INPUT_FILE = os.path.join(RESULTS_DIR, 'interannual_scaling_zeta2_2001_2020_COMPLETE.csv')
os.makedirs(PLOTS_DIR, exist_ok=True)
df = pd.read_csv(INPUT_FILE)
df = df.sort_values('year').reset_index(drop=True)
year = df['year'].to_numpy(dtype=float)
lon = df['lon_zeta2'].to_numpy(dtype=float)
lat = df['lat_zeta2'].to_numpy(dtype=float)
difference = df['zeta2_difference'].to_numpy(dtype=float)
ratio = df['zeta2_ratio'].to_numpy(dtype=float)
r_lon_lat, p_lon_lat = pearsonr(lon, lat)
r_lon_diff, p_lon_diff = pearsonr(lon, difference)
r_lat_diff, p_lat_diff = pearsonr(lat, difference)
trend_lon = linregress(year, lon)
trend_lat = linregress(year, lat)
trend_difference = linregress(year, difference)
trend_ratio = linregress(year, ratio)
lon_2001 = lon[0]
lon_2020 = lon[-1]
lat_2001 = lat[0]
lat_2020 = lat[-1]
diff_2001 = difference[0]
diff_2020 = difference[-1]
ratio_2001 = ratio[0]
ratio_2020 = ratio[-1]
lon_percent_change = (lon_2020 - lon_2001) / lon_2001 * 100
lat_percent_change = (lat_2020 - lat_2001) / lat_2001 * 100
diff_percent_change = (diff_2020 - diff_2001) / diff_2001 * 100
ratio_percent_change = (ratio_2020 - ratio_2001) / ratio_2001 * 100
lon_min_year = int(df.loc[df['lon_zeta2'].idxmin(), 'year'])
lon_max_year = int(df.loc[df['lon_zeta2'].idxmax(), 'year'])
lat_min_year = int(df.loc[df['lat_zeta2'].idxmin(), 'year'])
lat_max_year = int(df.loc[df['lat_zeta2'].idxmax(), 'year'])
diff_min_year = int(df.loc[df['zeta2_difference'].idxmin(), 'year'])
diff_max_year = int(df.loc[df['zeta2_difference'].idxmax(), 'year'])
statistics = pd.DataFrame({'quantity': ['lon_lat_correlation', 'lon_lat_correlation_pvalue', 'lon_trend_slope', 'lon_trend_pvalue', 'lon_trend_R2', 'lat_trend_slope', 'lat_trend_pvalue', 'lat_trend_R2', 'difference_trend_slope', 'difference_trend_pvalue', 'difference_trend_R2', 'ratio_trend_slope', 'ratio_trend_pvalue', 'ratio_trend_R2', 'lon_2001_to_2020_percent_change', 'lat_2001_to_2020_percent_change', 'difference_2001_to_2020_percent_change', 'ratio_2001_to_2020_percent_change'], 'value': [r_lon_lat, p_lon_lat, trend_lon.slope, trend_lon.pvalue, trend_lon.rvalue ** 2, trend_lat.slope, trend_lat.pvalue, trend_lat.rvalue ** 2, trend_difference.slope, trend_difference.pvalue, trend_difference.rvalue ** 2, trend_ratio.slope, trend_ratio.pvalue, trend_ratio.rvalue ** 2, lon_percent_change, lat_percent_change, diff_percent_change, ratio_percent_change]})
statistics_file = os.path.join(RESULTS_DIR, 'interannual_scaling_trend_analysis_2001_2020.csv')
statistics.to_csv(statistics_file, index=False)
plt.figure(figsize=(7, 6))
plt.scatter(lon, lat, s=55)
fit = np.polyfit(lon, lat, 1)
x_fit = np.linspace(lon.min(), lon.max(), 100)
y_fit = fit[0] * x_fit + fit[1]
plt.plot(x_fit, y_fit, linestyle='--', linewidth=1.5)
plt.xlabel('$\\zeta_{\\mathrm{lon}}(2)$')
plt.ylabel('$\\zeta_{\\mathrm{lat}}(2)$')
plt.title('Relationship between directional scaling exponents')
plt.grid(alpha=0.3)
plt.tight_layout()
plot_corr = os.path.join(PLOTS_DIR, 'interannual_zeta2_lon_vs_lat_2001_2020.png')
plt.savefig(plot_corr, dpi=300, bbox_inches='tight')
plt.close()
plt.figure(figsize=(10, 6))
plt.plot(year, lon, marker='o', linewidth=2, label='$\\zeta_{\\mathrm{lon}}(2)$')
plt.plot(year, lat, marker='s', linewidth=2, label='$\\zeta_{\\mathrm{lat}}(2)$')
lon_fit = trend_lon.intercept + trend_lon.slope * year
plt.plot(year, lon_fit, linestyle='--', linewidth=1)
lat_fit = trend_lat.intercept + trend_lat.slope * year
plt.plot(year, lat_fit, linestyle='--', linewidth=1)
plt.xlabel('Year')
plt.ylabel('$\\zeta(2)$')
plt.title('Interannual variation and linear trends of $\\zeta(2)$')
plt.xticks(year, rotation=45)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plot_trend = os.path.join(PLOTS_DIR, 'interannual_zeta2_trends_2001_2020.png')
plt.savefig(plot_trend, dpi=300, bbox_inches='tight')
plt.close()
plt.figure(figsize=(10, 6))
plt.plot(year, difference, marker='o', linewidth=2)
difference_fit = trend_difference.intercept + trend_difference.slope * year
plt.plot(year, difference_fit, linestyle='--', linewidth=1.5)
plt.axhline(0, linestyle=':', linewidth=1)
plt.xlabel('Year')
plt.ylabel('$\\zeta_{\\mathrm{lon}}(2)-\\zeta_{\\mathrm{lat}}(2)$')
plt.title('Interannual variation of directional scaling difference')
plt.xticks(year, rotation=45)
plt.grid(alpha=0.3)
plt.tight_layout()
plot_difference = os.path.join(PLOTS_DIR, 'interannual_zeta2_difference_trend_2001_2020.png')
plt.savefig(plot_difference, dpi=300, bbox_inches='tight')
plt.close()
