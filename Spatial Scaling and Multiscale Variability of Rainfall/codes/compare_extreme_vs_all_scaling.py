import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import ttest_rel, pearsonr
RESULTS_DIR = Path('D:\\SBS project\\sem 7\\chapter2\\results')
PLOTS_DIR = Path('D:\\SBS project\\sem 7\\chapter2\\plots')
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
ALL_FILE = RESULTS_DIR / 'interannual_scaling_zeta2_2001_2020_COMPLETE.csv'
EXTREME_FILE = RESULTS_DIR / 'extreme_spatial_scaling_zeta2_2001_2020.csv'
COMPARISON_FILE = RESULTS_DIR / 'extreme_vs_all_zeta2_comparison_2001_2020.csv'
SUMMARY_FILE = RESULTS_DIR / 'extreme_vs_all_zeta2_summary_2001_2020.csv'
TESTS_FILE = RESULTS_DIR / 'extreme_vs_all_zeta2_paired_tests_2001_2020.csv'
all_df = pd.read_csv(ALL_FILE)
extreme_df = pd.read_csv(EXTREME_FILE)
all_df = all_df[['year', 'lon_zeta2', 'lat_zeta2']].copy()
extreme_df = extreme_df[['year', 'zeta2_longitude_extreme', 'zeta2_latitude_extreme']].copy()
all_df = all_df.rename(columns={'lon_zeta2': 'zeta2_longitude_all', 'lat_zeta2': 'zeta2_latitude_all'})
df = pd.merge(all_df, extreme_df, on='year', how='inner')
df = df.sort_values('year').reset_index(drop=True)
if len(df) != 20:
    raise ValueError(f'Expected 20 matched years, found {len(df)}.')
df['difference_all'] = df['zeta2_longitude_all'] - df['zeta2_latitude_all']
df['difference_extreme'] = df['zeta2_longitude_extreme'] - df['zeta2_latitude_extreme']
df['change_longitude'] = df['zeta2_longitude_extreme'] - df['zeta2_longitude_all']
df['change_latitude'] = df['zeta2_latitude_extreme'] - df['zeta2_latitude_all']
df['change_difference'] = df['difference_extreme'] - df['difference_all']
df['percent_change_longitude'] = df['change_longitude'] / df['zeta2_longitude_all'] * 100
df['percent_change_latitude'] = df['change_latitude'] / df['zeta2_latitude_all'] * 100
df.to_csv(COMPARISON_FILE, index=False)
mean_lon_all = df['zeta2_longitude_all'].mean()
mean_lon_extreme = df['zeta2_longitude_extreme'].mean()
mean_lat_all = df['zeta2_latitude_all'].mean()
mean_lat_extreme = df['zeta2_latitude_extreme'].mean()
mean_diff_all = df['difference_all'].mean()
mean_diff_extreme = df['difference_extreme'].mean()
std_lon_all = df['zeta2_longitude_all'].std()
std_lon_extreme = df['zeta2_longitude_extreme'].std()
std_lat_all = df['zeta2_latitude_all'].std()
std_lat_extreme = df['zeta2_latitude_extreme'].std()
std_diff_all = df['difference_all'].std()
std_diff_extreme = df['difference_extreme'].std()
absolute_change_lon = mean_lon_extreme - mean_lon_all
absolute_change_lat = mean_lat_extreme - mean_lat_all
absolute_change_diff = mean_diff_extreme - mean_diff_all
percent_change_lon = absolute_change_lon / mean_lon_all * 100
percent_change_lat = absolute_change_lat / mean_lat_all * 100
t_lon, p_lon = ttest_rel(df['zeta2_longitude_extreme'], df['zeta2_longitude_all'])
t_lat, p_lat = ttest_rel(df['zeta2_latitude_extreme'], df['zeta2_latitude_all'])
t_diff, p_diff = ttest_rel(df['difference_extreme'], df['difference_all'])
r_lon, p_corr_lon = pearsonr(df['zeta2_longitude_all'], df['zeta2_longitude_extreme'])
r_lat, p_corr_lat = pearsonr(df['zeta2_latitude_all'], df['zeta2_latitude_extreme'])
n_lon_increase = np.sum(df['zeta2_longitude_extreme'] > df['zeta2_longitude_all'])
n_lat_increase = np.sum(df['zeta2_latitude_extreme'] > df['zeta2_latitude_all'])
n_difference_increase = np.sum(df['difference_extreme'] > df['difference_all'])
summary = pd.DataFrame({'metric': ['Longitude zeta2 - all rainfall', 'Longitude zeta2 - extreme rainfall', 'Longitude absolute change', 'Longitude percentage change', 'Latitude zeta2 - all rainfall', 'Latitude zeta2 - extreme rainfall', 'Latitude absolute change', 'Latitude percentage change', 'Directional difference - all rainfall', 'Directional difference - extreme rainfall', 'Directional difference absolute change', 'Longitude years extreme > all', 'Latitude years extreme > all', 'Difference years extreme > all', 'Longitude paired t statistic', 'Longitude paired p value', 'Latitude paired t statistic', 'Latitude paired p value', 'Directional difference paired t statistic', 'Directional difference paired p value', 'Longitude Pearson r', 'Longitude correlation p value', 'Latitude Pearson r', 'Latitude correlation p value'], 'value': [mean_lon_all, mean_lon_extreme, absolute_change_lon, percent_change_lon, mean_lat_all, mean_lat_extreme, absolute_change_lat, percent_change_lat, mean_diff_all, mean_diff_extreme, absolute_change_diff, n_lon_increase, n_lat_increase, n_difference_increase, t_lon, p_lon, t_lat, p_lat, t_diff, p_diff, r_lon, p_corr_lon, r_lat, p_corr_lat]})
summary.to_csv(SUMMARY_FILE, index=False)
tests = pd.DataFrame({'comparison': ['Longitude zeta2', 'Latitude zeta2', 'Directional difference'], 'mean_all': [mean_lon_all, mean_lat_all, mean_diff_all], 'mean_extreme': [mean_lon_extreme, mean_lat_extreme, mean_diff_extreme], 'absolute_change': [absolute_change_lon, absolute_change_lat, absolute_change_diff], 'percent_change': [percent_change_lon, percent_change_lat, absolute_change_diff / abs(mean_diff_all) * 100], 't_statistic': [t_lon, t_lat, t_diff], 'p_value': [p_lon, p_lat, p_diff]})
tests.to_csv(TESTS_FILE, index=False)
plt.figure(figsize=(8, 6))
plt.plot(df['year'], df['zeta2_longitude_all'], marker='o', linewidth=2, label='All rainfall')
plt.plot(df['year'], df['zeta2_longitude_extreme'], marker='o', linewidth=2, label='Extreme rainfall')
plt.xlabel('Year')
plt.ylabel('$\\zeta_2$')
plt.title('Longitudinal spatial scaling: all vs extreme rainfall')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'extreme_vs_all_zeta2_longitude_2001_2020.png', dpi=300)
plt.close()
plt.figure(figsize=(8, 6))
plt.plot(df['year'], df['zeta2_latitude_all'], marker='o', linewidth=2, label='All rainfall')
plt.plot(df['year'], df['zeta2_latitude_extreme'], marker='o', linewidth=2, label='Extreme rainfall')
plt.xlabel('Year')
plt.ylabel('$\\zeta_2$')
plt.title('Latitudinal spatial scaling: all vs extreme rainfall')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'extreme_vs_all_zeta2_latitude_2001_2020.png', dpi=300)
plt.close()
plt.figure(figsize=(9, 6))
plt.axhline(0, linewidth=1)
plt.plot(df['year'], df['change_longitude'], marker='o', linewidth=2, label='Longitude')
plt.plot(df['year'], df['change_latitude'], marker='o', linewidth=2, label='Latitude')
plt.xlabel('Year')
plt.ylabel('Extreme $\\zeta_2$ − All-rainfall $\\zeta_2$')
plt.title('Change in spatial scaling exponent during extreme rainfall')
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(PLOTS_DIR / 'extreme_minus_all_zeta2_2001_2020.png', dpi=300)
plt.close()
