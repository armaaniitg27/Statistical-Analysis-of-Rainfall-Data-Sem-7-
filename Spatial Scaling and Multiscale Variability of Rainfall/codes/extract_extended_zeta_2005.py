import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress
RESULTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\results'
PLOTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\plots'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
Q_VALUES = np.arange(0.5, 8.5, 0.5)
WINDOWS = {'longitude': [(80, 135), (90, 145), (100, 155), (110, 165), (120, 175), (130, 185), (140, 200), (150, 210), (155, 207), (160, 220), (170, 230), (180, 240), (190, 250)], 'latitude': [(78, 134), (90, 145), (100, 155), (100, 167), (110, 167), (110, 180), (120, 190), (130, 210), (140, 225), (145, 245), (150, 250), (155, 275), (165, 267), (175, 278)]}

def calculate_zeta(df, q_values, dmin, dmax):
    mask = (df['distance_km'] >= dmin) & (df['distance_km'] <= dmax)
    subset = df.loc[mask].copy()
    if len(subset) < 3:
        return None
    distances = subset['distance_km'].values
    zeta_results = []
    for q in q_values:
        column = f'S{q:g}'
        if column not in subset.columns:
            continue
        values = subset[column].values
        valid = np.isfinite(distances) & (distances > 0) & np.isfinite(values) & (values > 0)
        if np.sum(valid) < 3:
            continue
        x = np.log10(distances[valid])
        y = np.log10(values[valid])
        fit = linregress(x, y)
        zeta = fit.slope
        r2 = fit.rvalue ** 2
        zeta_results.append({'q': q, 'zeta': zeta, 'R2_scaling': r2})
    if len(zeta_results) < 3:
        return None
    zeta_df = pd.DataFrame(zeta_results)
    return zeta_df

def fit_zeta_models(zeta_df):
    q = zeta_df['q'].values
    zeta = zeta_df['zeta'].values
    linear_fit = linregress(q, zeta)
    a_linear = linear_fit.intercept
    b_linear = linear_fit.slope
    predicted_linear = a_linear + b_linear * q
    rss_linear = np.sum((zeta - predicted_linear) ** 2)
    tss = np.sum((zeta - np.mean(zeta)) ** 2)
    if tss > 0:
        r2_linear = 1 - rss_linear / tss
    else:
        r2_linear = np.nan
    rmse_linear = np.sqrt(np.mean((zeta - predicted_linear) ** 2))
    coefficients = np.polyfit(q, zeta, 2)
    c_quad = coefficients[0]
    b_quad = coefficients[1]
    a_quad = coefficients[2]
    predicted_quad = a_quad + b_quad * q + c_quad * q ** 2
    rss_quad = np.sum((zeta - predicted_quad) ** 2)
    if tss > 0:
        r2_quad = 1 - rss_quad / tss
    else:
        r2_quad = np.nan
    rmse_quad = np.sqrt(np.mean((zeta - predicted_quad) ** 2))
    n = len(q)
    if rss_linear > 0:
        aic_linear = n * np.log(rss_linear / n) + 2 * 2
    else:
        aic_linear = -np.inf
    if rss_quad > 0:
        aic_quad = n * np.log(rss_quad / n) + 2 * 3
    else:
        aic_quad = -np.inf
    delta_aic = aic_linear - aic_quad
    return {'linear_intercept': a_linear, 'linear_slope': b_linear, 'linear_R2': r2_linear, 'linear_RMSE': rmse_linear, 'quadratic_intercept': a_quad, 'quadratic_linear_term': b_quad, 'quadratic_term': c_quad, 'quadratic_R2': r2_quad, 'quadratic_RMSE': rmse_quad, 'delta_R2': r2_quad - r2_linear, 'AIC_linear': aic_linear, 'AIC_quadratic': aic_quad, 'delta_AIC': delta_aic}

def analyse_direction(direction, input_file):
    df = pd.read_csv(input_file)
    all_zeta = []
    all_window_summary = []
    for window_number, (dmin, dmax) in enumerate(WINDOWS[direction], start=1):
        zeta_df = calculate_zeta(df, Q_VALUES, dmin, dmax)
        if zeta_df is None:
            continue
        model_results = fit_zeta_models(zeta_df)
        mean_scaling_r2 = zeta_df['R2_scaling'].mean()
        min_scaling_r2 = zeta_df['R2_scaling'].min()
        for _, row in zeta_df.iterrows():
            all_zeta.append({'direction': direction, 'window_number': window_number, 'distance_min_km': dmin, 'distance_max_km': dmax, 'q': row['q'], 'zeta': row['zeta'], 'R2_scaling': row['R2_scaling']})
        summary = {'direction': direction, 'window_number': window_number, 'distance_min_km': dmin, 'distance_max_km': dmax, 'n_scales': len(zeta_df), 'mean_scaling_R2': mean_scaling_r2, 'min_scaling_R2': min_scaling_r2}
        summary.update(model_results)
        all_window_summary.append(summary)
    return (pd.DataFrame(all_zeta), pd.DataFrame(all_window_summary))
longitude_input = os.path.join(RESULTS_DIR, 'structure_functions_longitude_2005_q_extended.csv')
longitude_zeta, longitude_summary = analyse_direction('longitude', longitude_input)
latitude_input = os.path.join(RESULTS_DIR, 'structure_functions_latitude_2005_q_extended.csv')
latitude_zeta, latitude_summary = analyse_direction('latitude', latitude_input)
zeta_results = pd.concat([longitude_zeta, latitude_zeta], ignore_index=True)
summary_results = pd.concat([longitude_summary, latitude_summary], ignore_index=True)
zeta_output = os.path.join(RESULTS_DIR, 'extended_zeta_2005.csv')
summary_output = os.path.join(RESULTS_DIR, 'extended_zeta_window_summary_2005.csv')
zeta_results.to_csv(zeta_output, index=False)
summary_results.to_csv(summary_output, index=False)
for direction in ['longitude', 'latitude']:
    subset = summary_results[summary_results['direction'] == direction].copy()
    subset = subset[subset['mean_scaling_R2'] >= 0.95]
    subset = subset.sort_values(['mean_scaling_R2', 'min_scaling_R2'], ascending=False)

def plot_zeta_curves(direction, zeta_df, summary_df):
    plt.figure(figsize=(10, 7))
    good_windows = summary_df[summary_df['mean_scaling_R2'] >= 0.95]
    for _, window in good_windows.iterrows():
        wn = window['window_number']
        subset = zeta_df[zeta_df['window_number'] == wn]
        plt.plot(subset['q'], subset['zeta'], marker='o', linewidth=1.5, alpha=0.65, label=f"{window['distance_min_km']:.0f}-{window['distance_max_km']:.0f} km")
    plt.xlabel('Order q')
    plt.ylabel('$\\zeta(q)$')
    plt.title(f'Spatial Scaling Exponents - {direction.capitalize()}, 2005')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    output = os.path.join(PLOTS_DIR, f'extended_zeta_{direction}_2005.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
plot_zeta_curves('longitude', longitude_zeta, longitude_summary)
plot_zeta_curves('latitude', latitude_zeta, latitude_summary)

def plot_scaling_r2(direction, zeta_df):
    plt.figure(figsize=(10, 7))
    for window_number in sorted(zeta_df['window_number'].unique()):
        subset = zeta_df[zeta_df['window_number'] == window_number]
        plt.plot(subset['q'], subset['R2_scaling'], marker='o', linewidth=1.2, alpha=0.6, label=f'Window {window_number}')
    plt.axhline(0.95, linestyle='--', linewidth=1)
    plt.xlabel('Order q')
    plt.ylabel('$R^2$ of $\\log S_q(r)$ vs $\\log r$')
    plt.title(f'Scaling-Fit Quality - {direction.capitalize()}, 2005')
    plt.ylim(0, 1.02)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    output = os.path.join(PLOTS_DIR, f'extended_scaling_R2_{direction}_2005.png')
    plt.savefig(output, dpi=300, bbox_inches='tight')
    plt.close()
plot_scaling_r2('longitude', longitude_zeta)
plot_scaling_r2('latitude', latitude_zeta)
