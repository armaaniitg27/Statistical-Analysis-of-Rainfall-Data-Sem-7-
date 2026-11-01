import os
import numpy as np
import pandas as pd
from scipy.stats import linregress
RESULTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\results'
PLOTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\plots'
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
Q_VALUES = np.array([1, 2, 3, 4, 5], dtype=float)
WINDOWS = {'longitude': [(80, 135), (90, 145), (100, 155), (110, 165), (120, 175), (130, 185), (140, 200), (150, 210), (155, 207), (160, 220), (170, 230), (180, 240), (190, 250)], 'latitude': [(78, 134), (90, 145), (100, 155), (100, 167), (100, 189), (110, 167), (110, 180), (120, 200), (130, 220), (140, 230), (145, 289), (150, 250), (160, 270), (170, 280)]}

def fit_linear(q, zeta):
    slope, intercept, r_value, p_value, std_err = linregress(q, zeta)
    predicted = intercept + slope * q
    residuals = zeta - predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((zeta - np.mean(zeta)) ** 2)
    if ss_tot > 0:
        r2 = 1 - ss_res / ss_tot
    else:
        r2 = np.nan
    rmse = np.sqrt(np.mean(residuals ** 2))
    return (intercept, slope, r2, rmse)

def fit_quadratic(q, zeta):
    coefficients = np.polyfit(q, zeta, 2)
    c = coefficients[0]
    b = coefficients[1]
    a = coefficients[2]
    predicted = a + b * q + c * q ** 2
    residuals = zeta - predicted
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((zeta - np.mean(zeta)) ** 2)
    if ss_tot > 0:
        r2 = 1 - ss_res / ss_tot
    else:
        r2 = np.nan
    rmse = np.sqrt(np.mean(residuals ** 2))
    return (a, b, c, r2, rmse)

def analyse_window(df, q_values, min_distance, max_distance):
    mask = (df['distance_km'] >= min_distance) & (df['distance_km'] <= max_distance)
    subset = df.loc[mask].copy()
    if len(subset) < 3:
        return None
    distances = subset['distance_km'].values
    results = []
    for q in q_values:
        column = f'S{int(q)}'
        if column not in subset.columns:
            continue
        valid = np.isfinite(subset[column].values) & (subset[column].values > 0) & np.isfinite(distances) & (distances > 0)
        if np.sum(valid) < 3:
            continue
        x = np.log10(distances[valid])
        y = np.log10(subset.loc[valid, column].values)
        fit = linregress(x, y)
        zeta = fit.slope
        r2 = fit.rvalue ** 2
        results.append({'q': q, 'zeta': zeta, 'r2_scaling': r2})
    if len(results) < 3:
        return None
    zeta_df = pd.DataFrame(results)
    q = zeta_df['q'].values
    zeta = zeta_df['zeta'].values
    a_lin, b_lin, r2_lin, rmse_lin = fit_linear(q, zeta)
    a_quad, b_quad, c_quad, r2_quad, rmse_quad = fit_quadratic(q, zeta)
    delta_r2 = r2_quad - r2_lin
    n = len(q)
    rss_lin = np.sum((zeta - (a_lin + b_lin * q)) ** 2)
    rss_quad = np.sum((zeta - (a_quad + b_quad * q + c_quad * q ** 2)) ** 2)
    if rss_lin > 0:
        aic_lin = n * np.log(rss_lin / n) + 2 * 2
    else:
        aic_lin = -np.inf
    if rss_quad > 0:
        aic_quad = n * np.log(rss_quad / n) + 2 * 3
    else:
        aic_quad = -np.inf
    delta_aic = aic_lin - aic_quad
    return {'distance_min_km': distances.min(), 'distance_max_km': distances.max(), 'n_scales': len(zeta_df), 'zeta_q1': zeta_df.loc[zeta_df['q'] == 1, 'zeta'].iloc[0] if 1 in q else np.nan, 'zeta_q2': zeta_df.loc[zeta_df['q'] == 2, 'zeta'].iloc[0] if 2 in q else np.nan, 'zeta_q3': zeta_df.loc[zeta_df['q'] == 3, 'zeta'].iloc[0] if 3 in q else np.nan, 'zeta_q4': zeta_df.loc[zeta_df['q'] == 4, 'zeta'].iloc[0] if 4 in q else np.nan, 'zeta_q5': zeta_df.loc[zeta_df['q'] == 5, 'zeta'].iloc[0] if 5 in q else np.nan, 'linear_intercept': a_lin, 'linear_slope': b_lin, 'linear_R2': r2_lin, 'linear_RMSE': rmse_lin, 'quadratic_intercept': a_quad, 'quadratic_linear_term': b_quad, 'quadratic_term': c_quad, 'quadratic_R2': r2_quad, 'quadratic_RMSE': rmse_quad, 'delta_R2': delta_r2, 'AIC_linear': aic_lin, 'AIC_quadratic': aic_quad, 'delta_AIC': delta_aic}
FILES = {'longitude': os.path.join(RESULTS_DIR, 'structure_functions_longitude_2005.csv'), 'latitude': os.path.join(RESULTS_DIR, 'structure_functions_latitude_2005.csv')}
all_results = []
for direction in ['longitude', 'latitude']:
    df = pd.read_csv(FILES[direction])
    for window_number, (dmin, dmax) in enumerate(WINDOWS[direction], start=1):
        result = analyse_window(df, Q_VALUES, dmin, dmax)
        if result is None:
            continue
        result['direction'] = direction
        result['window_number'] = window_number
        all_results.append(result)
results_df = pd.DataFrame(all_results)
output_file = os.path.join(RESULTS_DIR, 'zeta_nonlinearity_tests_2005.csv')
results_df.to_csv(output_file, index=False)
for direction in ['longitude', 'latitude']:
    subset = results_df[results_df['direction'] == direction].copy()
    subset = subset.sort_values('delta_AIC', ascending=False)
