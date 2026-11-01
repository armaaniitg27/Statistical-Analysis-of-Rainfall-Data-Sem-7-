import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
RESULTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\results'
PLOTS_DIR = 'D:\\SBS project\\sem 7\\chapter2\\plots'
INPUT_FILE = os.path.join(RESULTS_DIR, 'extended_zeta_2005.csv')
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
PRIMARY_WINDOWS = {'longitude': (80.0, 135.0), 'latitude': (78.0, 134.0)}
ROBUSTNESS_WINDOWS = {'longitude': [(80.0, 135.0), (90.0, 145.0), (100.0, 155.0), (110.0, 165.0), (120.0, 175.0)], 'latitude': [(78.0, 134.0), (90.0, 145.0), (100.0, 155.0), (100.0, 167.0), (110.0, 167.0)]}
df = pd.read_csv(INPUT_FILE)
if 'q' not in df.columns:
    raise ValueError("Input file must contain a column named 'q'.")
q_values = np.sort(df['q'].unique())

def analyse_spectrum(data, direction, rmin, rmax):
    direction_data = data[data['direction'].str.lower() == direction.lower()].copy()
    if len(direction_data) == 0:
        raise ValueError(f"No rows found for direction '{direction}'.")
    window_data = direction_data[np.isclose(direction_data['distance_min_km'], rmin) & np.isclose(direction_data['distance_max_km'], rmax)].copy()
    if len(window_data) == 0:
        window_data = direction_data[np.abs(direction_data['distance_min_km'] - rmin) < 1e-06]
        window_data = window_data[np.abs(window_data['distance_max_km'] - rmax) < 1e-06]
    if len(window_data) == 0:
        raise ValueError(f'No matching window found for {direction}: {rmin}-{rmax} km')
    window_data = window_data.sort_values('q')
    q = window_data['q'].to_numpy(dtype=float)
    zeta = window_data['zeta'].to_numpy(dtype=float)
    tau = zeta - 1.0
    h = np.gradient(tau, q)
    D_h = q * h - tau
    h_min = np.min(h)
    h_max = np.max(h)
    spectrum_width = h_max - h_min
    D_max = np.max(D_h)
    q_at_Dmax = q[np.argmax(D_h)]
    h_at_Dmax = h[np.argmax(D_h)]
    left_width = h_at_Dmax - h_min
    right_width = h_max - h_at_Dmax
    if left_width + right_width > 0:
        asymmetry = (right_width - left_width) / (right_width + left_width)
    else:
        asymmetry = np.nan
    result = pd.DataFrame({'direction': direction, 'distance_min_km': rmin, 'distance_max_km': rmax, 'q': q, 'zeta': zeta, 'tau': tau, 'h': h, 'D_h': D_h})
    summary = {'direction': direction, 'distance_min_km': rmin, 'distance_max_km': rmax, 'h_min': h_min, 'h_max': h_max, 'spectrum_width': spectrum_width, 'D_max': D_max, 'q_at_Dmax': q_at_Dmax, 'h_at_Dmax': h_at_Dmax, 'left_width': left_width, 'right_width': right_width, 'asymmetry': asymmetry}
    return (result, summary)
required_columns = {'direction', 'distance_min_km', 'distance_max_km', 'q', 'zeta'}
missing = required_columns - set(df.columns)
if missing:
    raise ValueError('\nThe input file is missing these columns:\n' + str(sorted(missing)) + '\n\nPlease show me the first few rows/columns of extended_zeta_2005.csv.')
all_spectra = []
all_summaries = []
for direction in ['longitude', 'latitude']:
    rmin, rmax = PRIMARY_WINDOWS[direction]
    spectrum, summary = analyse_spectrum(df, direction, rmin, rmax)
    all_spectra.append(spectrum)
    all_summaries.append(summary)
primary_spectrum_df = pd.concat(all_spectra, ignore_index=True)
primary_summary_df = pd.DataFrame(all_summaries)
primary_spectrum_file = os.path.join(RESULTS_DIR, 'multifractal_spectrum_2005.csv')
primary_summary_file = os.path.join(RESULTS_DIR, 'multifractal_spectrum_summary_2005.csv')
primary_spectrum_df.to_csv(primary_spectrum_file, index=False)
primary_summary_df.to_csv(primary_summary_file, index=False)
robustness_summaries = []
for direction in ['longitude', 'latitude']:
    for rmin, rmax in ROBUSTNESS_WINDOWS[direction]:
        spectrum, summary = analyse_spectrum(df, direction, rmin, rmax)
        robustness_summaries.append(summary)
robustness_df = pd.DataFrame(robustness_summaries)
robustness_file = os.path.join(RESULTS_DIR, 'multifractal_spectrum_robustness_2005.csv')
robustness_df.to_csv(robustness_file, index=False)
for direction in ['longitude', 'latitude']:
    spectrum = primary_spectrum_df[primary_spectrum_df['direction'] == direction]
    plt.figure(figsize=(9, 6))
    plt.plot(spectrum['h'], spectrum['D_h'], marker='o', linewidth=2, markersize=5)
    plt.xlabel('Generalized Hurst exponent $h(q)$', fontsize=12)
    plt.ylabel('Multifractal spectrum $D(h)$', fontsize=12)
    plt.title(f'Spatial Multifractal Spectrum — {direction.capitalize()} — 2005', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = os.path.join(PLOTS_DIR, f'multifractal_spectrum_{direction}_2005.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
for direction in ['longitude', 'latitude']:
    spectrum = primary_spectrum_df[primary_spectrum_df['direction'] == direction]
    plt.figure(figsize=(9, 6))
    plt.plot(spectrum['q'], spectrum['zeta'], marker='o', linewidth=2, markersize=5)
    plt.xlabel('Statistical order $q$', fontsize=12)
    plt.ylabel('Scaling exponent $\\zeta(q)$', fontsize=12)
    plt.title(f'Spatial Scaling Exponents — {direction.capitalize()} — 2005', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_file = os.path.join(PLOTS_DIR, f'multifractal_zeta_{direction}_2005.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
