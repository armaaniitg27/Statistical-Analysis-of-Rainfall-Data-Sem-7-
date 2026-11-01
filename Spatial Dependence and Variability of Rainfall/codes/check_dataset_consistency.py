import os
import pandas as pd
DATA_ROOT = 'D:\\SBS project\\vikas'
results = []
for year in range(2001, 2021):
    folder = os.path.join(DATA_ROOT, f'kcsv{year}')
    file_path = os.path.join(folder, 'ktest2.csv')
    if not os.path.exists(folder):
        continue
    if not os.path.exists(file_path):
        continue
    df = pd.read_csv(file_path, usecols=['lat', 'lon', 'time'])
    latitudes = sorted(df['lat'].unique())
    longitudes = sorted(df['lon'].unique())
    results.append({'Year': year, 'Rows': len(df), 'Lat points': len(latitudes), 'Lon points': len(longitudes), 'Lat Min': df['lat'].min(), 'Lat Max': df['lat'].max(), 'Lon Min': df['lon'].min(), 'Lon Max': df['lon'].max(), 'Timestamp': df['time'].iloc[0]})
summary = pd.DataFrame(results)
if len(summary) > 0:
    for i in range(1, len(summary)):
        previous = summary.iloc[i - 1]
        current = summary.iloc[i]
        changes = []
        for column in ['Rows', 'Lat points', 'Lon points', 'Lat Min', 'Lat Max', 'Lon Min', 'Lon Max']:
            if previous[column] != current[column]:
                changes.append(f'{column}: {previous[column]} -> {current[column]}')
        if changes:
            for change in changes:
                pass
