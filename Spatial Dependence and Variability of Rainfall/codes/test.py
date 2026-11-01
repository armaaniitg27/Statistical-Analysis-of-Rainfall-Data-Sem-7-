import pandas as pd
import os
DATA_ROOT = 'D:\\SBS project\\vikas'
YEAR = 'kcsv2005'
FILE = 'ktest2.csv'
file_path = os.path.join(DATA_ROOT, YEAR, FILE)
df = pd.read_csv(file_path)
unique_lat = sorted(df['lat'].unique())
unique_lon = sorted(df['lon'].unique())
if len(unique_lat) > 1:
    lat_spacing = pd.Series(unique_lat).diff().dropna().unique()
if len(unique_lon) > 1:
    lon_spacing = pd.Series(unique_lon).diff().dropna().unique()
