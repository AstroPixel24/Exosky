import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_stars = pd.read_csv('stars_data_cleaned2.csv')  

numeric_columns = ['galactic_longitude', 'galactic_latitude', 'visual_magnitude', 'distance']
df_stars[numeric_columns] = df_stars[numeric_columns].apply(pd.to_numeric, errors='coerce')

df_stars.dropna(subset=numeric_columns, inplace=True)
df_stars.reset_index(drop=True, inplace=True)

df_stars['distance_pc'] = df_stars['distance'] 

df_stars['l_rad'] = np.deg2rad(df_stars['galactic_longitude'])
df_stars['b_rad'] = np.deg2rad(df_stars['galactic_latitude'])

df_stars['x_star'] = df_stars['distance_pc'] * np.cos(df_stars['b_rad']) * np.cos(df_stars['l_rad'])
df_stars['y_star'] = df_stars['distance_pc'] * np.cos(df_stars['b_rad']) * np.sin(df_stars['l_rad'])
df_stars['z_star'] = df_stars['distance_pc'] * np.sin(df_stars['b_rad'])

df_exoplanets = pd.read_csv('exoplanet_data_cleaned.csv')
required_columns = ['pl_name', 'galactic_latitude', 'galactic_longitude', 'distance', 'visual_mag']
df_exoplanets = df_exoplanets[required_columns]
df_exoplanets.dropna(subset=required_columns, inplace=True)
numeric_columns_exo = ['galactic_latitude', 'galactic_longitude', 'distance', 'visual_mag']
df_exoplanets[numeric_columns_exo] = df_exoplanets[numeric_columns_exo].apply(pd.to_numeric, errors='coerce')
df_exoplanets.dropna(subset=numeric_columns_exo, inplace=True)
df_exoplanets.reset_index(drop=True, inplace=True)

exoplanet = df_exoplanets.iloc[0] # change depending on which exoplanet you want

l_exo_deg = exoplanet['galactic_longitude']
b_exo_deg = exoplanet['galactic_latitude']
d_exo_pc = exoplanet['distance']
exoplanet_name = exoplanet['pl_name']

l_exo_rad = np.deg2rad(l_exo_deg)
b_exo_rad = np.deg2rad(b_exo_deg)

x_exo = d_exo_pc * np.cos(b_exo_rad) * np.cos(l_exo_rad)
y_exo = d_exo_pc * np.cos(b_exo_rad) * np.sin(l_exo_rad)
z_exo = d_exo_pc * np.sin(b_exo_rad)

df_stars['x_rel'] = df_stars['x_star'] - x_exo
df_stars['y_rel'] = df_stars['y_star'] - y_exo
df_stars['z_rel'] = df_stars['z_star'] - z_exo

df_stars['d_new'] = np.sqrt(df_stars['x_rel']**2 + df_stars['y_rel']**2 + df_stars['z_rel']**2)
df_stars = df_stars[df_stars['d_new'] > 0]

df_stars['b_new_rad'] = np.arcsin(df_stars['z_rel'] / df_stars['d_new'])
df_stars['l_new_rad'] = np.arctan2(df_stars['y_rel'], df_stars['x_rel'])
df_stars['l_new_rad'] = df_stars['l_new_rad'] % (2 * np.pi)
df_stars['galactic_longitude_new'] = np.rad2deg(df_stars['l_new_rad'])
df_stars['galactic_latitude_new'] = np.rad2deg(df_stars['b_new_rad'])

df_stars['M'] = df_stars['visual_magnitude'] - 5 * (np.log10(df_stars['distance_pc']) - 1)

df_stars['visual_magnitude_new'] = df_stars['M'] + 5 * (np.log10(df_stars['d_new']) - 1)

magnitude_limit = 15
df_visible_stars = df_stars[df_stars['visual_magnitude_new'] <= magnitude_limit].copy()

m_ref = magnitude_limit  
df_visible_stars['brightness'] = 10 ** (-0.4 * (df_visible_stars['visual_magnitude_new'] - m_ref))

size_scale = 50 
df_visible_stars['size'] = df_visible_stars['brightness'] * size_scale

df_visible_stars['size'] = df_visible_stars['size'].clip(lower=0.1)
print(df_visible_stars[['galactic_longitude_new', 'galactic_latitude_new', 'visual_magnitude_new', 'size']].head())

df_visible_stars[['hipparcos_star_name', 'galactic_latitude_new', 'galactic_longitude_new', 'brightness']].to_csv('output_stars.csv', index=False)
