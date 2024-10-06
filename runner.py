# this works

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

df_stars['b_new_rad'] = np.arcsin(np.clip(df_stars['z_rel'] / df_stars['d_new'], -1, 1))
df_stars['l_new_rad'] = np.arctan2(df_stars['y_rel'], df_stars['x_rel'])
df_stars['l_new_rad'] = df_stars['l_new_rad'] % (2 * np.pi)
df_stars['galactic_longitude_new'] = np.rad2deg(df_stars['l_new_rad'])
df_stars['galactic_latitude_new'] = np.rad2deg(df_stars['b_new_rad'])

df_stars['M'] = df_stars['visual_magnitude'] - 5 * (np.log10(df_stars['distance_pc']) - 1)

df_stars['visual_magnitude_new'] = df_stars['M'] + 5 * (np.log10(df_stars['d_new']) - 1)

magnitude_limit = 15
df_visible_stars = df_stars[df_stars['visual_magnitude_new'] <= magnitude_limit].copy()

if not df_visible_stars.empty:
    # Correct m_ref and brightness calculation
    m_ref = df_visible_stars['visual_magnitude_new'].min()
    df_visible_stars['brightness'] = 10 ** (-0.4 * (df_visible_stars['visual_magnitude_new'] - m_ref))

    # Adjust size_scale and cap sizes
    size_scale = 100  # Adjust this value as needed
    df_visible_stars['size'] = df_visible_stars['brightness'] * size_scale
    df_visible_stars['size'] = df_visible_stars['size'].clip(lower=0.1, upper=50)

    # Drop NaN values if any
    df_visible_stars.dropna(subset=['galactic_longitude_new', 'galactic_latitude_new', 'size'], inplace=True)

    # Print Data Ranges for Verification
    print(f"Number of visible stars: {df_visible_stars.shape[0]}")
    print("Galactic Longitude New Range:", df_visible_stars['galactic_longitude_new'].min(), "-", df_visible_stars['galactic_longitude_new'].max())
    print("Galactic Latitude New Range:", df_visible_stars['galactic_latitude_new'].min(), "-", df_visible_stars['galactic_latitude_new'].max())

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.scatter(
        df_visible_stars['galactic_longitude_new'],
        df_visible_stars['galactic_latitude_new'],
        s=df_visible_stars['size'],
        color='white',
        edgecolors='none'  # or 'face'
    )

    plt.xlabel('Galactic Longitude (degrees)')
    plt.ylabel('Galactic Latitude (degrees)')
    plt.title(f"Sky as Seen from Exoplanet {exoplanet_name}")

    # Set axes limits to standard galactic coordinate ranges
    plt.xlim(0, 360)
    plt.ylim(-90, 90)
    plt.gca().invert_xaxis()
    plt.gca().set_facecolor('black')
    plt.show()
else:
    print("No visible stars found with the current magnitude limit.")
