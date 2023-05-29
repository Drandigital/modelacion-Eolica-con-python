import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Carga los datos (reemplaza esto con tus datos reales)
data = pd.read_csv("datos_eolicos.csv")

# Define los bordes de los compartimentos de la rosa de vientos
bin_edges = np.arange(-7.5, 367.5, 22.5)
num_bins = len(bin_edges) - 1

# Cuenta la frecuencia de las direcciones del viento
wind_freq, _ = np.histogram(data['direccion_viento'], bins=bin_edges)

# Normaliza las frecuencias
wind_freq_norm = wind_freq / np.sum(wind_freq)

# Define los ángulos y las etiquetas de la rosa de vientos
angles = np.linspace(0, 2 * np.pi, num_bins, endpoint=False)
labels = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

# Crea la gráfica de la rosa de vientos
fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

ax.bar(angles, wind_freq_norm, width=2 * np.pi / num_bins, alpha=0.6)
ax.set_thetagrids(angles * 180 / np.pi, labels)
ax.set_rlabel_position(-112.5)
ax.set_yticklabels([])
ax.set_xticklabels(labels)

plt.show()
