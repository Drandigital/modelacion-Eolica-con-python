import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Cargar datos
data = pd.read_csv("datos_eolicos.csv")
wind_speed = data['velocidad_viento']
height = data['altura']

# Estimar el exponente de la Ley de Potencia del Viento (α)
alpha = 0.143  # Asumiendo un valor típico para un terreno abierto

# Calcular las velocidades del viento a la altura de interés usando la Ley de Potencia del Viento
reference_height = 10  # Altura de referencia en metros
target_height = 80  # Altura objetivo en metros (por ejemplo, altura de las palas del aerogenerador)
wind_speed_target_height = wind_speed * (target_height / reference_height)**alpha

# Calcular el potencial eólico teórico usando la Ley de Potencia del Viento
air_density = 1.225  # kg/m³, densidad del aire al nivel del mar y 15°C
theoretical_wind_power = 0.5 * air_density * np.power(wind_speed_target_height, 3)

# Comparar las predicciones de potencial eólico de los modelos generados con las predicciones teóricas
# Asumiendo que y_pred_linear, y_pred_poly y y_pred_lagrange son las predicciones de potencial eólico
# obtenidas de los modelos generados (regresión lineal, regresión polinomial y polinomios de interpolación de Lagrange)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(data[['velocidad_viento']], theoretical_wind_power, test_size=0.2, random_state=42)

# Definir el modelo de regresión lineal
linear_model = LinearRegression()

# Ajustar el modelo a los datos de entrenamiento
linear_model.fit(X_train, y_train)

# Definir el modelo de regresión polinómica
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)

# Transformar las características de entrenamiento
X_train_poly = poly_features.fit_transform(X_train)

# Ajustar el modelo a los datos de entrenamiento
poly_model.fit(X_train_poly, y_train)

def lagrange_interpolation2(x, y, x_val):
    n = len(x)
    L = 0
    x = x.to_numpy()  # Convertir x en un array de numpy para evitar errores de indexación
    y = y.to_numpy()  # Convertir y en un array de numpy para evitar errores de indexación
    for i in range(n):
        Li = 1
        for j in range(n):
            if i != j:
                Li *= (x_val - x[j]) / (x[i] - x[j])
        L += y[i] * Li
    return L



models = ['Regresión lineal', 'Regresión polinomial', 'Polinomios de interpolación de Lagrange']


y_pred_linear_all = linear_model.predict(data[['velocidad_viento']])
y_pred_poly_all = poly_model.predict(poly_features.fit_transform(data[['velocidad_viento']]))
y_pred_lagrange_all = np.array([lagrange_interpolation2(X_train['velocidad_viento'], y_train, x_val) for x_val in data['velocidad_viento']])

predictions = [y_pred_linear_all, y_pred_poly_all, y_pred_lagrange_all]

for model, prediction in zip(models, predictions):
    mse = mean_squared_error(theoretical_wind_power, prediction)
    r2 = r2_score(theoretical_wind_power, prediction)
    print(f"{model}:")
    print(f"Error Cuadratico: {mse:.2f}")
    print(f"R^2 Score: {r2:.2f}\n")



plt.figure(figsize=(10, 8))
plt.scatter(wind_speed, theoretical_wind_power, label='Potencial eólico teórico')
plt.scatter(wind_speed, y_pred_linear_all, label='Regresión lineal')
plt.scatter(wind_speed, y_pred_poly_all, label='Regresión polinomial')
plt.scatter(wind_speed, y_pred_lagrange_all, label='Polinomios de interpolación de Lagrange')
plt.xlabel('Velocidad del viento (m/s)')
plt.ylabel('Potencia del viento (W)')
plt.title('Comparación de modelos de predicción de potencial eólico')
plt.legend()
plt.show()
