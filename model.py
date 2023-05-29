import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Cargar datos
data = pd.read_csv("datos_eolicos.csv")
X = data[['temperatura', 'humedad', 'velocidad_viento']]

# Convertir la velocidad del viento a la energía eólica (suponiendo que la densidad del aire es constante)
air_density = 1.135  # kg/m³, densidad del aire al nivel del mar y 28.5°C (temperatura promedio de Cartagena) Resultado en Watts (W)
y = 0.5 * air_density * np.power(X['velocidad_viento'], 3)

# Imprimir el potencial eólico promedio
print("Potencial eólico promedio:", y.mean())

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Regresión lineal
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)

# Regresión polinomial de segundo grado
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# Métricas de evaluación
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

mse_poly = mean_squared_error(y_test, y_pred_poly)
r2_poly = r2_score(y_test, y_pred_poly)

print("Regresión lineal:")
print("Error cuadrático medio:", mse_linear)
print("R^2 Score:", r2_linear)

print("\nRegresión polinomial:")
print("Mean Squared Error:", mse_poly)
print("R^2 Score:", r2_poly)

# Estadísticas
y_mean = np.mean(y_test)
total_standard_deviation = np.std(y_test)
standard_error_estimate_linear = np.sqrt(mean_squared_error(y_test, y_pred_linear))
standard_error_estimate_poly = np.sqrt(mean_squared_error(y_test, y_pred_poly))
correlation_coefficient_linear = np.sqrt(r2_score(y_test, y_pred_linear))
correlation_coefficient_poly = np.sqrt(r2_score(y_test, y_pred_poly))
coefficient_of_determination_linear = r2_score(y_test, y_pred_linear)
coefficient_of_determination_poly = r2_score(y_test, y_pred_poly)

print("Desviación estándar total:", total_standard_deviation)
print("\nRegresión lineal:")
print("Error estándar del estimado:", standard_error_estimate_linear)
print("Coeficiente de correlación:", correlation_coefficient_linear)
print("Coeficiente de determinación:", coefficient_of_determination_linear)

print("\nRegresión polinomial de segundo grado:")
print("Error estándar del estimado:", standard_error_estimate_poly)
print("Coeficiente de correlación:", correlation_coefficient_poly)
print("Coeficiente de determinación:", coefficient_of_determination_poly)

# Gráfico de Velocidad del viento vs Energía eólica
plt.scatter(X['velocidad_viento'], y, color='blue')
plt.xlabel('Velocidad del viento (m/s)')
plt.ylabel('Energía eólica (W)')
plt.title('Velocidad del viento vs Energía eólica')
plt.show()

# Observaciones: La energía eólica parece aumentar con la velocidad del viento.

# Gráfico de Temperatura vs Energía eólica
plt.scatter(X['temperatura'], y, color='red')
plt.xlabel('Temperatura (°C)')
plt.ylabel('Energía eólica (W)')
plt.title('Temperatura vs Energía eólica')
plt.show()

# Observaciones: No se observa una correlación clara entre la temperatura y la energía eólica.

# Gráfico de Humedad vs Energía eólica
plt.scatter(X['humedad'], y, color='green')
plt.xlabel('Humedad (%)')
plt.ylabel('Energía eólica (W)')
plt.title('Humedad vs Energía eólica')
plt.show()

# Graficar los resultados
plt.scatter(X_test['velocidad_viento'], y_test, color='blue', label='Datos reales')
plt.scatter(X_test['velocidad_viento'], y_pred_linear, color='red', label='Regresión lineal')
plt.scatter(X_test['velocidad_viento'], y_pred_poly, color='green', label='Regresión polinomial')
plt.xlabel('Velocidad del viento')
plt.ylabel('Energía eólica')
plt.legend()
plt.show()

# Crear un nuevo DataFrame para almacenar los resultados
results = pd.DataFrame({'Velocidad del viento': X_test['velocidad_viento'],
                        'Energía eólica': y_test,
                        'Predicción lineal': y_pred_linear,
                        'Predicción polinomial': y_pred_poly})

# Guardar los resultados en un nuevo archivo CSV
results.to_csv('resultados_energia_eolica.csv', index=False)

def lagrange_interpolation(x, y, x_val):
    n = len(x)
    y_interpolated = 0
    for i in range(n):
        L = 1
        for j in range(n):
            if i != j:
                L *= (x_val - x[j]) / (x[i] - x[j])
        y_interpolated += y[i] * L
    return y_interpolated

# Cargar datos
data = pd.read_csv("datos_eolicos.csv")
X = data['velocidad_viento'].tolist()
air_density = 1.135  # kg/m³, densidad del aire al nivel del mar y 28.5°C (temperatura promedio de Cartagena) Resultado en Watts (W)
y = (0.5 * air_density * np.power(X, 3)).tolist()

# Separar datos impares y pares
odd_indices = [i for i in range(0, len(X), 2)]
even_indices = [i for i in range(1, len(X), 2)]
X_odd, y_odd = [X[i] for i in odd_indices], [y[i] for i in odd_indices]
X_even, y_even = [X[i] for i in even_indices], [y[i] for i in even_indices]

# Grado máximo de interpolación
max_degree = len(X_odd)

# Calcular errores relativos porcentuales para cada grado de polinomios de Lagrange
errors = []
for degree in range(1, max_degree + 1):
    y_interpolated = [lagrange_interpolation(X_odd[:degree+1], y_odd[:degree+1], x_val) for x_val in X_even]
    relative_error_percentages = np.abs((np.array(y_even) - np.array(y_interpolated)) / np.array(y_even)) * 100
    errors.append(np.mean(relative_error_percentages))
    print(f"Grado {degree}: Error relativo porcentual promedio = {np.mean(relative_error_percentages)}")


# Graficar errores relativos porcentuales promedio en función del grado de interpolación
plt.plot(range(1, max_degree + 1), errors, marker='o')
plt.xlabel('Grado de interpolación')
plt.ylabel('Error relativo porcentual promedio')
plt.title('Polinomios de interpolación de Lagrange')
plt.show()


