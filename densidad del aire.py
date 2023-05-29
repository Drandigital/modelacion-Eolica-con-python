import math

def calcular_densidad_aire(temperatura_celsius, presion=101325, R_aire=287.05):
    """
    Calcular la densidad del aire en función de la temperatura y presión.

    :param temperatura_celsius: Temperatura en grados Celsius
    :type temperatura_celsius: float
    :param presion: Presión en Pascales (Pa), por defecto es la presión al nivel del mar (101325 Pa)
    :type presion: float
    :param R_aire: Constante de los gases para el aire seco (287.05 J/(kg·K))
    :type R_aire: float
    :return: Densidad del aire en kg/m^3
    :rtype: float
    """

    # Convertir la temperatura a Kelvin
    temperatura_kelvin = temperatura_celsius + 273.15

    # Calcular la densidad del aire utilizando la ecuación del gas ideal
    densidad_aire = presion / (R_aire * temperatura_kelvin)

    return densidad_aire

if __name__ == "__main__":
    # Solicitar la temperatura en grados Celsius
    temperatura_celsius = float(input("Por favor, ingrese la temperatura en grados Celsius: "))

    # Calcular la densidad del aire
    densidad_aire = calcular_densidad_aire(temperatura_celsius)

    # Imprimir la densidad del aire
    print("La densidad del aire a {:.2f} grados Celsius es {:.4f} kg/m^3.".format(temperatura_celsius, densidad_aire))
