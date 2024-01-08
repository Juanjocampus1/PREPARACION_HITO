import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def kelvin_a_celsius_modelo():
    kelvin_temps = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=float)
    celsius_temps = np.array([-273.15, -263.15, -253.15, -243.15, -233.15, -223.15, -213.15, -203.15, -193.15, -183.15, -173.15], dtype=float)

    kelvin_temps_norm = kelvin_temps / 400.0
    celsius_temps_norm = celsius_temps / 400.0

    capa = Dense(units=1, input_shape=[1])
    modelo = Sequential([capa])
    modelo.compile(optimizer=Adam(0.1), loss='mean_squared_error')
    historial = modelo.fit(kelvin_temps_norm, celsius_temps_norm, epochs=500, verbose=False)
    print("Modelo entrenado")

    return modelo

def kelvin_a_celsius(modelo, temp_k):
    temp_k_norm = temp_k / 400.0
    temp_c_norm = modelo.predict([temp_k_norm])[0][0]

    temp_c = temp_c_norm * 400.0
    return temp_c

def main():
    modelo_entrenado = kelvin_a_celsius_modelo()
    temp_k = float(input("Introduce la temperatura en Kelvin: "))
    temp_c = kelvin_a_celsius(modelo_entrenado, temp_k)
    print(f"{temp_k} Kelvin es igual a {temp_c} Celsius.")