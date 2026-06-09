"""
===================================================
Test de QA: Transfer Learning Edge (SGD Estático) para capas Linear
Evalúa la integridad de SGD Estático generado en C++ para ESP32 (On-Device Learning en el borde)
"""

from miniml.autograd.tensor import Tensor
from miniml.autograd.layers import Sequential, Linear, ReLU
from miniml.exporters.cpp_writer import generate_cpp_code

print("--- MiniML Edge: Test de Exportador Deep Transfer Learning ---")

# 1. Construir una topología profunda (Multilayer Perceptron)
# Entrada: 2 características -> Salida: 1 predicción
model = Sequential([
    Linear(2, 8),   # layer_0: Extractor de bajo nivel
    ReLU(),         # layer_1
    Linear(8, 4),   # layer_2: Extractor de alto nivel
    ReLU(),         # layer_3
    Linear(4, 1)    # layer_4: Clasificador/Regresor Final
])

# 2. Configuración del "Layer Freezing" (El núcleo del Transfer Learning)
print("Configurando Semáforos de Memoria...")

# Primero, por seguridad, congelamos TODAS las capas del modelo.
# Esto asegura que todo vaya a la memoria PROGMEM (Flash) por defecto.
for layer in model.layers:
    layer.freeze()

# Luego, descongelamos únicamente la última capa Linear (layer_4).
# El exportador detectará esto y la enviará a la SRAM con el motor SGD.
model.layers[-1].unfreeze()

print(f"Capa Oculta 1 (Linear 2->8): Trainable = {model.layers[0].trainable} (Flash)")
print(f"Capa Oculta 2 (Linear 8->4): Trainable = {model.layers[2].trainable} (Flash)")
print(f"Capa Final    (Linear 4->1): Trainable = {model.layers[-1].trainable} (SRAM)")

# 3. Exportar a C++
try:
    print("\nGenerando librería C++...")
    cpp_code = generate_cpp_code(
        model=model,
        input_shape=(1, 2), # Un batch, dos características de entrada
        model_name="MiniMLModel",
        on_device_learning=True,  # Activamos el motor SGD
        loss_type='MSE',          # Usamos Error Cuadrático Medio
        learning_rate=0.01
    )

    # Guardar el archivo
    with open("exports/miniml_model.h", "w") as f:
        f.write(cpp_code)
    print("  [OK] C++ Exportado exitosamente.")
    print("Llévalo a Wokwi y ejecuta el testbench.")
except:
    print(f"  [ERROR C++] El exportador falló al procesar el archivo.")