"""
===================================================
Test de QA Final: MLP Quantizado y MiniScaler
Evalúa: Cuantificación INT8 Híbrida y Preprocesamiento in-place.
"""

import os
from miniml.ml_runtime import MiniNeuralNetwork, MiniScaler

print("1. [MLP - Sin Escalar] Entrenando red para XOR estándar...")
dataset_unscaled = [
    [0.0, 0.0, 0],
    [0.0, 1.0, 1],
    [1.0, 0.0, 1],
    [1.0, 1.0, 0]
]
nn_unscaled = MiniNeuralNetwork(n_inputs=2, n_hidden=4, n_outputs=1, epochs=2000, learning_rate=0.1, seed=42)
nn_unscaled.fit(dataset_unscaled)

print("2. [MLP - Con Escalar] Entrenando red con datos de gran magnitud...")
# Simulamos lecturas de sensores que van de 0 a 100
dataset_raw = [
    [0.0, 0.0, 0],
    [0.0, 100.0, 1],
    [100.0, 0.0, 1],
    [100.0, 100.0, 0]
]

# Separar para ajustar el escalador
X_raw = [row[:-1] for row in dataset_raw]
y_raw = [row[-1] for row in dataset_raw]

scaler = MiniScaler(method='minmax', feature_range=(0, 1))
scaler.fit(X_raw)

# Aplicar transformación al dataset de entrenamiento
dataset_scaled = []
for x, y in zip(X_raw, y_raw):
    dataset_scaled.append(scaler.transform(x) + [y])

nn_scaled = MiniNeuralNetwork(n_inputs=2, n_hidden=4, n_outputs=1, epochs=2000, learning_rate=0.1, seed=42)
nn_scaled.fit(dataset_scaled)

print("\n3. Evaluando IA en PC (Python):")
# Prueba Unscaled
test_unscaled = [[0.0, 1.0], [1.0, 1.0]]
preds_uns = nn_unscaled.predict(test_unscaled)
print(f"  [Sin Escalar] Muestra [0.0, 1.0] -> Predicción: {preds_uns[0][0]:.4f} (Esperado: ~1.0)")
print(f"  [Sin Escalar] Muestra [1.0, 1.0] -> Predicción: {preds_uns[1][0]:.4f} (Esperado: ~0.0)")

# Prueba Scaled
test_raw_inputs = [[0.0, 100.0], [100.0, 100.0]]
test_raw_scaled = [scaler.transform(row) for row in test_raw_inputs]
preds_scl = nn_scaled.predict(test_raw_scaled)
print(f"  [Con Escalar] Muestra [0.0, 100.0] -> Predicción: {preds_scl[0][0]:.4f} (Esperado: ~1.0)")
print(f"  [Con Escalar] Muestra [100.0, 100.0] -> Predicción: {preds_scl[1][0]:.4f} (Esperado: ~0.0)")

print("\n4. Cuantizando Modelos (INT8/Híbrido) y Exportando a C++...")
os.makedirs("exports", exist_ok=True)
# La cuantización requiere haber corrido un fit() o calibrate() previo
nn_unscaled.quantize()
nn_scaled.quantize()

cpp_nn_unscaled = nn_unscaled.to_arduino_code(fn_name="predict_nn_unscaled")
cpp_scaler = scaler.to_arduino_code(fn_name="preprocess_minmax")
cpp_nn_scaled = nn_scaled.to_arduino_code(fn_name="predict_nn_scaled")

header_content = f"""#ifndef MINIML_MLP_TEST_H
#define MINIML_MLP_TEST_H

#include <stdint.h>
#include <math.h>

{cpp_nn_unscaled}

{cpp_scaler}

{cpp_nn_scaled}

#endif
"""

with open("exports/miniml_mlp.h", "w") as f:
    f.write(header_content)
print("  [OK] Exportación exitosa. Listo para Wokwi.")