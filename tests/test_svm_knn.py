"""
===================================================
Test de QA Dual: MiniSVM y K-Nearest Neighbors
Evalúa: Hiperplanos lineales y Búsqueda Espacial in-place en SRAM.
"""

import os
from miniml.ml_runtime import MiniSVM, KNearestNeighbors

print("1. [SVM] Entrenando Support Vector Machine...")
# Dataset Linealmente Separable (Clases: 1 y -1)
dataset_svm = [
    [2.0, 2.0, 1],
    [3.0, 3.0, 1],
    [-2.0, -2.0, -1],
    [-3.0, -3.0, -1]
]
svm = MiniSVM(learning_rate=0.01, n_iters=1000)
svm.fit(dataset_svm)

print("2. [KNN] Entrenando K-Nearest Neighbors...")
# Dataset Multiclase Agrupado (Clases: 0, 1, 2)
dataset_knn = [
    [0.0, 0.0, 0], [0.1, 0.1, 0], [0.0, 0.1, 0], # Cluster 0
    [5.0, 5.0, 1], [5.1, 5.1, 1], [5.0, 5.1, 1], # Cluster 1
    [9.0, 9.0, 2], [9.1, 9.1, 2], [9.0, 9.1, 2]  # Cluster 2
]
knn = KNearestNeighbors(k=3, task='classification')
knn.fit(dataset_knn)

print("\n3. Evaluando IA en PC (Python):")
# Pruebas para SVM
test_svm = [[2.5, 2.5], [-2.5, -2.5]]
preds_svm = svm.predict(test_svm)
print(f"  [SVM] Muestra [ 2.5,  2.5] -> Predicción: {preds_svm[0]} (Esperado: 1)")
print(f"  [SVM] Muestra [-2.5, -2.5] -> Predicción: {preds_svm[1]} (Esperado: -1)")

# Pruebas para KNN
test_knn = [[0.2, 0.2], [5.2, 5.2], [8.9, 8.9]]
preds_knn = knn.predict(test_knn)
print(f"  [KNN] Muestra [0.2, 0.2] -> Predicción: {preds_knn[0]} (Esperado: 0)")
print(f"  [KNN] Muestra [5.2, 5.2] -> Predicción: {preds_knn[1]} (Esperado: 1)")
print(f"  [KNN] Muestra [8.9, 8.9] -> Predicción: {preds_knn[2]} (Esperado: 2)")

print("\n4. Exportando Firmware C++ Unificado...")
os.makedirs("exports", exist_ok=True)
cpp_svm = svm.to_arduino_code(fn_name="predict_svm")
cpp_knn = knn.to_arduino_code(fn_name="predict_knn")

header_content = f"""#ifndef MINIML_LEGACY_TESTS_H
#define MINIML_LEGACY_TESTS_H

#include <stdint.h>
#include <math.h>

{cpp_svm}

{cpp_knn}

#endif
"""

with open("exports/miniml_legacy.h", "w") as f:
    f.write(header_content)
print("  [OK] Exportación exitosa. Listo para Wokwi.")