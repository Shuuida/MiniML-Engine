"""
===================================================
Test de QA: Random Forest (MiniML Legacy)
Evalúa: Generación de Árboles en PROGMEM y Votación Mayoritaria.
"""

import os
from miniml.ml_runtime import RandomForestClassifier

print("1. Creando Dataset Tabular IoT...")
# Columnas: [Temperatura, Humedad, Vibración, CLASE]
# Clase 0 = Normal, Clase 1 = Falla de Máquina
dataset = [
    [22.0, 45.0, 0.1, 0],
    [23.0, 47.0, 0.2, 0],
    [21.5, 44.0, 0.1, 0],
    [45.0, 80.0, 1.5, 1],
    [48.0, 85.0, 1.7, 1],
    [50.0, 90.0, 2.0, 1]
]

print("2. Entrenando Random Forest...")
# 3 árboles son suficientes para probar la votación C++
rf = RandomForestClassifier(n_trees=3, max_depth=3, seed=42)
rf.fit(dataset)

print("\n3. Evaluando IA (Python):")
X_test = [
    [22.5, 46.0, 0.15], # Perfil Normal (Esperado: 0)
    [47.0, 82.0, 1.6]   # Perfil de Falla (Esperado: 1)
]

preds = rf.predict(X_test)
for i, (x, p) in enumerate(zip(X_test, preds)):
    print(f"  Muestra {i+1} -> Predicción: {p}")

print("\n4. Exportando Firmware C++...")
os.makedirs("exports", exist_ok=True)
cpp_code = rf.to_arduino_code(fn_name="predict_rf")

# Envolvemos el código generado en guardas de cabecera estándar
header_content = f"""#ifndef MINIML_RF_H
#define MINIML_RF_H

#include <stdint.h>

{cpp_code}

#endif
"""

with open("exports/miniml_rf.h", "w") as f:
    f.write(header_content)
print("  [OK] Exportación exitosa. Listo para Wokwi.")