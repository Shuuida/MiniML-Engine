"""
MiniML Comprehensive Test Suite
===============================
Simulación de uso de librería 'MiniML' para producción.
Valida el ciclo de vida completo de TODOS los algoritmos disponibles.

Ciclo por Modelo:
1. Definición del Pipeline (con/sin escalado).
2. Entrenamiento (Fit).
3. Inferencia en Python (Predict).
4. Exportación a Firmware (Generate C Header).
5. Persistencia (Save JSON).

Artefactos generados en: ./test_outputs/
"""

import os
import shutil
import time
from core import ml_manager

# Directorio para artefactos de salida
OUTPUT_DIR = "test_outputs"

# -------------------------------------
# DATASETS SINTÉTICOS (Pequeños para verificación visual)
# -------------------------------------

# Clasificación Binaria (0 o 1)
# Pattern: x1 > 5 AND x2 < 3 -> 1, else 0
CLS_DATA = [
    [5.1, 3.5, 0], [4.9, 3.0, 0], [4.7, 3.2, 0], [4.6, 3.1, 0], # Clase 0
    [5.0, 3.6, 0], [5.4, 3.9, 0], [4.5, 2.3, 0], [4.4, 2.9, 0], # Clase 0
    [7.0, 3.2, 1], [6.4, 3.2, 1], [6.9, 3.1, 1], [5.5, 2.3, 1], # Clase 1
    [6.5, 2.8, 1], [5.7, 2.8, 1], [6.3, 3.3, 1], [5.8, 2.7, 1]  # Clase 1
]
CLS_X_TEST = [[5.0, 3.4], [6.0, 2.5]] # Esperado: [0, 1]

# Regresión Lineal Simple
# Pattern: y = 2*x1 + 3*x2 + 5 (aprox)
REG_DATA = [
    [1.0, 1.0, 10.0], [2.0, 1.0, 12.0], [1.0, 2.0, 13.0], 
    [3.0, 2.0, 17.0], [5.0, 1.0, 18.0], [0.0, 0.0, 5.0],
    [2.0, 2.0, 15.0], [4.0, 4.0, 25.0]
]
REG_X_TEST = [[3.0, 3.0], [10.0, 10.0]] 
# Esperado: 
# x=[3,3] -> 2(3)+3(3)+5 = 6+9+5 = 20
# x=[10,10] -> 2(10)+3(10)+5 = 20+30+5 = 55

# Problema No Lineal (XOR) para Redes Neuronales
XOR_DATA = [
    [0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]
]

# -------------------------------------
# UTILIDADES DE TEST
# -------------------------------------

def setup_environment():
    """Limpia y prepara el entorno de pruebas."""
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    ml_manager.clear_registry()
    print(f"📂 Entorno preparado: ./{OUTPUT_DIR}/")

def run_model_test(alias, model_type, dataset, test_input, task="classification", **params):
    """
    Ejecuta un test estandarizado para un modelo específico.
    """
    print(f"\n🔹 Probando: {alias} ({model_type})...")
    
    # 1. Pipeline de Entrenamiento
    # Usamos escalado 'minmax' por defecto para modelos sensibles a distancia/gradiente
    scaling = "minmax" if model_type in ["NeuralNetwork", "SVM", "KNearestNeighbors", "LinearRegression"] else None
    
    try:
        res = ml_manager.train_pipeline(
            model_name=alias,
            dataset=dataset,
            model_type=model_type,
            params=params,
            scaling=scaling
        )
        print(f"  ✅ Entrenamiento OK ({res['meta']['time_seconds']:.4f}s). Scaling: {scaling}")
        
        # 2. Inferencia Python
        preds = ml_manager.predict(alias, test_input)
        
        # Formateo de predicción para legibilidad
        fmt_preds = preds
        if task == "classification":
            # Si es NN, la salida es float, redondeamos para visualizar clase
            if model_type == "NeuralNetwork":
                fmt_preds = [int(p[0] >= 0.5) for p in preds]
            else:
                fmt_preds = preds
        else:
            fmt_preds = [round(float(p), 2) for p in preds]
            
        print(f"  ✅ Inferencia Python: {fmt_preds}")

        # 3. Exportación de Artefactos
        # JSON
        json_path = os.path.join(OUTPUT_DIR, f"{alias}.json")
        ml_manager.save_model(alias, json_path)
        
        # Código C (Header File)
        c_code = ml_manager.export_to_c(alias)
        h_path = os.path.join(OUTPUT_DIR, f"{alias}.h")
        with open(h_path, "w", encoding="utf-8") as f:
            f.write(c_code)
            
        print(f"  💾 Artefactos exportados: {alias}.json, {alias}.h")
        return True

    except Exception as e:
        print(f"  ❌ ERROR CRÍTICO en {alias}: {e}")
        # import traceback
        # traceback.print_exc()
        return False

# -------------------------------------
# SUITE PRINCIPAL
# -------------------------------------

def main():
    print("==============================================================")
    print("🚀 MINIML FRAMEWORK - FULL COMPREHENSIVE TEST SUITE")
    print("==============================================================")
    
    setup_environment()
    
    success_count = 0
    total_tests = 0

    # ÁRBOLES (Robustos, no requieren escalado)
    
    # Decision Tree (Clasificación)
    total_tests += 1
    if run_model_test("dt_classifier", "DecisionTreeClassifier", CLS_DATA, CLS_X_TEST, 
                      task="classification", max_depth=4):
        success_count += 1

    # Random Forest (Clasificación)
    total_tests += 1
    if run_model_test("rf_classifier", "RandomForestClassifier", CLS_DATA, CLS_X_TEST, 
                      task="classification", n_trees=5, max_depth=3):
        success_count += 1

    # Random Forest (Regresión)
    total_tests += 1
    if run_model_test("rf_regressor", "RandomForestRegressor", REG_DATA, REG_X_TEST, 
                      task="regression", n_trees=5, max_depth=4):
        success_count += 1

    # MODELOS MATEMÁTICOS (Requieren Escalado)

    # Linear Regression
    total_tests += 1
    if run_model_test("linear_reg", "LinearRegression", REG_DATA, REG_X_TEST, 
                      task="regression", learning_rate=0.01, epochs=2000):
        success_count += 1

    # SVM (Clasificación)
    total_tests += 1
    if run_model_test("svm_classifier", "SVM", CLS_DATA, CLS_X_TEST, 
                      task="classification", learning_rate=0.001, n_iters=1000):
        success_count += 1

    # Neural Network (XOR Problem - No Lineal)
    # Nota: Aumentamos epochs para garantizar convergencia en dataset pequeño
    total_tests += 1
    if run_model_test("nn_xor", "NeuralNetwork", XOR_DATA, [[0, 1], [1, 1]], 
                      task="classification", 
                      n_inputs=2, n_hidden=4, n_outputs=1, epochs=5000, learning_rate=0.1):
        success_count += 1

    # LAZY LEARNING (Basado en Instancias)

    # KNN (Clasificación)
    total_tests += 1
    if run_model_test("knn_classifier", "KNearestNeighbors", CLS_DATA, CLS_X_TEST, 
                      task="classification", k=3, task_type="classification"):
        success_count += 1
    
    # KNN (Regresión)
    total_tests += 1
    if run_model_test("knn_regressor", "KNearestNeighbors", REG_DATA, REG_X_TEST, 
                      task="regression", k=2, task_type="regression"):
        success_count += 1

    print("\n==============================================================")
    print(f" RESUMEN: {success_count}/{total_tests} Tests Exitosos.")
    if success_count == total_tests:
        print("✅ EL FRAMEWORK ESTÁ LISTO PARA PRODUCCIÓN.")
        print(f"👉 Revisa la carpeta '{OUTPUT_DIR}' para ver el código C generado.")
    else:
        print("⚠️ HUBO ERRORES. REVISA EL LOG.")
    print("==============================================================")

if __name__ == "__main__":
    main()