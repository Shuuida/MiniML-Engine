"""
Test Suite de Integración para MiniML
Valida Pipeline completo, predicciones, y utilidades de runtime.
"""
import json
import traceback
from miniml import ml_manager, ml_runtime
from typing import List, Any

# -------------------------------------
# CONFIGURACIÓN DE DATASETS
# -------------------------------------
CLASSIFICATION_DATA = [
    [2.78, 2.55, 0], [1.46, 2.36, 0], [3.39, 4.40, 0], 
    [7.62, 2.75, 1], [5.33, 2.08, 1], [6.92, 1.77, 1]
]

REGRESSION_DATA = [
    [1.0, 3.1], [2.0, 5.2], [3.0, 6.8], [4.0, 9.1], [5.0, 11.2]
]
X_reg_test = [[6.0], [7.0]]
Y_reg_test = [13.0, 15.0]

# -------------------------------------
# TEST 1: Clasificación (Decision Tree)
# -------------------------------------
def test_classification_pipeline():
    print("\n--- 🧠 TEST 1: Clasificación (Decision Tree) ---")
    try:
        # 1. Entrenamiento
        result = ml_manager.train_pipeline(
            model_type="DecisionTreeClassifier",
            dataset=CLASSIFICATION_DATA,
            model_name="dt_test",
            params={"max_depth": 3}
        )

        print(f"[INFO] Modo de entrenamiento: {ml_manager.available_mode()}")
        print(f"✅ Modelo 'dt_test' entrenado.")

        # 2. Predicción en nuevos datos (Uso de nueva API predict)
        X_new = [[5.0, 2.0], [1.5, 3.0]]
        new_preds = ml_manager.predict("dt_test", X_new)
        print(f"✅ Predicciones en nuevos datos {X_new}: {new_preds}")

        # 3. Evaluación
        y_true = [row[-1] for row in CLASSIFICATION_DATA]
        X_eval = [row[:-1] for row in CLASSIFICATION_DATA]
        y_pred = ml_manager.predict("dt_test", X_eval)
        
        metrics = ml_manager.evaluate_ext(
            y_true=y_true, y_pred=y_pred, metrics=["accuracy", "recall"], detailed=True
        )
        print(f"✅ Evaluación (Acc: {metrics.get('accuracy', 'N/A'):.4f})")
        
        # 4. Exportación
        c_code_snippet = ml_manager.export_to_c("dt_test")
        if "predict_model" in c_code_snippet:
            print("✅ Exportación a C para 'dt_test' validada.")
        else:
             print("⚠️ Advertencia en exportación a C.")

    except Exception as e:
        print(f"❌ ERROR en TEST 1: {e}")
        traceback.print_exc()

# -------------------------------------
# TEST 2: Regresión (MiniLinearModel)
# -------------------------------------
def test_regression_pipeline():
    print("\n--- 📈 TEST 2: Regresión (MiniLinearModel) ---")
    try:
        # 1. Entrenamiento
        ml_manager.train_pipeline(
            model_type="MiniLinearModel",
            dataset=REGRESSION_DATA,
            model_name="lm_test",
            params={"learning_rate": 0.01, "epochs": 100}
        )
        print(f"✅ Modelo 'lm_test' entrenado.")

        # 2. Predicción
        # La nueva API permite predict("nombre", X) sin keyword, pero 'X=' es más seguro
        y_pred_new = ml_manager.predict("lm_test", X=X_reg_test)
        print(f"✅ Predicciones en {X_reg_test}: {y_pred_new}")
        
        # 3. Evaluación
        metrics = ml_manager.evaluate_ext(y_true=Y_reg_test, y_pred=y_pred_new, metrics=["mse", "r2"])
        print(f"✅ Evaluación (MSE: {metrics.get('mse', 'N/A'):.4f})")
        
    except Exception as e:
        print(f"❌ ERROR en TEST 2: {e}")
        traceback.print_exc()

# -------------------------------------
# TEST 3: Validación de Utilidad ml_runtime (MiniScaler)
# -------------------------------------
def test_ml_runtime_utility():
    print("\n--- ⚙️ TEST 3: Utilidad de ml_runtime (MiniScaler) ---")
    try:
        # Datos: 3 filas, 2 columnas
        raw_data = [[10, 100], [0, 0], [20, 200]]
        
        scaler = ml_runtime.MiniScaler()
        scaler.fit(raw_data)
        
        print(f"✅ Datos originales: {raw_data}")
        
        # MiniScaler.transform está diseñado para vectores (1 fila), 
        # simulando el comportamiento en C/Arduino.
        scaled_data = [scaler.transform(row) for row in raw_data]
        
        print(f"✅ Datos escalados: {scaled_data}")
        
        assert len(scaled_data) == 3, "❌ El escalado falló en la dimensión."
        print("✅ Validación de escalado de datos completada.")

    except Exception as e:
        print(f"❌ ERROR en TEST 3: {e}")
        traceback.print_exc()

# -------------------------------------
# EJECUTOR PRINCIPAL
# -------------------------------------
def run_full_simulation():
    ml_manager.clear_registry() 
    test_classification_pipeline()
    test_regression_pipeline()
    test_ml_runtime_utility()
    print("\n==============================================================")
    print("🎯 SIMULACIÓN COMPLETA FINALIZADA.")
    print("==============================================================")
    
if __name__ == "__main__":
    run_full_simulation()