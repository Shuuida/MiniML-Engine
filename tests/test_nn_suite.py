"""
test_nn_suite.py
----------------
Test Suite de Integración Robusta para MiniML.
Diseñado para validar: ml_manager, ml_exporter, ml_runtime (NN Quantization) y adapter.
"""
import sys
import os

current_path = os.path.dirname(os.path.abspath(__file__))
if current_path not in sys.path:
    sys.path.insert(0, current_path)

print(f"[INIT] Directorio de ejecución: {current_path}")

try:
    from miniml import ml_runtime
    from miniml import ml_factory 
    from miniml import ml_manager
    from miniml import ml_exporter
    print("[INIT] Módulos importados correctamente.")
except ImportError as e:
    print("\n" + "!"*60)
    print(f"[CRITICAL ERROR] Faltan archivos: {e}")
    print("Asegúrate de tener: ml_manager.py, ml_runtime.py, ml_exporter.py, ml_factory.py y adapter.py")
    print("en la misma carpeta.")
    print("!"*60 + "\n")
    sys.exit(1)

# Configuración
MODEL_NAME = "nn_test_integration"
JSON_FILE = "test_nn_model.json"

# Dataset XOR Clásico
DATASET_XOR = [
    [0.0, 0.0, 0],
    [0.0, 1.0, 1],
    [1.0, 0.0, 1],
    [1.0, 1.0, 0]
]
X_TEST = [[0.0, 1.0], [1.0, 1.0]]

def run_integration_test():
    print("\n" + "="*50)
    print("   INICIANDO TEST DE INTEGRACIÓN: NN & QUANTIZATION")
    print("="*50)

    # LIMPIEZA
    ml_manager.clear_registry()
    if os.path.exists(JSON_FILE): os.remove(JSON_FILE)

    # ENTRENAMIENTO
    print("\n[PASO 1] Entrenando Neural Network (Pipeline)...")
    try:
        # Usamos parámetros altos para intentar convergencia
        params = {
            "n_inputs": 2, "n_hidden": 4, "n_outputs": 1,
            "epochs": 2000, "learning_rate": 0.1, "seed": 42
        }
        res = ml_manager.train_pipeline(
            model_name=MODEL_NAME,
            dataset=DATASET_XOR,
            model_type="neural_network",
            params=params,
            scaling="minmax" # Probamos integración con scaler
        )
        model = res['model']
        print(f" -> Entrenamiento OK. Metadata: {res['meta']}")
        
        # Validación inmediata de predicción
        preds = ml_manager.predict(MODEL_NAME, X_TEST)
        print(f" -> Predicciones preliminares: {[round(p[0], 2) for p in preds]}")
        
        # Validar que act_scales existe
        if hasattr(model, 'act_scales') and model.act_scales:
            print(f" -> [OK] act_scales detectado: {model.act_scales}")
        else:
            print(" -> [WARNING] act_scales vacío o no existente tras entrenamiento.")

    except Exception as e:
        print(f"[FAIL] Error en entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return

    # SERIALIZACIÓN & GUARDADO
    print("\n[PASO 2] Guardando modelo en disco...")
    try:
        ml_manager.save_model(MODEL_NAME, JSON_FILE)
        if os.path.exists(JSON_FILE):
            print(f" -> Archivo {JSON_FILE} creado correctamente.")
        else:
            raise FileNotFoundError("El archivo JSON no aparece.")
    except Exception as e:
        print(f"[FAIL] Error guardando modelo: {e}")
        return

    # CARGA & RESTAURACIÓN
    print("\n[PASO 3] Limpiando memoria y recargando...")
    try:
        ml_manager.clear_registry()
        ml_manager.load_model(MODEL_NAME, JSON_FILE)
        
        loaded_model = ml_manager.get_model(MODEL_NAME)
        
        # Validar tipo
        if isinstance(loaded_model, ml_runtime.MiniNeuralNetwork):
            print(" -> [OK] El modelo cargado es MiniNeuralNetwork.")
        else:
            print(f" -> [FAIL] Tipo incorrecto: {type(loaded_model)}")
            
        # Validar Scaler restaurado
        if hasattr(loaded_model, 'scaler') and loaded_model.scaler:
            print(" -> [OK] Scaler restaurado correctamente.")
        else:
            print(" -> [FAIL] Scaler perdido en la carga.")

        # Validar persistencia de pesos y escalas
        if hasattr(loaded_model, 'W1') and len(loaded_model.W1) > 0:
            print(" -> [OK] Matriz W1 presente.")
        
        # Validar act_scales recuperado
        if hasattr(loaded_model, 'act_scales') and 'input' in loaded_model.act_scales:
             print(f" -> [OK] act_scales persistió: {loaded_model.act_scales}")
        else:
             print(" -> [FAIL] act_scales se perdió en save/load.")

    except Exception as e:
        print(f"[FAIL] Error cargando modelo: {e}")
        return

    # EXPORTACIÓN A C
    print("\n[PASO 4] Exportando a C (Probando to_arduino_code)...")
    try:
        # Esto disparará model.quantize() internamente si no está cuantizado
        c_code = ml_manager.export_to_c(MODEL_NAME)
        
        print(f" -> Código generado ({len(c_code)} bytes).")
        
        # ANÁLISIS DEL CÓDIGO GENERADO
        errors_c = []
        
        # 1. Verificar PROGMEM
        if "PROGMEM" not in c_code:
            errors_c.append("No se encontró la directiva PROGMEM.")
            
        # 2. Verificar arrays int8 (tu implementación usa int8_t para pesos)
        if "int8_t" not in c_code:
            errors_c.append("No se encontraron arrays int8_t (¿Falló la cuantización?).")
            
        # 3. Verificar las escalas float
        if "const float" not in c_code:
            errors_c.append("No se encontraron arrays de escalas float.")
            
        # 4. Verificar la función de inferencia
        if "void predict(" not in c_code and "model_predict_core" not in c_code:
            errors_c.append("Falta la función principal de predicción.")

        if not errors_c:
            print(" -> [OK] El código C parece válido y usa PROGMEM/int8.")
            
            # Guardar artifact
            with open("test_output.h", "w") as f:
                f.write(c_code)
            print(" -> Código guardado en 'test_output.h' para inspección visual.")
        else:
            print(" -> [FAIL] Problemas en el código C:")
            for err in errors_c:
                print(f"    - {err}")

    except Exception as e:
        print(f"[FAIL] Error fatal en exportación: {e}")
        import traceback
        traceback.print_exc()

    # ESTRUCTURA
    print("\n[PASO 5] Extracción de Estructura (Visualización)...")
    try:
        struct = ml_exporter.extract_model_structure(loaded_model)
        if struct['type'] == 'NeuralNetwork':
            print(" -> [OK] Tipo de estructura correcto.")
        if 'W1' in struct and 'calibration' in struct:
             print(" -> [OK] Bloques de pesos y calibración presentes.")
        else:
             print(" -> [FAIL] Faltan datos en la estructura.")
    except Exception as e:
        print(f"[FAIL] Error en extracción: {e}")

    # CLEANUP
    if os.path.exists(JSON_FILE): os.remove(JSON_FILE)
    print("\n=== TEST FINALIZADO ===")

if __name__ == "__main__":
    run_integration_test()