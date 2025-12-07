"""
ml_manager.py
-------------
Gestor unificado para el ciclo de vida de modelos MiniML.
Implementa un pipeline "Línea de Montaje": Imputación -> Escalado -> Entrenamiento.

API Pública Principal:
- train_pipeline(): Único punto de entrada para entrenar cualquier modelo.
- predict(): Realiza inferencias (acepta nombre o instancia).
- evaluate_ext(): Calcula métricas de rendimiento.
- save_model() / load_model(): Persistencia.
- export_to_c(): Generación de código.
"""

from typing import Any, Dict, List, Optional, Union
import json
import os
import time
import tempfile

# Módulos internos del framework
try:
    from . import ml_runtime
    from . import ml_factory
    from .ml_compat import impute_missing_values
except ImportError:
    # Fallback para ejecución local sin paquete
    import ml_runtime
    import ml_factory
    from ml_compat import impute_missing_values

# INTENTO DE IMPORTACIÓN DE ADAPTER (Seguro)
_CMSIS_AVAILABLE = False
try:
    from adapters.cmsis_nn.adapter import CMSISAdapter
    _CMSIS_AVAILABLE = True
except ImportError:
    # No pasa nada, simplemente no usaremos CMSIS
    pass

# Registro en memoria para modelos
_MODEL_REGISTRY: Dict[str, Any] = {}

# Intenta importar sklearn y joblib opcionalmente.
_sklearn_available = False
try:
    import sklearn
    _sklearn_available = True
except ImportError:
    pass

# -----------------------------------------------------------------------------
# GESTIÓN DEL REGISTRO Y UTILIDADES BÁSICAS
# -----------------------------------------------------------------------------

def available_mode() -> str:
    """Devuelve el modo de ML actualmente disponible ('sklearn' o 'mini')."""
    return "sklearn" if _sklearn_available else "mini"

def list_models() -> List[str]:
    """Lista todos los modelos actualmente registrados en memoria."""
    return list(_MODEL_REGISTRY.keys())

def get_model(name: str) -> Any:
    """Obtiene un modelo registrado por nombre."""
    return _MODEL_REGISTRY.get(name, {}).get("model")

def clear_registry():
    """Limpia el registro de modelos en memoria."""
    _MODEL_REGISTRY.clear()

def _is_regression_dataset(dataset):
    """Heurística para determinar si es regresión o clasificación."""
    if not dataset or not isinstance(dataset, list):
        return False
    try:
        y = [row[-1] for row in dataset]
        if any(isinstance(v, float) and not v.is_integer() for v in y):
            return True
        return len(set(y)) > 10
    except Exception:
        return False

# -----------------------------------------------------------------------------
# PIPELINE DE ENTRENAMIENTO UNIFICADO
# -----------------------------------------------------------------------------

def train_pipeline(model_name: str, dataset: List[List[Any]], model_type: str, 
                   params: Optional[Dict[str, Any]] = None, scaling: str = None) -> Dict[str, Any]:
    
    if not dataset:
        raise ValueError("El dataset de entrada está vacío.")

    print(f"--- Iniciando Pipeline para '{model_name}' ({model_type}) ---")
    start_time = time.time()
    
    # Imputación (Limpieza)
    print(f" Ejecutando imputación de datos...")
    clean_dataset = impute_missing_values(dataset, strategy='mean')

    # Escalado (Preprocesamiento)
    scaler = None
    if scaling:
        print(f" Aplicando escalado ({scaling})...")
        X_raw = [row[:-1] for row in clean_dataset]
        y_raw = [row[-1] for row in clean_dataset]
        
        scaler = ml_runtime.MiniScaler(method=scaling)
        scaler.fit(X_raw)
        
        X_scaled = [scaler.transform(row) for row in X_raw]
        processed_dataset = [x + [y] for x, y in zip(X_scaled, y_raw)]
    else:
        print(" Escalado omitido.")
        processed_dataset = clean_dataset

    # Construcción y Entrenamiento
    print(f" Entrenando modelo {model_type}...")
    try:
        # Usamos Factory para instanciar el modelo correcto
        model = ml_factory.create_model(model_type, params)
        model.fit(processed_dataset)
        
    except Exception as e:
        raise RuntimeError(f"Error crítico durante el entrenamiento: {e}")

    # Ensamblaje Final
    if scaler:
        model.scaler = scaler
        if not hasattr(model, 'metadata'): model.metadata = {}
        model.metadata['scaling_method'] = scaling

    # Registro
    is_regression = _is_regression_dataset(processed_dataset)
    meta = {
        'mode': 'mini',
        'type': model_type,
        'task': 'regression' if is_regression else 'classification',
        'scaling': scaling,
        'params': params,
        'time_seconds': time.time() - start_time
    }
    
    _MODEL_REGISTRY[model_name] = {
        'mode': 'mini',
        'model': model,
        'meta': meta
    }

    print(f"--- Pipeline finalizado exitosamente en {meta['time_seconds']:.4f}s ---")
    return {'model': model, 'meta': meta}


# -----------------------------------------------------------------------------
# INFERENCIA (PREDICCIÓN) INTELIGENTE - REFACTORIZADO
# -----------------------------------------------------------------------------

def predict(name_or_model: Union[str, Any], X: List[List[Any]]) -> List[Any]:
    """
    Realiza predicciones utilizando un modelo entrenado.
    AUTOMÁTICAMENTE aplica escalado si el modelo fue entrenado con uno.

    Args:
        name_or_model: Puede ser el string del nombre registrado O el objeto modelo.
        X (list): Datos de entrada (features). Lista de listas.

    Returns:
        List: Predicciones.
    """
    if X is None:
        raise ValueError("predict requiere datos de entrada 'X'.")

    # Resolución del modelo (Polimorfismo de argumentos)
    model_obj = None
    if isinstance(name_or_model, str):
        if name_or_model not in _MODEL_REGISTRY:
            raise KeyError(f"Modelo '{name_or_model}' no encontrado en el registro.")
        model_obj = _MODEL_REGISTRY[name_or_model]['model']
    else:
        model_obj = name_or_model

    if model_obj is None:
        raise ValueError("No se pudo resolver un modelo válido para predecir.")

    # Preprocesamiento Automático (Escalado)
    if hasattr(model_obj, 'scaler') and model_obj.scaler is not None:
        # MiniScaler.transform espera una fila (vector), iteramos:
        X_proc = [model_obj.scaler.transform(row) for row in X]
    else:
        X_proc = X

    # Inferencia
    try:
        preds = model_obj.predict(X_proc)
        return preds
    except AttributeError:
        raise RuntimeError(f"El objeto modelo {type(model_obj)} no tiene método 'predict'.")


# -----------------------------------------------------------------------------
# EVALUACIÓN Y MÉTRICAS
# -----------------------------------------------------------------------------

def evaluate_ext(y_true=None, y_pred=None, metrics=None, detailed=False):
    """Calcula métricas de rendimiento (Accuracy, MSE, R2, etc.)."""
    epsilon = 1e-10
    
    if isinstance(y_pred, str): # Soporte legacy para nombres de variables
        try: y_pred = globals()[y_pred]
        except KeyError: pass

    if not isinstance(y_true, list): y_true = list(y_true)
    if not isinstance(y_pred, list): y_pred = list(y_pred)

    if metrics is None: metrics = ["accuracy"]
    if isinstance(metrics, str): metrics = [metrics]
    
    results = {}

    # Clasificación
    if "accuracy" in metrics:
        results["accuracy"] = ml_runtime.accuracy_score(y_true, y_pred)
    
    # Regresión
    if any(m in metrics for m in ["mae", "mse", "r2"]):
        try:
            y_true_f = [float(y) for y in y_true]
            y_pred_f = [float(y) for y in y_pred]
            
            if "mae" in metrics: results["mae"] = ml_runtime.mae(y_true_f, y_pred_f)
            if "mse" in metrics: results["mse"] = ml_runtime.mse(y_true_f, y_pred_f)
            if "r2" in metrics: results["r2"] = ml_runtime.r2_score(y_true_f, y_pred_f)
        except Exception as e:
            print(f"Warning: Error calculando métricas de regresión: {e}")

    if len(metrics) > 1 or detailed:
        return results
    return results.get(metrics[0])


# -----------------------------------------------------------------------------
# PERSISTENCIA Y EXPORTACIÓN
# -----------------------------------------------------------------------------

def save_model(name: str, path: str) -> None:
    """Guarda el modelo y su scaler (si existe) en JSON."""
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Modelo '{name}' no encontrado.")
    
    entry = _MODEL_REGISTRY[name]
    model = entry['model']
    
    serial_data = {
        'meta': entry.get('meta', {}),
        'model_data': {}
    }

    # Serialización Scaler
    if hasattr(model, 'scaler') and model.scaler:
        serial_data['scaler'] = {
            'method': model.scaler.method,
            'params': model.scaler.params,
            'feature_range': model.scaler.feature_range,
            'n_features_trained': model.scaler.n_features_trained
        }

    # Delegación de serialización de estructura al exporter (si está disponible)
    # Para mantener ml_manager limpio, hacemos una serialización básica aquí o llamamos a exporter
    # Por simplicidad y robustez en este archivo, replicamos la lógica básica de guardado de estructura
    
    if hasattr(model, 'root'): # DecisionTree
        from .ml_compat import _flatten_tree_to_arrays
        serial_data['model_data']['type'] = 'tree'
        if model.root:
            serial_data['model_data']['struct'] = _flatten_tree_to_arrays(model.root)
            
    elif hasattr(model, 'trees'): # RandomForest
        from .ml_compat import _flatten_tree_to_arrays
        serial_data['model_data']['type'] = 'forest'
        serial_data['model_data']['trees'] = []
        for t in model.trees:
            if hasattr(t, 'root') and t.root:
                serial_data['model_data']['trees'].append(_flatten_tree_to_arrays(t.root))

    # MiniNeuralNetworks (Verificación por W1 y B1)
    elif hasattr(model, 'W1') and hasattr(model, 'n_hidden'):
        serial_data['model_data']['type'] = 'neural_network'
        serial_data['model_data']['config'] = {
            'n_inputs': model.n_inputs, 'n_hidden': model.n_hidden, 'n_outputs': model.n_outputs,
            'hidden_activation': getattr(model, 'hidden_activation', 'sigmoid'),
            'output_activation': getattr(model, 'output_activation', 'sigmoid')
        }
        # Guardar pesos y escalas
        serial_data['model_data']['W1'] = model.W1
        serial_data['model_data']['W2'] = model.W2
        serial_data['model_data']['B1'] = model.B1
        serial_data['model_data']['B2'] = model.B2
        if hasattr(model, 'act_scales'):
            serial_data['model_data']['act_scales'] = model.act_scales

    elif hasattr(model, 'weights'): 
        serial_data['model_data']['type'] = 'linear_svm'
        serial_data['model_data']['weights'] = model.weights
        if hasattr(model, 'bias'): serial_data['model_data']['bias'] = model.bias # SVM
        # LinearModel guarda el bias dentro de weights usualmente, pero por seguridad:
        if hasattr(model, 'intercept'): serial_data['model_data']['intercept'] = model.intercept

    elif hasattr(model, 'X_train'): # KNN
        serial_data['model_data']['type'] = 'knn'
        serial_data['model_data']['X_train'] = model.X_train
        serial_data['model_data']['y_train'] = model.y_train
        serial_data['model_data']['k'] = model.k
        serial_data['model_data']['task'] = model.task

    # Crear directorio si no existe (solo si hay un directorio en la ruta)
    dir_path = os.path.dirname(path)
    if dir_path:  # Solo crear si hay un directorio (no cadena vacía)
        os.makedirs(dir_path, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serial_data, f, indent=2)
    print(f"Modelo '{name}' guardado en {path}")


def load_model(name: str, path: str) -> None:
    """Carga un modelo MiniML desde JSON."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Archivo '{path}' no encontrado.")
        
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    meta = data.get('meta', {})
    model_data = data.get('model_data', {})
    model_type_str = model_data.get('type') or meta.get('type', 'unknown')
    
    params = meta.get('params') or {}
    
    # Determinar el tipo para el factory (debe coincidir con lo que espera ml_factory)
    # El factory busca "neural" o "network" en el string, así que usamos el tipo de meta o model_data
    factory_type = meta.get('type', model_type_str)
    # Asegurar que el tipo sea compatible con el factory
    if model_type_str == 'neural_network' or 'neural' in factory_type.lower() or 'network' in factory_type.lower():
        factory_type = 'neural_network'  # El factory acepta cualquier string con "neural" o "network"
    elif model_type_str == 'linear_svm':
        factory_type = 'linear_regression'

    try:
        # Si es NN, usamos la config guardada para instanciar correctamente
        if model_type_str == 'neural_network' and 'config' in model_data:
            cfg = model_data['config']
            # Actualizar params con lo guardado, pero filtrar los que no son del constructor
            # El constructor acepta: n_inputs, n_hidden, n_outputs, learning_rate, epochs, seed
            # hidden_activation y output_activation se establecen después como atributos
            constructor_params = ['n_inputs', 'n_hidden', 'n_outputs', 'learning_rate', 'epochs', 'seed']
            for key in constructor_params:
                if key in cfg:
                    params[key] = cfg[key]
        
        # Usar factory_type en lugar de meta.get('type') para asegurar compatibilidad
        model = ml_factory.create_model(factory_type, params)
    except Exception as e:
        print(f"Warning: Fallo factory instanciando '{factory_type}': {e}")
        import traceback
        traceback.print_exc()
        raise RuntimeError(f"No se pudo cargar el modelo '{name}': fallo en la creación del modelo") from e

    # Restauración de estado
    if model_type_str == 'neural_network':
        # Restaurar Matrices
        model.W1 = model_data.get('W1', [])
        model.W2 = model_data.get('W2', [])
        model.B1 = model_data.get('B1', [])
        model.B2 = model_data.get('B2', [])
        # Restaurar Configuración de Activación
        cfg = model_data.get('config', {})
        model.hidden_activation = cfg.get('hidden_activation', 'sigmoid')
        model.output_activation = cfg.get('output_activation', 'sigmoid')
        # Restaurar escalas
        if 'act_scales' in model_data:
            model.act_scales = model_data['act_scales']

    elif model_type_str == 'tree' and hasattr(model, 'root'):
        from .ml_compat import _unflatten_arrays_to_tree
        if 'struct' in model_data:
            model.root = _unflatten_arrays_to_tree(model_data['struct'])
            
    elif model_type_str == 'forest' and hasattr(model, 'trees'):
        from .ml_compat import _unflatten_arrays_to_tree
        model.trees = []
        for tree_struct in model_data.get('trees', []):
            dt = ml_runtime.DecisionTreeClassifier()
            dt.root = _unflatten_arrays_to_tree(tree_struct)
            model.trees.append(dt)

    elif model_type_str == 'linear_svm':
        model.weights = model_data.get('weights', [])
        if hasattr(model, 'bias') and 'bias' in model_data: model.bias = model_data['bias']
        if hasattr(model, 'intercept') and 'intercept' in model_data: model.intercept = model_data['intercept']

    elif model_type_str == 'knn':
        model.X_train = model_data.get('X_train', [])
        model.y_train = model_data.get('y_train', [])

    # Restaurar Scaler
    if 'scaler' in data:
        s_data = data['scaler']
        scaler = ml_runtime.MiniScaler(method=s_data['method'], feature_range=s_data['feature_range'])
        scaler.params = s_data['params']
        scaler.n_features_trained = s_data.get('n_features_trained')
        model.scaler = scaler

    _MODEL_REGISTRY[name] = {'mode': 'mini', 'model': model, 'meta': meta}
    print(f"Modelo '{name}' cargado exitosamente.")

def export_to_c(name: str) -> str:
    """
    Genera el código C completo.
    Detecta automáticamente si debe usar CMSISAdapter (NN) o el exportador nativo.
    """
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Modelo '{name}' no encontrado")
    
    entry = _MODEL_REGISTRY[name]
    model = entry['model']
    
    # Optimización Automática (Cuantificación)
    if hasattr(model, 'quantize') and not getattr(model, 'quantized', False):
        print(f"[Export] Optimizando modelo '{name}' (Cuantificación int8)...")
        # Si load_model restauró act_scales, esto funcionará sin dataset
        try:
            model.quantize()
        except RuntimeError as e:
            return f"// Error de exportación: {e} (¿Olvidaste entrenar o cargar act_scales?)"

    code = []
    
    # Decisión de Exportador: ¿Usar Adapter o Nativo?
    used_adapter = False
    adapter_code = ""
    
    # Si es una Neural Network (tiene q_W1) usamos el Adapter avanzado
    if hasattr(model, 'q_W1'):
        try:
            print(f"[Export] Usando CMSISAdapter para '{name}'...")
            adapter = CMSISAdapter(model)
            
            # Truco para capturar la salida del adapter (que escribe a archivo) en un string
            with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp:
                tmp_path = tmp.name
            
            adapter.generate_c(tmp_path)
            
            with open(tmp_path, 'r') as f:
                adapter_code = f.read()
            
            os.remove(tmp_path)
            used_adapter = True
        except Exception as e:
            print(f"[Export] Falló CMSISAdapter: {e}. Usando fallback nativo.")

    # Construcción del Código Final
    code.append(f"// --- MiniML Export: {name} ---")
    
    # Preprocessing (Scaler)
    has_scaler = False
    if hasattr(model, 'scaler') and model.scaler:
        has_scaler = True
        code.append("// [1] Preprocessing Module")
        code.append(model.scaler.to_arduino_code(fn_name="model_preprocess"))
        code.append("")
        
    # Inference Core
    code.append("// [2] Inference Core")
    if used_adapter:
        code.append("// Generated via CMSISAdapter (Fixed-Point / CMSIS-NN compatible)")
        code.append(adapter_code)
        # El adapter genera una función llamada 'predict_int8', hay que tenerlo en cuenta en el wrapper
    elif hasattr(model, "to_arduino_code"):
        code.append("// Generated via MiniML Runtime (Native)")
        code.append(model.to_arduino_code(fn_name="model_predict_core"))
    else:
        return "// Error: Modelo no exportable."

    code.append("")
    code.append("// [3] Public API Wrapper")
    
    # Wrapper Unificado
    if used_adapter:
        # Wrapper para el Adapter (convierte float -> int8 -> float)
        # Asume que el Adapter generó 'predict_int8(int8_t* in, int8_t* out)'
        # Y necesitamos las escalas para des-cuantizar entrada/salida
        s_in = model.act_scales['input']
        s_out = model.act_scales['output']
        
        code.append(f"void predict(float *inputs, float *outputs) {{")
        if has_scaler:
            code.append("  model_preprocess(inputs); // Scale inputs (0-1 approx)")
            
        code.append(f"  // Quantize Inputs (Float -> Int8)")
        code.append(f"  int8_t input_q[{model.n_inputs}];")
        code.append(f"  for(int i=0; i<{model.n_inputs}; i++) input_q[i] = (int8_t)(inputs[i] / {s_in:.8f});")
        
        code.append(f"  // Run Inference")
        code.append(f"  int8_t output_q[{model.n_outputs}];")
        code.append(f"  predict_int8(input_q, output_q);")
        
        code.append(f"  // Dequantize Outputs (Int8 -> Float)")
        code.append(f"  for(int i=0; i<{model.n_outputs}; i++) outputs[i] = (float)output_q[i] * {s_out:.8f};")
        code.append("}")
        
    elif hasattr(model, 'n_outputs'): # NN Nativa (ml_runtime)
        code.append("void predict(float *inputs, float *outputs) {")
        if has_scaler:
            code.append("  model_preprocess(inputs);")
        code.append("  model_predict_core(inputs, outputs);")
        code.append("}")
        
    else: # Arboles / Regresión lineal
        code.append("float predict(float *inputs) {")
        if has_scaler:
            code.append("  model_preprocess(inputs);")
        code.append("  return model_predict_core(inputs);")
        code.append("}")
    
    return "\n".join(code)

# -----------------------------------------------------------------------------
# Self-test (Manager Integration)
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=== MiniML Manager Self-Test ===")
    
    # Datos de prueba con huecos (para probar imputación)
    # Feature 1: lineal, Feature 2: ruido/constante, Target: 2*f1
    dirty_dataset = [
        [1.0, 5.0, 2.0],
        [2.0, None, 4.0], # Faltan valores
        [3.0, 5.0, 6.0],
        [4.0, 5.0, 8.0],
        [5.0, None, 10.0]
    ]

    model_name = "test_linear_pipeline"
    filename = "test_model_export.json"

    try:
        # 1. Entrenamiento con Pipeline (Imputación -> Scaling -> Train)
        print("\n[TEST] Entrenando Pipeline Completo...")
        res = train_pipeline(
            model_name=model_name,
            dataset=dirty_dataset,
            model_type="linear_regression", # mapeado a MiniLinearModel en factory
            params={"learning_rate": 0.01, "epochs": 1000},
            scaling="minmax"
        )
        print(f"  > Modelo entrenado. Metadata: {res['meta']}")

        # 2. Predicción (Debe aplicar scaler automáticamente)
        print("\n[TEST] Predicción (con auto-scaling)...")
        # Entrada cruda (sin escalar), el manager debe escalarla
        X_new = [[6.0, 5.0]] 
        preds = predict(name=model_name, X=X_new)
        print(f"  > Predicción para input {X_new}: {preds} (Esperado ~12.0)")

        # 3. Persistencia (Guardar y Cargar)
        print("\n[TEST] Guardar y Cargar Modelo...")
        save_model(model_name, filename)
        
        # Limpiar memoria para asegurar carga real
        clear_registry()
        assert len(list_models()) == 0
        
        # Recargar
        load_model(model_name, filename)
        print("  > Modelo recargado desde disco.")
        
        # Predicción post-carga
        preds_loaded = predict(name=model_name, X=X_new)
        print(f"  > Predicción post-carga: {preds_loaded}")
        
        # 4. Exportación a C
        print("\n[TEST] Generación de Código C...")
        c_code = export_to_c(model_name)
        print(f"  > Código generado ({len(c_code)} caracteres). Preview:")
        print("-" * 20)
        print('\n'.join(c_code.split('\n')[:10])) # Primeras 10 líneas
        print("...")
        print("-" * 20)

        # 5. Evaluación
        print("\n[TEST] Evaluación de Métricas...")
        y_true = [12.0]
        metrics = evaluate_ext(y_true, preds_loaded, metrics=["mse", "mae"])
        print(f"  > Métricas: {metrics}")

        # Limpieza
        if os.path.exists(filename):
            os.remove(filename)
        print("\n=== Todos los tests de Manager pasaron correctamente. ===")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[ERROR] El test de Manager falló: {e}")