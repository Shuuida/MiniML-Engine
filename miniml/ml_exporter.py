"""
==============================
Gestor de exportación para pipelines ML en el ecosistema MiniML/Sklearn.

Provee herramientas para:
- Exportar estructuras ML (clasificación/regresión) a JSON.
- Reconstruir pipelines desde JSON.
- Guardar modelos entrenados (_MODEL_REGISTRY) junto con su pipeline.
- Registrar errores y trazas de exportación.
"""

from typing import List, Dict, Any, Optional
import json
import time
import os
import importlib
from .ml_compat import _flatten_tree_to_arrays

# Importación de ml_factory con fallback
try:
    from . import ml_factory
except ImportError:
    # Fallback para ejecución local sin paquete
    import ml_factory

# Opcional: importar _MODEL_REGISTRY desde ml_manager
try:
    # Intentamos importar ml_manager para acceder al registro de modelos
    ml_manager = importlib.import_module(".ml_manager")
    _MODEL_REGISTRY = getattr(ml_manager, "_MODEL_REGISTRY", {})
    _MINIML_AVAILABLE = True
except Exception:
    _MODEL_REGISTRY = {}
    _MINIML_AVAILABLE = False

# Intenta importar sklearn opcionalmente (para detección y desambiguación)
_sklearn = None
try:
    _sklearn = importlib.import_module("sklearn.tree")
except Exception:
    pass

# Registro de eventos internos
_EXPORT_LOG: List[str] = []


# ------------------------
# UTILIDADES INTERNAS

def _log(msg: str):
    """Registra un mensaje de depuración o advertencia en el buffer interno."""
    ts = time.strftime("[%H:%M:%S]")
    _EXPORT_LOG.append(f"{ts} {msg}")


def get_export_log(limit: int = 25) -> List[str]:
    """Devuelve las últimas líneas del registro interno de exportación."""
    return _EXPORT_LOG[-limit:]

def export_struct_to_json_file(struct_data: Dict[str, Any], path: str, *,
                               pretty: bool = True) -> None:
    """
    Guarda la estructura ML como un archivo JSON exportable.
    Incluye trazas internas del proceso.
    """
    # Crear directorio si no existe (solo si hay un directorio en la ruta)
    dir_path = os.path.dirname(path)
    if dir_path:  # Solo crear si hay un directorio (no cadena vacía)
        os.makedirs(dir_path, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(struct_data, f, indent=4, ensure_ascii=False)
        else:
            json.dump(struct_data, f, separators=(",", ":"), ensure_ascii=False)
    _log(f"Estructura ML exportada correctamente a {path}")


# -------------------------
# EXPORTACIÓN DE MODELOS

def export_model_snapshot(model_name: str, *,
                          include_pipeline: bool = True,
                          pipeline_struct: Optional[Dict[str, Any]] = None,
                          output_dir: str = "exports",
                          pretty: bool = True) -> Optional[str]:
    """
    Exporta un modelo entrenado del registro interno junto con su pipeline (si aplica).

    Parámetros:
        model_name: nombre del modelo en el registro interno (_MODEL_REGISTRY).
        include_pipeline: si True, incluye la estructura del pipeline asociado.
        pipeline_struct: estructura ML opcional a incluir.
        output_dir: carpeta donde se guardará el archivo exportado.
        pretty: si True, guarda el JSON con formato legible.

    Retorna:
        Ruta del archivo exportado o None si el modelo no existe.
    """
    model_entry = _MODEL_REGISTRY.get(model_name)
    if not model_entry:
        _log(f"Modelo '{model_name}' no encontrado en _MODEL_REGISTRY.")
        return None

    snapshot = {
        "meta": {
            "engine": "MiniML",
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "mode": model_entry.get("mode", "unknown"),
            "type": model_entry.get("type", "unknown"),
        },
        "model": {}
    }

    # Intentar serializar parámetros del modelo (si es sklearn o MiniML)
    model = model_entry.get("model")
    if hasattr(model, "__dict__"):
        try:
            snapshot["model"]["params"] = model.__dict__
        except Exception as e:
            _log(f"No se pudo serializar __dict__ del modelo: {e}")

    # Incluir pipeline si se solicita
    if include_pipeline:
        if not pipeline_struct:
            pipeline_struct = {"pipeline": [], "meta": {"info": "Sin estructura asociada"}}
        snapshot["pipeline"] = pipeline_struct

    # Guardar archivo JSON
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{model_name}_snapshot.json")
    with open(filename, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(snapshot, f, indent=4, ensure_ascii=False)
        else:
            json.dump(snapshot, f, separators=(",", ":"), ensure_ascii=False)

    _log(f"Snapshot de modelo '{model_name}' exportado a {filename}")
    return filename

# -----------------------------------------------------------------
# SERIALIZACIÓN Y DESERIALIZACIÓN DE MODELOS (MiniML / sklearn)

def serialize_model(model_obj, metadata=None):
    """
    Serializa un modelo ML (MiniML o sklearn) en un diccionario JSON-safe.
    Usado por file_handler.save_model().
    """
    import inspect
    metadata = metadata or {}
    data = {"meta": metadata}

    try:
        # sklearn
        if hasattr(model_obj, "get_params"):
            params = model_obj.get_params()
            data["framework"] = "sklearn"
            data["params"] = params
            data["repr"] = repr(model_obj)
            return data

        # MiniML (DecisionTree / RandomForest personalizados)
        elif hasattr(model_obj, "root") or hasattr(model_obj, "trees"):
            data["framework"] = "MiniML"
            # CORRECCIÓN C-COMPATIBLE
            if hasattr(model_obj, "trees"): # RandomForest
                data["type"] = "RandomForest"
                data["tree_structs"] = []
                for t in model_obj.trees:
                    root = getattr(t, "root", None)
                    if root:
                        data["tree_structs"].append(_flatten_tree_to_arrays(root))
            elif hasattr(model_obj, "root"): # DecisionTree
                data["type"] = "DecisionTree"
                if model_obj.root:
                    data["tree_struct"] = _flatten_tree_to_arrays(model_obj.root)
            data["repr"] = f"{type(model_obj).__name__} - MiniML structure"
            return data

        elif hasattr(model_obj, "coefficients") and hasattr(model_obj, "intercept"):
            data["framework"] = "MiniML"
            data["type"] = "MiniLinearModel"
            data["coefficients"] = getattr(model_obj, "coefficients", [])
            data["intercept"] = getattr(model_obj, "intercept", 0.0)
            data["repr"] = "MiniLinearModel - MiniML"
            return data

        elif hasattr(model_obj, "kernel") and hasattr(model_obj, "weights"):
            data["framework"] = "MiniML"
            data["type"] = "MiniSVM"
            data["kernel"] = getattr(model_obj, "kernel", "linear")
            data["weights"] = getattr(model_obj, "weights", [])
            data["bias"] = getattr(model_obj, "bias", 0.0)
            data["support_vectors"] = getattr(model_obj, "support_vectors", []) 
            data["repr"] = "MiniSVM - MiniML"
            return data

        elif hasattr(model_obj, "W1") and hasattr(model_obj, "n_hidden"):
            data["framework"] = "MiniML"
            data["type"] = "MiniNeuralNetwork"
            data["W1"] = getattr(model_obj, "W1", [])
            data["W2"] = getattr(model_obj, "W2", [])
            data["B1"] = getattr(model_obj, "B1", [])
            data["B2"] = getattr(model_obj, "B2", [])
            
            data["config"] = {
                "n_inputs": getattr(model_obj, "n_inputs", 0),
                "n_hidden": getattr(model_obj, "n_hidden", 0),
                "n_outputs": getattr(model_obj, "n_outputs", 0),
                "hidden_activation": getattr(model_obj, "hidden_activation", "sigmoid"),
                "output_activation": getattr(model_obj, "output_activation", "sigmoid")
            }
            
            # Guardar act_scales si existen
            if hasattr(model_obj, "act_scales"):
                data["act_scales"] = model_obj.act_scales

            data["repr"] = "MiniNeuralNetwork (MLP) - MiniML"
            return data

        # KNN (K-Nearest Neighbors)
        elif hasattr(model_obj, "X_train") and hasattr(model_obj, "k"):
            data["framework"] = "MiniML"
            data["type"] = "KNearestNeighbors"
            data["k"] = getattr(model_obj, "k", 3)
            data["task"] = getattr(model_obj, "task", "classification")
            data["X_train"] = getattr(model_obj, "X_train", [])
            data["y_train"] = getattr(model_obj, "y_train", [])
            data["repr"] = f"KNearestNeighbors(k={data['k']}) - MiniML"
            return data


        # Caso genérico
        else:
            data["framework"] = "unknown"
            data["repr"] = str(model_obj)
            return data

    except Exception as e:
        raise RuntimeError(f"Error serializando modelo: {e}")


def deserialize_model(data):
    """
    Deserializa un modelo guardado (desde JSON) y lo reconstruye.
    Compatible con MiniML y sklearn.
    """
    from .ml_runtime import ml_runtime
    framework = data.get("framework")
    model = None

    try:
        # sklearn
        if framework == "sklearn":
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            
            # Extraemos la representación del modelo
            model_repr = data.get("repr", "")
            params = data.get("params", {})

            # Reconstruir el objeto modelo basándose en la representación
            # (Esto es necesario porque el JSON no guarda el tipo exacto)
            if "DecisionTreeRegressor" in model_repr:
                model = DecisionTreeRegressor()
            elif "DecisionTreeClassifier" in model_repr:
                model = DecisionTreeClassifier()
            # ... (Añadir aquí más modelos sklearn a futuro si son soportados por el framework)
            else:
                # Si no se reconoce, devolver el stub como antes para evitar un crash
                model = {"info": "sklearn model stub", "params": params, "error": "Unknown model type in repr"}
                return model # Salir temprano si no se puede reconstruir

            # Si el modelo se creó, re-aplicar sus parámetros
            if params and hasattr(model, "set_params"):
                model.set_params(**params)
            # Ahora 'model' es un objeto sklearn real, no un dict

        # MiniML
            elif framework == "MiniML":
                model_type = data.get("type", "").lower()
                
                # Usar Factory para crear la instancia base
                # Mapeamos tipos del JSON a tipos del Factory
                factory_type = model_type
                if "tree" in model_type: factory_type = "DecisionTree"
                if "forest" in model_type: factory_type = "RandomForest"
                
                # Manejo especial para RandomForest que necesita n_trees en init
                params = {}
                if "tree_structs" in data:
                    params['n_trees'] = len(data["tree_structs"])
                    
                try:
                    model = ml_factory.create_model(model_type, params)
                except ValueError:
                    # Fallback manual si el factory falla o el tipo es muy custom
                    pass
            
            # KNN (K-Nearest Neighbors)
            elif "knearest" in model_type or "knn" in model_type:
                k = data.get("k", 3)
                task = data.get("task", "classification")
                model = ml_runtime.KNearestNeighbors(k=k, task=task)
                model.X_train = data.get("X_train", [])
                model.y_train = data.get("y_train", [])
                # Restauración de metadata dimensional
                if model.X_train and len(model.X_train) > 0:
                    model.n_features_trained = len(model.X_train[0])
                return model
            
            # LinearRegression
            elif "linear" in model_type or "regression" in model_type:
                model = ml_runtime.MiniLinearModel() 
                model.coefficients = data.get("coefficients", [])
                model.intercept = data.get("intercept", 0.0)
                return model

            # SVM
            elif "svm" in model_type:
                model = ml_runtime.MiniSVM()
                model.support_vectors = data.get("support_vectors", [])
                model.weights = data.get("weights", [])
                model.bias = data.get("bias", 0.0)
                return model

            # Neural Network
            elif "neural" in data.get("type", "").lower() or "network" in data.get("type", "").lower():
                # Recuperar configuración para instanciar correctamente
                config = data.get("config", {})
                
                model = ml_runtime.MiniNeuralNetwork(
                    n_inputs=config.get("n_inputs", 1),
                    n_hidden=config.get("n_hidden", 1),
                    n_outputs=config.get("n_outputs", 1)
                )
                
                # Restaurar Pesos
                model.W1 = data.get("W1", [])
                model.W2 = data.get("W2", [])
                model.B1 = data.get("B1", [])
                model.B2 = data.get("B2", [])
                
                # Restaurar Activaciones
                model.hidden_activation = config.get("hidden_activation", "sigmoid")
                model.output_activation = config.get("output_activation", "sigmoid")
                
                model.act_scales = data.get("act_scales", {})
                
                return model

        return {"framework": framework or "unknown", "repr": repr(data)}

    except Exception as e:
        raise RuntimeError(f"Error deserializando modelo: {e}")

# ---------------------------------------------------------------
# EXTRACTOR DE ESTRUCTURAS DE MODELOS (para visualización y exportación)

def extract_model_structure(model_obj):
    """
    Extrae una representación estructural uniforme de un modelo ML.
    Compatible con MiniML y sklearn.
    
    Devuelve un diccionario estructurado que puede ser convertido
    fácilmente en bloques visuales por ml_struct_rules.struct_to_block().
    """
    if isinstance(model_obj, dict) and 'model' in model_obj:
        model_obj = model_obj['model']

    try:
        # MiniML: DecisionTree o RandomForest
        if hasattr(model_obj, "root") or hasattr(model_obj, "trees"):
            struct = {
                "framework": "MiniML",
                "type": "RandomForest" if hasattr(model_obj, "trees") else "DecisionTree",
            }
            # Aplanar los árboles a arrays compatibles con C
            if hasattr(model_obj, "trees"):
                struct["tree_structs"] = [] # Usar el mismo nombre que en serialize_model
                for t in model_obj.trees:
                    root = getattr(t, "root", None)
                    if root:
                        struct["tree_structs"].append(_flatten_tree_to_arrays(root))
            elif hasattr(model_obj, "root"):
                root = getattr(model_obj, "root", None)
                if root:
                    struct["tree_struct"] = _flatten_tree_to_arrays(root) # Usar el mismo nombre

            if hasattr(model_obj, "metadata") and "saved_at" in model_obj.metadata:
                struct["saved_at"] = model_obj.metadata["saved_at"]

            return struct
        
        model_type = str(type(model_obj)).split(".")[-1].replace("'>", "").lower()


        # MiniML KNN (K-Nearest Neighbors) - Integración Fiel
        if "knearest" in model_type or "knn" in model_type:
            struct = {
                "framework": "MiniML",
                "type": "KNearestNeighbors",
                "k": getattr(model_obj, "k", 3),
                "task": getattr(model_obj, "task", "classification"),
                "n_samples": len(model_obj.X_train) if hasattr(model_obj, "X_train") and model_obj.X_train else 0,
                "n_features": len(model_obj.X_train[0]) if hasattr(model_obj, "X_train") and model_obj.X_train else 0,
                "X_train": getattr(model_obj, "X_train", []),
                "y_train": getattr(model_obj, "y_train", [])
            }
            return struct

        # MiniML SVM
        if "svm" in model_type:
            struct = {
                "framework": "MiniML",
                "type": "SVM",
                "support_vectors": getattr(model_obj, "support_vectors", []),
                "weights": getattr(model_obj, "weights", []),
                "bias": getattr(model_obj, "bias", 0.0),
                "kernel": getattr(model_obj, "kernel", "linear"),
            }
            return struct

        # MiniML Linear Regression
        if "linear" in model_type or "regression" in model_type:
            struct = {
                "framework": "MiniML",
                "type": "LinearRegression",
                "coefficients": getattr(model_obj, "coefficients", []),
                "intercept": getattr(model_obj, "intercept", 0.0),
                "trained_at": getattr(model_obj, "trained_at", None),
            }
            return struct

        # MiniML Neural Network
        model_type = str(type(model_obj)).lower()
        if "neural" in model_type or "network" in model_type:
            struct = {
                "framework": "MiniML",
                "type": "NeuralNetwork",
                "W1": getattr(model_obj, "W1", []),
                "W2": getattr(model_obj, "W2", []),
                "B1": getattr(model_obj, "B1", []),
                "B2": getattr(model_obj, "B2", []),
                # Metadata útil
                "config": {
                    "n_inputs": getattr(model_obj, "n_inputs", 0),
                    "n_hidden": getattr(model_obj, "n_hidden", 0),
                    "n_outputs": getattr(model_obj, "n_outputs", 0)
                },
                "trained_at": getattr(model_obj, "trained_at", None),
            }
            # Incluir escalas si existen (útil para debug visual)
            if hasattr(model_obj, "act_scales"):
                struct["calibration"] = model_obj.act_scales
                
            return struct

        try:
            # Intentar detectar/extraer estructura sklearn (si aplica)
            if "sklearn" in str(type(model_obj)):
                struct = {"framework": "sklearn"}
                model_type = type(model_obj).__name__

                # RandomForest (ensemble)
                if hasattr(model_obj, "estimators_"):
                    struct["type"] = "RandomForest"
                    struct["trees"] = []
                    for est in model_obj.estimators_:
                        # est.tree_ tiene arrays; guardamos representaciones simples
                        tree = est.tree_
                        struct["trees"].append({
                            "features": getattr(tree, "feature", []).tolist() if hasattr(tree, "feature") else [],
                            "thresholds": getattr(tree, "threshold", []).tolist() if hasattr(tree, "threshold") else [],
                            "values": getattr(tree, "value", []).tolist() if hasattr(tree, "value") else []
                        })

                # DecisionTree (single)
                elif hasattr(model_obj, "tree_"):
                    tree = model_obj.tree_
                    struct["type"] = "DecisionTree"
                    struct["tree"] = {
                        "features": getattr(tree, "feature", []).tolist() if hasattr(tree, "feature") else [],
                        "thresholds": getattr(tree, "threshold", []).tolist() if hasattr(tree, "threshold") else [],
                        "values": getattr(tree, "value", []).tolist() if hasattr(tree, "value") else []
                    }

                # metadata opcional
                if hasattr(model_obj, "metadata") and "saved_at" in getattr(model_obj, "metadata", {}):
                    struct["saved_at"] = model_obj.metadata["saved_at"]

                # Si hemos llegado hasta aquí, retornamos la estructura sklearn válida (o parcialmente válida)
                return struct

            # Si no es sklearn, salimos del try para retornar fallback "unknown" abajo
        except Exception as e:
            # Si sklearn estaba presente pero hubo algún error al extraer la estructura,
            # devolvemos un struct que indique que es sklearn pero no soportado (y el error).
            return {"framework": "sklearn", "type": "unsupported", "error": str(e), "repr": repr(model_obj)}
        
        # No era sklearn (o no se intentó extracción) -> devolver unknown (fallthrough seguro)
        return {"framework": "unknown", "repr": repr(model_obj)}

    except Exception as e:
        raise RuntimeError(f"Error extrayendo estructura del modelo: {e}")