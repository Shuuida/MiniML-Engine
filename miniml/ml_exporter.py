"""
ML Exporter para MiniML Engine (CLI)
======================================================
Gestor de exportación para pipelines ML en el ecosistema MiniML/Sklearn y MiniTensor.

Provee herramientas para:
- Exportar estructuras ML (Clásico y Deep Learning) a JSON.
- Reconstruir pipelines y pesos desde disco.
- Generar Firmware C++/Rust para microcontroladores.
- Imprimir resúmenes de arquitectura en terminal.
"""

from typing import List, Dict, Any, Optional
import json
import time
import os
import importlib
from .ml_compat import _flatten_tree_to_arrays

try:
    from . import ml_factory
except ImportError:
    import ml_factory

try:
    from .autograd import layers as nn
    from .exporters import rust_writer, cpp_writer
except ImportError:
    nn = None
    rust_writer = None
    cpp_writer = None

# Registro de modelos
try:
    ml_manager = importlib.import_module(".ml_manager")
    _MODEL_REGISTRY = getattr(ml_manager, "_MODEL_REGISTRY", {})
except Exception:
    _MODEL_REGISTRY = {}

# Soporte opcional para Sklearn
try:
    importlib.import_module("sklearn.tree")
    _SKLEARN_AVAILABLE = True
except Exception:
    _SKLEARN_AVAILABLE = False

_EXPORT_LOG: List[str] = []

def _log(msg: str):
    """Registra un mensaje en el buffer y en consola si estamos en modo CLI."""
    ts = time.strftime("[%H:%M:%S]")
    log_msg = f"{ts} {msg}"
    _EXPORT_LOG.append(log_msg)
    # Print directo para el entorno CLI
    print(f"[EXPORTER] {msg}")

def get_export_log(limit: int = 25) -> List[str]:
    return _EXPORT_LOG[-limit:]

def export_struct_to_json_file(struct_data: Dict[str, Any], path: str, *, pretty: bool = True) -> None:
    dir_path = os.path.dirname(path)
    if dir_path: os.makedirs(dir_path, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        if pretty:
            json.dump(struct_data, f, indent=4, ensure_ascii=False)
        else:
            json.dump(struct_data, f, separators=(",", ":"), ensure_ascii=False)
    _log(f"Estructura ML exportada correctamente a {path}")

def export_model_snapshot(model_name: str, *, include_pipeline: bool = True, pipeline_struct: Optional[Dict[str, Any]] = None, output_dir: str = "exports", pretty: bool = True) -> Optional[str]:
    model_entry = _MODEL_REGISTRY.get(model_name)
    if not model_entry:
        _log(f"Modelo '{model_name}' no encontrado en registro.")
        return None

    snapshot = {
        "meta": {
            "engine": "MiniML",
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_name": model_name,
            "type": model_entry.get("type", "unknown"),
        },
        "model": extract_model_structure(model_entry.get("model"))
    }

    if include_pipeline:
        snapshot["pipeline"] = pipeline_struct or {"meta": {"info": "Sin pipeline"}}

    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{model_name}_snapshot.json")
    export_struct_to_json_file(snapshot, filename, pretty=pretty)
    return filename

def serialize_model(model_obj, metadata=None):
    """Convierte modelos a diccionario JSON-safe (Soporta Sklearn, MiniML y MiniTensor)."""
    data = {"meta": metadata or {}}

    try:
        # Soporte MiniTensor (Deep Learning)
        if hasattr(model_obj, "state_dict") and hasattr(model_obj, "layers"):
            data["framework"] = "MiniTensor"
            data["type"] = "Sequential"
            
            # Guardar arquitectura para reconstrucción
            layers_config = []
            for layer in model_obj.layers:
                cfg = {"type": layer.__class__.__name__}

                #----------------------------------------------------------
                # Guardar el estado de trainable para el futuro (En desarrollo)
                cfg["trainable"] = getattr(layer, "trainable", True)
                # Cuando el generador C++ (en el futuro) lea esto:
                # if not cfg["trainable"]: usar const PROGMEM
                # else: declarar variable normal en RAM

                #----------------------------------------------------------
                if cfg["type"] == "Linear" and hasattr(layer, "weights"):
                    cfg["in"] = len(layer.weights.data[0])
                    cfg["out"] = len(layer.weights.data)
                elif cfg["type"] == "Conv2d":
                    cfg["in_channels"] = getattr(layer, "in_channels", 1)
                    cfg["out_channels"] = getattr(layer, "out_channels", 1)
                    cfg["kernel_size"] = getattr(layer, "kernel_size", (3,3))
                    cfg["stride"] = getattr(layer, "stride", 1)
                    cfg["padding"] = getattr(layer, "padding", 0)
                layers_config.append(cfg)
                
            data["config"] = {"layers_config": layers_config}
            data["state_dict"] = model_obj.state_dict()
            data["repr"] = "MiniTensor Deep Learning Model"
            return data

        # Soporte Sklearn
        if hasattr(model_obj, "get_params"):
            data["framework"] = "sklearn"
            data["params"] = model_obj.get_params()
            data["repr"] = repr(model_obj)
            return data

        # Soporte MiniML Clásico
        if hasattr(model_obj, "root") or hasattr(model_obj, "trees"):
            data["framework"] = "MiniML"
            if hasattr(model_obj, "trees"):
                data["type"] = "RandomForest"
                data["tree_structs"] = [_flatten_tree_to_arrays(getattr(t, "root")) for t in model_obj.trees if getattr(t, "root", None)]
            else:
                data["type"] = "DecisionTree"
                if model_obj.root: data["tree_struct"] = _flatten_tree_to_arrays(model_obj.root)
            return data

        # Linear, SVM, MLP Clásico
        for m_type, attrs in [("MiniLinearModel", ["coefficients", "intercept"]), ("MiniSVM", ["kernel", "weights"])]:
            if all(hasattr(model_obj, a) for a in attrs):
                data["framework"] = "MiniML"
                data["type"] = m_type
                for a in attrs + ["bias", "support_vectors"]:
                    if hasattr(model_obj, a): data[a] = getattr(model_obj, a)
                return data

        return {"framework": "unknown", "repr": str(model_obj)}

    except Exception as e:
        raise RuntimeError(f"Error serializando modelo: {e}")

def deserialize_model(data):
    """Reconstruye un modelo desde JSON."""
    framework = data.get("framework")
    try:
        if framework == "sklearn":
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            model_repr = data.get("repr", "")
            if "DecisionTreeRegressor" in model_repr: model = DecisionTreeRegressor()
            elif "DecisionTreeClassifier" in model_repr: model = DecisionTreeClassifier()
            else: return {"info": "sklearn stub", "params": data.get("params")}
            
            if data.get("params"): model.set_params(**data.get("params"))
            return model

        elif framework == "MiniML":
            m_type = data.get("type", "").lower()
            
            # Arboles Clásicos
            if "tree" in m_type or "forest" in m_type:
                f_type = "RandomForest" if "forest" in m_type else "DecisionTree"
                params = {'n_trees': len(data["tree_structs"])} if "tree_structs" in data else {}
                return ml_factory.create_model(f_type, params) # Fix: Llama al módulo correctamente
            
            # KNN
            if "knn" in m_type or "knearest" in m_type:
                model = ml_factory.create_model("knn", {"k": data.get("k", 3), "task": data.get("task", "classification")})
                model.X_train = data.get("X_train", [])
                model.y_train = data.get("y_train", [])
                return model

        elif framework == "MiniTensor" or data.get("type") == "Sequential":
            config = data.get("config", {})
            layers_conf = config.get("layers_config", [])
            model = ml_factory.create_model("sequential", {"layers_config": layers_conf})
            if "state_dict" in data:
                model.load_state_dict(data["state_dict"])
            return model

        return None
    except Exception as e:
        _log(f"Fallo al reconstruir modelo desde JSON: {e}")
        return None

def extract_model_structure(model_obj):
    """Extrae estructura para informes CLI o JSON."""
    if isinstance(model_obj, dict) and 'model' in model_obj: model_obj = model_obj['model']
    return serialize_model(model_obj)

def print_cli_summary(model_obj: Any):
    """Genera un informe detallado por consola de la arquitectura (Modo CLI MLOps)."""
    print("\n" + "="*60)
    print(" 🛠️  MiniML Engine - Resumen de Arquitectura del Modelo")
    print("="*60)
    
    if hasattr(model_obj, "layers"):
        print(" Framework   : MiniTensor (Autograd Deep Learning)")
        print(" Tipo        : Sequential")
        print("-" * 60)
        print(f" {'ID':<5} | {'CAPA':<20} | {'PARAMETROS / CONFIGURACIÓN'}")
        print("-" * 60)
        
        total_params = 0
        for idx, layer in enumerate(model_obj.layers):
            l_type = layer.__class__.__name__
            info = ""
            
            # Extracción de info segura por tipo de capa
            if hasattr(layer, "weights"):
                w_shape = getattr(layer.weights, "shape", "N/A")
                info = f"Pesos: {w_shape}"
                try:
                    # Intento de cálculo de parámetros aproximado (si shape es tupla)
                    if isinstance(w_shape, tuple):
                        import math
                        params = math.prod(w_shape)
                        if hasattr(layer, "bias") and layer.bias:
                            params += math.prod(getattr(layer.bias, "shape", (0,)))
                        total_params += params
                except: pass
            elif "Conv2d" in l_type:
                info = f"In:{getattr(layer,'in_channels',1)} Out:{getattr(layer,'out_channels',1)} K:{getattr(layer,'kernel_size','?')}"
            elif "MaxPool" in l_type:
                info = f"Kernel: {getattr(layer, 'kernel_size', '?')} Stride: {getattr(layer, 'stride', '?')}"
            
            print(f" {idx:<5} | {l_type:<20} | {info}")
            
        print("-" * 60)
        print(f" Parámetros Entrenables Estimados: {total_params}")

    else:
        # Información para ML Clásico (Árboles, SVM, etc.)
        m_type = model_obj.__class__.__name__
        print(f" Framework   : MiniML Clásico / Sklearn")
        print(f" Algoritmo   : {m_type}")
        print("-" * 60)
        if hasattr(model_obj, "trees"):
            print(f" N° de Árboles en el Bosque: {len(model_obj.trees)}")
        elif hasattr(model_obj, "X_train") and hasattr(model_obj, "k"):
            print(f" K-Vecinos: {getattr(model_obj, 'k', '?')}")
            print(f" Muestras Entrenadas: {len(getattr(model_obj, 'X_train', []))}")
        elif hasattr(model_obj, "weights"):
            print(f" Pesos (Weights): {len(getattr(model_obj, 'weights', []))}")
            print(f" Sesgo (Bias): {getattr(model_obj, 'bias', 0.0)}")

    print("="*60 + "\n")

def export_to_code(model_obj, language="cpp", input_shape=(1, 28, 28)):
    """Exporta el modelo a código fuente C++ o Rust para hardware Edge."""
    if language == "rust":
        if rust_writer: return rust_writer.generate_rust_code(model_obj, input_shape)
        return "// Error: Módulo rust_writer no disponible."
    else:
        if cpp_writer: return cpp_writer.generate_cpp_code(model_obj, input_shape)
        return "// Error: Módulo cpp_writer no disponible."