"""
ML Factory
===========================
Patrón Factory para la instanciación de modelos MiniML.
Desacopla ml_exporter de ml_runtime para evitar dependencias circulares
y facilitar la extensión (ej. futuros modelos GPU).
"""

from typing import Any, Dict, Optional
from miniml import ml_runtime
from miniml.autograd import layers as nn
from miniml.autograd import qat

def create_model(model_type: str, params: Optional[Dict[str, Any]] = None) -> Any:
    """
    Crea una instancia de un modelo MiniML basado en su tipo (string).
    
    Args:
        model_type (str): Identificador del tipo de modelo (ej. 'DecisionTree', 'RandomForest', 'NeuralNetwork').
        params (dict, optional): Parámetros para inicializar el modelo.

    Returns:
        Instance: Instancia del modelo configurado.
    
    Raises:
        ValueError: Si el tipo de modelo no es reconocido.
    """
    params = params or {}
    model_type = model_type.lower()

    # Árboles
    if "decisiontree" in model_type:
        # Detectar si es regresión o clasificación basado en params o nombre
        if "regressor" in model_type:
            return ml_runtime.DecisionTreeRegressor(**params)
        return ml_runtime.DecisionTreeClassifier(**params)

    elif "randomforest" in model_type:
        if "regressor" in model_type:
            return ml_runtime.RandomForestRegressor(**params)
        return ml_runtime.RandomForestClassifier(**params)

    # Modelos Lineales / SVM
    elif "linear" in model_type or "regression" in model_type:
        return ml_runtime.MiniLinearModel(**params)
    
    elif "svm" in model_type:
        return ml_runtime.MiniSVM(**params)

    # Redes Neuronales
    elif "neural" in model_type or "network" in model_type:
        return ml_runtime.MiniNeuralNetwork(**params)

    # KNN
    elif "knn" in model_type or "neighbor" in model_type:
        return ml_runtime.KNearestNeighbors(**params)

    # Preprocesamiento
    elif "scaler" in model_type:
        return ml_runtime.MiniScaler(**params)

    elif "sequential" in model_type or "deepnet" in model_type:
        layers_config = params.get("layers_config", [])

        # Default Perceptron si no hay config
        if not layers_config:
            model_layers = [
                nn.Linear(2, 4), 
                nn.ReLU(), 
                nn.Linear(4, 1)
            ]
        else:
            # Construcción dinámica
            model_layers = []
            for layer_def in layers_config:
                l_type = str(layer_def.get("type", "")).lower()
                
                # Capas densas
                if "linear" in l_type or "dense" in l_type:
                    n_in = int(layer_def.get("in", 1))
                    n_out = int(layer_def.get("out", 1))
                    model_layers.append(nn.Linear(n_in, n_out))
                
                # Activaciones
                elif "relu" in l_type:
                    model_layers.append(nn.ReLU())
                
                # Convoluciones
                elif "conv2d" in l_type or "conv" in l_type:
                    in_c = int(layer_def.get("in_channels", 1))
                    out_c = int(layer_def.get("out_channels", 1))
                    k_size = layer_def.get("kernel_size", 3)
                    stride = int(layer_def.get("stride", 1))
                    padding = int(layer_def.get("padding", 0))
                    
                    model_layers.append(nn.Conv2d(
                        in_channels=in_c,
                        out_channels=out_c,
                        kernel_size=k_size,
                        stride=stride,
                        padding=padding
                    ))

                # Pooling y utilidades
                elif "maxpool" in l_type or "pool" in l_type:
                    k_size = layer_def.get("kernel_size", 2)
                    stride = layer_def.get("stride", 2)
                    model_layers.append(nn.MaxPool2d(kernel_size=k_size, stride=stride))
                
                elif "flatten" in l_type:
                    model_layers.append(nn.Flatten())
                
                # Cuantización
                elif "quant" in l_type or "qat" in l_type:
                    q_mode = layer_def.get("mode", "per_tensor") 
                    model_layers.append(qat.FakeQuant(mode=q_mode))

        return nn.Sequential(model_layers)

    else:
        raise ValueError(f"Factory: Tipo de modelo desconocido '{model_type}'")