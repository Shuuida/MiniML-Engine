"""
ml_factory.py
-------------
Patrón Factory para la instanciación de modelos MiniML.
Desacopla ml_exporter de ml_runtime para evitar dependencias circulares
y facilitar la extensión (ej. futuros modelos GPU).
"""

from typing import Any, Dict, Optional
from . import ml_runtime

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

    else:
        raise ValueError(f"Factory: Tipo de modelo desconocido '{model_type}'")