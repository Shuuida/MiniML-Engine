"""
MiniML: Framework de Machine Learning Optimizado para Sistemas Embebidos.
Sin dependencias externas.
"""

from .ml_manager import (
    train_pipeline,
    predict,
    save_model,
    load_model,
    evaluate_ext,
    export_to_c
)

from .ml_runtime import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    MiniLinearModel,
    MiniSVM,
    MiniNeuralNetwork,
    KNearestNeighbors,
    MiniScaler
)

# Exportar módulos completos para acceso directo desde tests
from . import ml_manager
from . import ml_runtime
from . import ml_factory
from . import ml_exporter

__version__ = "1.0.0"
__author__ = "Wilner Manzanares (Michego Takoro 'Shuuida')"