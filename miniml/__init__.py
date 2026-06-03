"""
MiniML Engine: Framework de Machine Learning y Deep Learning Optimizado 
para Sistemas Embebidos (Edge AI).
Filosofía "Zero-Dependencies" - Python Puro.
"""

# ---------------------------------------------------------
# API CLÁSICA DE MINIML
# ---------------------------------------------------------
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

# ---------------------------------------------------------
# HERRAMIENTAS MLOps & CLI
# ---------------------------------------------------------
from .ml_exporter import print_cli_summary, serialize_model, deserialize_model

# ---------------------------------------------------------
# EXTENSIÓN MINITENSOR (Deep Learning & Autograd)
# ---------------------------------------------------------
try:
    from .autograd.tensor import Tensor
    from .autograd import layers as nn
    from .autograd import optim
    from .autograd import qat
except ImportError:
    pass  # Permite que MiniML siga funcionando si faltan archivos

# ---------------------------------------------------------
# EXPORTADORES EDGE AI (C++ y Rust)
# ---------------------------------------------------------
try:
    from .exporters import cpp_writer
    from .exporters import rust_writer
    from .exporters import quantizer
    from .exporters import library_packer
except ImportError:
    pass

# ---------------------------------------------------------
# HARDWARE Y PROCESAMIENTO DE SEÑALES (DSP)
# ---------------------------------------------------------
try:
    from .hardware.serial_manager import serial_manager
    from .hardware.virtual_sensor import VirtualSerial
    from .dsp import dsp_runtime
except ImportError:
    pass


# Metadatos del paquete
__version__ = "1.1.0"
__author__ = "Wilner Manzanares (Michego Takoro 'Shuuida')"