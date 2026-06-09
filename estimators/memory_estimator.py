"""
Memory Estimator for MiniML Engine / Arduino Uno
=========================================
Estima el consumo de memoria Flash (Programa) y SRAM (Variables)
para modelos MiniML exportados a C/C++ y Rust (Arduino).

Soporta:
- ml_runtime (Classic ML): DT, RF, SVM, KNN, MLP.
- autograd (Deep Learning v3.0): CNN, TCN, DSC-CNN, ResNet, Autoencoders.
- Edge Transfer Learning (SGD en SRAM).

Base de cálculo (Arduino Uno - ATmega328P):
- Flash Total: 32,256 bytes (32KB - 0.5KB bootloader)
- SRAM Total: 2,048 bytes (2KB)
- Float size: 4 bytes (Standard) / 1 byte (Quantized INT8)
- Pointer/Int size: 2 bytes
"""

from typing import Dict, Any, Tuple
import math

# Intentar importar módulos del core si están disponibles
try:
    from miniml.ml_runtime import ml_runtime
except ImportError:
    ml_runtime = None

try:
    from miniml.autograd import layers as nn
    from miniml.autograd import tensor
except ImportError:
    nn = None
    tensor = None

# Constantes de Hardware (Arduino Uno)
UNO_FLASH_LIMIT = 32256
UNO_SRAM_LIMIT = 2048
FLOAT_SIZE = 4
INT8_SIZE = 1
INT_SIZE = 2

# Overheads estimados por Lenguaje (Bytes)
OVERHEADS = {
    'C': 600,       # Estructuras simples, sin runtime pesado
    'C++': 1500,    # Clases, templates, MiniTensor Lib
    'Rust': 3500    # Panic handler, Core lib (no_std)
}

def estimate_memory(model: Any, quantized: bool = False, target_flash: int = UNO_FLASH_LIMIT, 
                    target_sram: int = UNO_SRAM_LIMIT, language: str = 'C', 
                    input_shape: Tuple[int, ...] = None, on_device_learning: bool = False) -> Dict[str, Any]:
    """
    Calcula el uso de memoria estimado del modelo.
    """
    flash_bytes = 0
    sram_bytes = 0
    weights_bytes = 0
    
    # Seleccionar peso de datos base
    data_size = INT8_SIZE if quantized else FLOAT_SIZE
    
    # Overhead base del lenguaje
    overhead_code = OVERHEADS.get(language, 1000)

    model_type = type(model).__name__

    # ---------------------------------------------------------
    # MODELOS DEEP LEARNING (V3.0 / MiniTensor / Autograd)
    # ---------------------------------------------------------
    if nn and (isinstance(model, nn.Module) or isinstance(model, nn.Sequential)):
        # Overhead extra por la librería MiniTensor
        overhead_code += 1000 

        # Flash: Suma de todos los parámetros (Flat)
        # Recorremos recursivamente los parámetros del modelo
        total_params = 0
        mutable_params = 0          # Parámetros movidos a SRAM por el SGD
        transfer_learning_sram = 0  # Overhead total extra en RAM

        def _count_elements(data):
            if isinstance(data, list):
                return sum(_count_elements(x) for x in data)
            return 1

        if hasattr(model, 'parameters'):
            params = model.parameters()
            for p in params:
                total_params += _count_elements(p.data)

        # SRAM: Estimación de Arena (Activaciones)
        # Requiere input_shape para propagar dimensiones
        if input_shape:
            max_activation = 0
            current_shape = input_shape
            # Asumimos sequential para la estimación simple
            layers_list = model.layers if hasattr(model, 'layers') else [model]

            # Input inicial ocupa RAM
            input_bytes = math.prod(current_shape) * data_size
            max_activation = input_bytes
            prev_bytes = input_bytes
            
            for layer in layers_list:
                l_type = type(layer).__name__
                is_trainable = getattr(layer, 'trainable', False)

                # Lógica de Estimación On-Device Learning
                if on_device_learning and is_trainable and 'Linear' in l_type:
                    # Extracción directa de las listas anidadas de MiniTensor
                    w_in = len(layer.weights.data)
                    w_out = len(layer.weights.data[0])
                    b_size = len(layer.bias.data[0]) if getattr(layer, 'bias', None) else 0
                    
                    layer_params = (w_in * w_out) + b_size
                    mutable_params += layer_params
                    
                    # Pesos mutables en SRAM (FLOAT32)
                    transfer_learning_sram += layer_params * FLOAT_SIZE 
                    # Buffer estático del error[] para el Gradiente
                    transfer_learning_sram += w_out * FLOAT_SIZE 
                    # Puntero de estado latente
                    transfer_learning_sram += 4 

                # Calcular output shape
                out_shape = _infer_output_shape(layer, current_shape)
                out_bytes = math.prod(out_shape) * data_size

                is_inplace = any(x in l_type for x in ['ReLU', 'Flatten', 'Reshape', 'Dropout'])
                
                if is_inplace:
                    # Si es in-place, no duplica memoria, usa el mismo buffer
                    current_activation = max(prev_bytes, out_bytes)
                else:
                    # Si no, necesita memoria para Entrada y Salida simultáneamente
                    current_activation = prev_bytes + out_bytes

                # Buffer temporal (ej: Im2Col para CNN)
                temp_buffer = _estimate_temp_buffer(layer, current_shape, data_size)
                # El pico es: Lo que ya tenía + (Lo nuevo o el buffer temporal)
                peak = current_activation + temp_buffer
                
                if peak > max_activation:
                    max_activation = peak
                
                current_shape = out_shape
                prev_bytes = out_bytes
            
            # Sumar la carga del Transfer Learning a la memoria volatil
            sram_bytes = max_activation + transfer_learning_sram
        else:
            # Fallback si no hay input_shape: SRAM desconocida (Warning)
            sram_bytes = 0 

        # Restar los parámetros mutables de la Flash (Fueron movidos a RAM)
        weights_bytes = (total_params - mutable_params) * data_size
        flash_bytes = weights_bytes + overhead_code

    # ---------------------------------------------------------
    # MODELOS ML CLÁSICOS (ml_runtime)
    # ---------------------------------------------------------
    elif ml_runtime and isinstance(model, ml_runtime.MiniNeuralNetwork):
        overhead_code = 1500 if language == 'C' else 2500
        total_params = (len(model.W1) * len(model.W1[0])) + len(model.B1) + (len(model.W2) * len(model.W2[0])) + len(model.B2)
        weights_bytes = total_params * data_size
        flash_bytes = weights_bytes + overhead_code
        sram_bytes = (model.n_in + model.n_hid + model.n_out) * data_size

    elif ml_runtime and isinstance(model, (ml_runtime.DecisionTreeClassifier, ml_runtime.DecisionTreeRegressor)):
        overhead_code = 800
        n_nodes = _count_tree_nodes(model.root)
        weights_bytes = n_nodes * 14 # node_size
        flash_bytes = weights_bytes + overhead_code
        sram_bytes = 50 

    elif ml_runtime and isinstance(model, (ml_runtime.RandomForestClassifier, ml_runtime.RandomForestRegressor)):
        overhead_code = 1200
        total_nodes = sum(_count_tree_nodes(tree.root) for tree in model.trees)
        weights_bytes = total_nodes * 14
        flash_bytes = weights_bytes + overhead_code
        sram_bytes = (len(model.trees) * FLOAT_SIZE) + 100

    elif ml_runtime and isinstance(model, ml_runtime.KNearestNeighbors):
        overhead_code = 1000
        n_samples = len(model.y_train)
        weights_bytes = (n_samples * model.n_features_trained * data_size) + (n_samples * data_size)
        flash_bytes = weights_bytes + overhead_code
        sram_bytes = 100

    elif ml_runtime and isinstance(model, (ml_runtime.MiniLinearModel, ml_runtime.MiniSVM)):
        overhead_code = 600
        weights_bytes = (len(model.weights) * data_size) + data_size
        flash_bytes = weights_bytes + overhead_code
        sram_bytes = 40

    else:
        return {
            "flash_percent": 0,
            "sram_percent": 0,
            "error": f"Estimador no implementado para {model_type}"
        }

    # Resultados Seguros
    return {
        "model_type": model_type,
        "language": language,
        "quantized": quantized,
        "weights_bytes": int(weights_bytes),
        "flash_bytes": int(flash_bytes),     
        "flash_total": target_flash,
        "flash_percent": round((flash_bytes / target_flash) * 100, 2),
        "sram_bytes": int(sram_bytes),
        "sram_total": target_sram,
        "sram_percent": round((sram_bytes / target_sram) * 100, 2)
    }

# HELPERS DEEP LEARNING (Shapes & Buffers)
def _infer_output_shape(layer, input_shape):
    l_type = type(layer).__name__
    
    if 'Linear' in l_type:
        # data[0] nos da el número de columnas (n_out)
        w_out = len(layer.weights.data[0])
        return (input_shape[0], w_out) if len(input_shape) >= 2 else (w_out,)
    
    elif 'Conv2d' in l_type and not 'Separable' in l_type and not 'Conv1d' in l_type:
        if len(input_shape) != 4: return input_shape
        b, c, h, w = input_shape
        k = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
        p = layer.padding[0] if isinstance(layer.padding, tuple) else layer.padding
        s = layer.stride
        h_out = (h + 2 * p - k) // s + 1
        w_out = (w + 2 * p - k) // s + 1
        return (b, layer.out_channels, h_out, w_out)
        
    elif 'SeparableConv2d' in l_type:
        return _infer_output_shape(layer.pointwise, input_shape)

    elif 'Conv1d' in l_type:
        if len(input_shape) == 3:
            b, c, t = input_shape
            k = layer.internal.kernel_size[1]
            p = layer.internal.padding[1]
            s = layer.internal.stride
            t_out = (t + 2 * p - k) // s + 1
            return (b, layer.internal.out_channels, t_out)
        return input_shape

    # MaxPool1d
    elif 'MaxPool1d' in l_type:
        if len(input_shape) == 3:
            b, c, l = input_shape
            k = layer.kernel_size[0] if isinstance(layer.kernel_size, (tuple, list)) else layer.kernel_size
            s = layer.stride[0] if isinstance(layer.stride, (tuple, list)) else layer.stride
            l_out = (l - k) // s + 1
            return (b, c, l_out)
        return input_shape

    elif 'MaxPool' in l_type: # MaxPool2d
        if len(input_shape) == 4:
            b, c, h, w = input_shape
            k = layer.kernel_size[0] if isinstance(layer.kernel_size, (tuple, list)) else layer.kernel_size
            s = layer.stride[0] if isinstance(layer.stride, (tuple, list)) else layer.stride
            h_out = (h - k) // s + 1
            w_out = (w - k) // s + 1
            return (b, c, h_out, w_out)
        return input_shape

    elif 'Flatten' in l_type:
        # Se asume que el primer elemento de la forma es el Batch Size
        feature_size = math.prod(input_shape[1:]) if len(input_shape) > 1 else input_shape[0]
        return (input_shape[0], feature_size) if len(input_shape) > 1 else (feature_size,)
        
    elif 'ResidualBlock' in l_type:
        # El bloque residual mantiene la dimensión si stride=1, o la reduce si stride>1.
        # Calculamos la salida de la primera conv interna para saber la verdad.
        return _infer_output_shape(layer.conv1, input_shape)

    return input_shape

def _estimate_temp_buffer(layer, input_shape, data_size):
    """Estima memoria temporal necesaria (ej: Im2Col buffer)."""
    l_type = type(layer).__name__

    if 'ResidualBlock' in l_type: 
        return 256 

    if 'Conv2d' in l_type and not hasattr(layer, 'depthwise'):
        if len(input_shape) == 4: return 256 
    elif 'SeparableConv2d' in l_type:
        return 64
    return 0

# HELPERS ML CLÁSICO
def _count_tree_nodes(node):
    """Cuenta nodos recursivamente."""
    if not isinstance(node, dict): 
        return 1
    if node.get('index') == -1: 
        return 1
    return 1 + _count_tree_nodes(node.get('left')) + _count_tree_nodes(node.get('right'))