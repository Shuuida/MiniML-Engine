"""
Quantizer Module for MiniTensor
====================================
Módulo de cuantización Post-Entrenamiento (PTQ) que actúa como adaptador
cerebral para los exportadores C++ y Rust.

Características:
- Convierte pesos a INT8 (ahorro 75% RAM/Flash).
- Captura hiperparámetros estructurales (Stride, Padding, Kernel).
- Genera metadatos de tipos para C++ (int8_t) y Rust (i8).
- Soporta estrategias Per-Tensor y Per-Channel.
"""

#import math

try:
    from miniml.autograd import layers as nn
    from miniml.autograd import tensor
    from miniml.exporters import rust_writer, cpp_writer
except ImportError:
    tensor = None
    nn = None
    ml_runtime = None

class Quantizer:
    def __init__(self, model, calibration_data=None, qmin=-128, qmax=127):
        self.model = model
        self.calibration_data = calibration_data
        self.qmin = qmin
        self.qmax = qmax

    def _get_min_max(self, data):
        """Obtiene min y max de una estructura de datos recursiva."""
        if hasattr(data, 'data'): data = data.data
        def _recursive_min_max(d):
            if isinstance(d, list):
                if not d: return float('inf'), float('-inf')
                mins, maxs = zip(*[_recursive_min_max(x) for x in d])
                return min(mins), max(maxs)
            return d, d
        return _recursive_min_max(data)

    def _calculate_scale_zp(self, min_val, max_val):
        """Calcula Scale y ZeroPoint."""
        if min_val == max_val: return 1.0, 0
        min_val = min(min_val, 0.0)
        max_val = max(max_val, 0.0)
        scale = (max_val - min_val) / (self.qmax - self.qmin)
        if scale == 0: scale = 1.0
        zero_point = self.qmin - round(min_val / scale)
        zero_point = max(self.qmin, min(self.qmax, zero_point))
        return scale, int(zero_point)

    def _quantize_structure(self, data, scale, zp):
        """Cuantiza recursivamente."""
        if isinstance(data, list):
            return [self._quantize_structure(x, scale, zp) for x in data]
        # Fórmula PTQ: Clamp(Round(x / scale) + zp)
        q = round(data / scale) + zp
        return max(self.qmin, min(self.qmax, int(q)))
    
    def _extract_geometry(self, layer):
        """Extrae hiperparámetros geométricos (Stride, Padding, Kernel) de forma segura."""
        geo = {}
        # Atributos comunes en Conv y Pool
        for attr in ['stride', 'padding', 'dilation']:
            if hasattr(layer, attr):
                val = getattr(layer, attr)
                # Normalizar tuplas a listas o enteros para JSON/Dict compatibility
                if isinstance(val, (tuple, list)):
                    if len(val) == 2 and val[0] == val[1]:
                        val = val[0]
                    else:
                        val = list(val)
                geo[attr] = val
        
        # Normalizar Kernel Size
        if hasattr(layer, 'kernel_size'):
            k = layer.kernel_size
            if isinstance(k, (tuple, list)):
                if len(k) == 2 and k[0] == k[1]:
                    k = k[0]
                else:
                    k = list(k)
            geo['kernel_size'] = k
            
        return geo

    def quantize(self):
        """
        Genera un diccionario completo para exportadores C++ y Rust.
        Incluye 'typenames' para facilitar la generación de código.
        """
        # Generar datos cuantizados
        if nn and (isinstance(self.model, nn.Module) or isinstance(self.model, nn.Sequential)):
            data = self._quantize_deep_learning()
        elif ml_runtime and isinstance(self.model, ml_runtime.MiniNeuralNetwork):
            data = self._quantize_mlp_legacy()
        else:
            data = {"info": "Modelo no soportado para INT8"}

        # Agregar soporte explícito para Exportadores (Tipos de datos)
        data["__metadata__"] = {
            "cpp_types": {
                "weight_type": "int8_t",
                "scale_type": "float",
                "zp_type": "int8_t",
                "tensor_format": "{...}" 
            },
            "rust_types": {
                "weight_type": "i8",
                "scale_type": "f32",
                "zp_type": "i8",
                "tensor_format": "[...]"
            }
        }
        return data

    def _quantize_deep_learning(self):
        export_data = {}
        layers = self.model.layers if hasattr(self.model, 'layers') else [self.model]
        
        for i, layer in enumerate(layers):
            l_type = type(layer).__name__
            layer_id = f"layer_{i}_{l_type}"
            
            # CAPAS CON PESOS (Conv2d, Linear)
            if hasattr(layer, 'weights'):
                weights = layer.weights.data
                
                # Datos base del layer
                layer_dict = {
                    "type": l_type,
                    "quantized": True,
                    "bias": layer.bias.data if hasattr(layer, 'bias') else None
                }
                
                # Inyectar Geometría (Stride, Padding...)
                layer_dict.update(self._extract_geometry(layer))
                
                # ESTRATEGIA PER-CHANNEL (Conv2d, Linear)
                if 'Conv2d' in l_type or 'Linear' in l_type:
                    num_channels = len(weights)
                    q_weights = []
                    scales = []
                    zps = []
                    
                    for ch in range(num_channels):
                        c_min, c_max = self._get_min_max(weights[ch])
                        s, z = self._calculate_scale_zp(c_min, c_max)
                        q_weights.append(self._quantize_structure(weights[ch], s, z))
                        scales.append(s)
                        zps.append(z)
                        
                    layer_dict["scheme"] = "per_channel"
                    layer_dict["weights"] = q_weights
                    layer_dict["scale"] = scales
                    layer_dict["zero_point"] = zps
                    
                # ESTRATEGIA PER-TENSOR (Fallback)
                else:
                    g_min, g_max = self._get_min_max(weights)
                    s, z = self._calculate_scale_zp(g_min, g_max)
                    
                    layer_dict["scheme"] = "per_tensor"
                    layer_dict["weights"] = self._quantize_structure(weights, s, z)
                    layer_dict["scale"] = [s]
                    layer_dict["zero_point"] = [z]
                
                export_data[layer_id] = layer_dict
            
            # CAPAS SIN PESOS (MaxPool, Flatten, ReLU)
            else:
                # Aún necesitamos capturar la geometría para MaxPool
                layer_dict = {
                    "type": l_type,
                    "quantized": False
                }
                # Si es MaxPool, AvgPool, etc, extraer kernel/stride
                if 'Pool' in l_type:
                    layer_dict.update(self._extract_geometry(layer))
                
                export_data[layer_id] = layer_dict
                
        return export_data

    def _quantize_mlp_legacy(self):
        min1, max1 = self._get_min_max(self.model.W1)
        s1, z1 = self._calculate_scale_zp(min1, max1)
        
        min2, max2 = self._get_min_max(self.model.W2)
        s2, z2 = self._calculate_scale_zp(min2, max2)
        
        return {
            "mlp": {
                "type": "MiniNeuralNetwork",
                "quantized": True,
                "W1": {"data": self._quantize_structure(self.model.W1, s1, z1), "scale": s1, "zp": z1},
                "W2": {"data": self._quantize_structure(self.model.W2, s2, z2), "scale": s2, "zp": z2},
                "B1": self.model.B1,
                "B2": self.model.B2
            }
        }