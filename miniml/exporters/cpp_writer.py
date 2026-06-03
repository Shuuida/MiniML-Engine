"""
Generador de código C++ (Arduino/ESP32) para MiniML Engine.
Convierte modelos MiniTensor en librerías portables .h optimizadas.

Soporta:
- Arquitectura AVR (Arduino Uno/Mega) usando PROGMEM.
- Arquitectura ESP32/STM32 (Memoria plana).
- Operadores: Conv2d, Linear, MaxPool, ReLU, Flatten.
- Soporte para Modelos Cuantizados INT8 (MiniTensor).
- ResidualBlock1D estable matemáticamente para Arduino.
"""

import math

def _flatten(data_list):
    """Aplanado recursivo de listas."""
    if isinstance(data_list, (list, tuple)):
        if len(data_list) == 0:
            return []
        if not isinstance(data_list[0], (list, tuple)):
            return data_list
        return [item for sublist in data_list for item in _flatten(sublist)]
    return [data_list]

def _get_val(param):
    """Extrae valor entero seguro de int o list/tuple."""
    if isinstance(param, (list, tuple)): return int(param[0])
    return int(param)

def _get_kernel_dims(kernel_size):
    if isinstance(kernel_size, (list, tuple)): return kernel_size[0], kernel_size[1]
    k = int(kernel_size)
    return k, k

def generate_cpp_code(model, input_shape, model_name="MiniMLModel"):
    """
    Genera una librería Single-Header (.h) en C++.
    Soporta:
    1. Modelos Float32 (Clases MiniTensor).
    2. Modelos INT8 (Diccionarios de Quantizer).
    """
    
    # Detección de modo: ¿Es un modelo normal o un diccionario cuantizado?
    is_quantized = isinstance(model, dict)
    
    code = []
    
    guard = f"{model_name.upper()}_H"
    code.append(f"#ifndef {guard}")
    code.append(f"#define {guard}")
    code.append("")
    code.append("#include <Arduino.h>")
    code.append("#include <math.h>")
    code.append("#include <stdint.h>") # Necesario para int8_t
    code.append("")
    
    # Macros de Memoria
    code.append("// Optimizacion de memoria para AVR")
    code.append("#if defined(__AVR__)")
    code.append("  #include <avr/pgmspace.h>")
    code.append("  #define MINITENSOR_MEM PROGMEM")
    code.append("  #define READ_FLOAT(ptr) pgm_read_float(ptr)")
    if is_quantized:
        code.append("  #define READ_BYTE(ptr) ((int8_t)pgm_read_byte(ptr))") # Nuevo para INT8
    code.append("#else")
    code.append("  #define MINITENSOR_MEM")
    code.append("  #define READ_FLOAT(ptr) (*(ptr))")
    if is_quantized:
        code.append("  #define READ_BYTE(ptr) (*(ptr))")
    code.append("#endif")
    code.append("")

    code.append(f"class {model_name} {{")
    code.append("public:")
    
    # Calcular tamaño de entrada plano
    input_flat_size = 1
    for dim in input_shape: input_flat_size *= dim
    
    inference_body = []
    weight_definitions = [] # Almacenamos para escribir al final (static members)
    
    current_var = "input"
    current_shape = list(input_shape)
    current_flat_size = input_flat_size
    
    # PREPARAR ITERADOR DE CAPAS
    # Si es cuantizado, el diccionario tiene claves "layer_0_Conv2d", etc.
    # Si es float, iteramos model.layers
    
    if is_quantized:
        # Filtrar metadatos y ordenar por nombre (layer_0, layer_1...)
        layers_data = []
        for k, v in model.items():
            if k == "__metadata__": continue
            # Extraer índice del nombre "layer_0_Type"
            try:
                idx = int(k.split('_')[1])
                layers_data.append((idx, k, v))
            except: continue
        layers_data.sort(key=lambda x: x[0])
        iterator = [(item[0], item[2]) for item in layers_data] # (idx, data_dict)
    else:
        # Modo Legacy Float
        iterator = enumerate(model.layers)

    # BUCLE DE GENERACIÓN
    for i, layer_obj in iterator:
        
        # Unificar interfaz de acceso a datos
        if is_quantized:
            layer_type = layer_obj['type']
            # En modo cuantizado, 'layer_obj' es el diccionario de datos
            # Los pesos ya vienen aplanados o estructurados en listas
            data_src = layer_obj 
            layer_quantized = layer_obj.get('quantized', False)
        else:
            layer_type = layer_obj.__class__.__name__
            # En modo float, 'layer_obj' es el objeto capa
            data_src = None
            layer_quantized = False

        # > CAPA LINEAR
        if 'Linear' in layer_type:
            if is_quantized:
                # Recuperar shape de los pesos originales (In, Out)
                w_in = len(data_src['weights'])
                w_out = len(data_src['weights'][0])
                w_flat = _flatten(data_src['weights'])
                b_flat = _flatten(data_src['bias']) if data_src['bias'] else [0.0] * w_out
                
                scales = data_src['scale']
                zps = data_src['zero_point']
                scheme = data_src['scheme']
            else:
                # El Tensor original es (In, Out)
                w_in, w_out = layer_obj.weights.shape
                w_flat = _flatten(layer_obj.weights.data)
                b_flat = _flatten(layer_obj.bias.data) if layer_obj.bias else [0.0] * w_out

            w_name = f"W_{i}"
            b_name = f"B_{i}"
            
            # Definición de Arrays
            if layer_quantized:
                weight_definitions.append(f"    static const int8_t {w_name}[{len(w_flat)}] MINITENSOR_MEM;")
                s_name = f"S_{i}"
                z_name = f"Z_{i}"
                weight_definitions.append(f"    static const float {s_name}[{len(scales)}] MINITENSOR_MEM;")
                weight_definitions.append(f"    static const int8_t {z_name}[{len(zps)}] MINITENSOR_MEM;")
            else:
                weight_definitions.append(f"    static const float {w_name}[{len(w_flat)}] MINITENSOR_MEM;")
            
            weight_definitions.append(f"    static const float {b_name}[{len(b_flat)}] MINITENSOR_MEM;")

            # Inferencia
            next_var = f"layer{i}_out"
            inference_body.append(f"        // --- Linear ({w_in} -> {w_out}) [Quant: {layer_quantized}] ---")
            inference_body.append(f"        static float {next_var}[{w_out}];")
            
            loop = f"""        for (int out_c = 0; out_c < {w_out}; out_c++) {{
            float sum = READ_FLOAT(&{b_name}[out_c]);
            """
            
            if layer_quantized:
                if scheme == 'per_channel':
                    loop += f"            float s = READ_FLOAT(&{s_name}[out_c]);\n"
                    loop += f"            int8_t z = READ_BYTE(&{z_name}[out_c]);\n"
                else:
                    loop += f"            float s = READ_FLOAT(&{s_name}[0]);\n"
                    loop += f"            int8_t z = READ_BYTE(&{z_name}[0]);\n"
                    
                loop += f"""            for (int in_c = 0; in_c < {w_in}; in_c++) {{
                int8_t w_raw = READ_BYTE(&{w_name}[in_c * {w_out} + out_c]);
                float w_val = (float)(w_raw - z) * s; // Dequantize
                sum += {current_var}[in_c] * w_val;
            }}"""
            else:
                loop += f"""            for (int in_c = 0; in_c < {w_in}; in_c++) {{
                // Acceso seguro: Fila (in_c) * Columnas Totales (w_out) + Columna Actual (out_c)
                float w_val = READ_FLOAT(&{w_name}[in_c * {w_out} + out_c]);
                sum += {current_var}[in_c] * w_val;
            }}"""
            
            loop += f"\n            {next_var}[out_c] = sum;\n        }}"
            inference_body.append(loop)
            
            current_var = next_var
            current_shape = [w_out]
            current_flat_size = w_out

        # > CAPA CONV2D
        elif 'Conv2d' in layer_type and 'Separable' not in layer_type:
            if is_quantized:
                out_ch = len(data_src['weights'])
                in_ch = len(data_src['weights'][0])
                
                # Fix: Usar kernel_size del metadato (Seguro)
                if 'kernel_size' in data_src:
                    ks = data_src['kernel_size']
                    kh, kw = _get_kernel_dims(ks)
                else:
                    # Fallback
                    kh = len(data_src['weights'][0][0])
                    kw = len(data_src['weights'][0][0][0])
                
                w_flat = _flatten(data_src['weights'])
                b_flat = _flatten(data_src['bias']) if data_src['bias'] else [0.0] * out_ch
                scales = data_src.get('scale', [])
                zps = data_src.get('zero_point', [])
                scheme = data_src.get('scheme', 'per_channel')
                
                stride = _get_val(data_src.get('stride', 1))
                padding = _get_val(data_src.get('padding', 1))

            else:
                # Modo Float Legacy
                out_ch = layer_obj.out_channels
                in_ch = layer_obj.in_channels
                kh, kw = _get_kernel_dims(layer_obj.kernel_size)
                w_flat = _flatten(layer_obj.weights.data)
                b_flat = _flatten(layer_obj.bias.data) if layer_obj.bias else [0.0] * out_ch
                
                stride = _get_val(layer_obj.stride)
                padding = _get_val(layer_obj.padding)

            if len(current_shape) == 3:
                # Caso: Entrada ya tiene forma (C, H, W)
                in_c, in_h, in_w = current_shape
            else:
                # Caso: Entrada plana (Flat). Inferimos geometría.
                # Asumimos que in_c coincide con lo que esperan los pesos (in_ch)
                in_c = in_ch
                side = int(math.sqrt(current_flat_size / in_c))
                in_h, in_w = side, side

            out_h = int((in_h + 2 * padding - kh) / stride + 1)
            out_w = int((in_w + 2 * padding - kw) / stride + 1)
            output_vol = out_ch * out_h * out_w

            w_name, b_name = f"W_{i}", f"B_{i}"
            if layer_quantized:
                weight_definitions.append(f"    static const int8_t {w_name}[{len(w_flat)}] MINITENSOR_MEM;")
                s_name, z_name = f"S_{i}", f"Z_{i}"
                weight_definitions.append(f"    static const float {s_name}[{len(scales)}] MINITENSOR_MEM;")
                weight_definitions.append(f"    static const int8_t {z_name}[{len(zps)}] MINITENSOR_MEM;")
            else:
                weight_definitions.append(f"    static const float {w_name}[{len(w_flat)}] MINITENSOR_MEM;")
            weight_definitions.append(f"    static const float {b_name}[{len(b_flat)}] MINITENSOR_MEM;")

            # INFERENCIA
            next_var = f"layer{i}_out"
            inference_body.append(f"        static float {next_var}[{output_vol}];")

            loop = f"""        for(int och=0; och<{out_ch}; och++) {{
            float bias = READ_FLOAT(&{b_name}[och]);
            """
            if layer_quantized:
                idx = "och" if scheme == 'per_channel' else "0"
                loop += f"            float s = READ_FLOAT(&{s_name}[{idx}]);\n"
                loop += f"            int8_t z = READ_BYTE(&{z_name}[{idx}]);\n"
            
            loop += f"""            for(int oy=0; oy<{out_h}; oy++) {{
                for(int ox=0; ox<{out_w}; ox++) {{
                    float sum = bias;
                    int in_start_y = (oy * {stride}) - {padding};
                    int in_start_x = (ox * {stride}) - {padding};
                    for(int ich=0; ich<{in_c}; ich++) {{
                        for(int ky=0; ky<{kh}; ky++) {{
                            for(int kx=0; kx<{kw}; kx++) {{
                                int iy = in_start_y + ky;
                                int ix = in_start_x + kx;
                                if(iy >= 0 && iy < {in_h} && ix >= 0 && ix < {in_w}) {{
                                    int in_idx = (ich * {in_h} * {in_w}) + (iy * {in_w}) + ix;
                                    int w_idx = (och * {in_ch} * {kh} * {kw}) + (ich * {kh} * {kw}) + (ky * {kw}) + kx;
                                    """
            if layer_quantized:
                loop += f"""int8_t w_raw = READ_BYTE(&{w_name}[w_idx]);
                                    sum += {current_var}[in_idx] * ((float)(w_raw - z) * s);"""
            else:
                loop += f"""sum += {current_var}[in_idx] * READ_FLOAT(&{w_name}[w_idx]);"""
            loop += f"""
                                }}
                            }}
                        }}
                    }}
                    {next_var}[(och * {out_h} * {out_w}) + (oy * {out_w}) + ox] = sum;
                }}
            }}
        }}"""
            inference_body.append(loop)
            current_var = next_var
            current_shape = [out_ch, out_h, out_w]
            current_flat_size = output_vol

        # > CAPA CONV1D
        elif 'Conv1d' in layer_type:
            in_channels = current_shape[-2] if len(current_shape) >= 2 else 1
            in_len = current_shape[-1]
            
            if is_quantized:
                out_channels = data_src.get('out_channels', 1)
                k_size = int(data_src.get('kernel_size', 1))
                stride = int(data_src.get('stride', 1))
                w_flat = _flatten(data_src['weights'])
                b_flat = _flatten(data_src['bias']) if data_src.get('bias') else [0.0] * out_channels
            else:
                out_channels = layer_obj.internal.out_channels if hasattr(layer_obj, 'internal') else layer_obj.weights.shape[0]
                
                # Extracción segura del Kernel y Stride
                raw_k = layer_obj.internal.kernel_size if hasattr(layer_obj, 'internal') else getattr(layer_obj, 'kernel_size', 1)
                k_size = int(raw_k[1] if isinstance(raw_k, tuple) else raw_k)
                
                raw_s = layer_obj.internal.stride if hasattr(layer_obj, 'internal') else getattr(layer_obj, 'stride', 1)
                stride = int(raw_s[1] if isinstance(raw_s, tuple) else raw_s)
                
                w_flat = _flatten(layer_obj.weights.data)
                b_flat = _flatten(layer_obj.bias.data) if layer_obj.bias else [0.0] * out_channels

            out_len = (in_len - k_size) // stride + 1
            
            w_name = f"W_{i}"
            b_name = f"B_{i}"
            
            weight_definitions.append(f"    static const float {w_name}[{len(w_flat)}] MINITENSOR_MEM;")
            weight_definitions.append(f"    static const float {b_name}[{len(b_flat)}] MINITENSOR_MEM;")
            
            next_var = f"layer{i}_out"
            inference_body.append(f"        // --- Conv1D ({in_channels} -> {out_channels}, K:{k_size}, S:{stride}) ---")
            inference_body.append(f"        static float {next_var}[{out_channels * out_len}];")
            
            loop = f"""        for (int out_c = 0; out_c < {out_channels}; out_c++) {{
            for (int l = 0; l < {out_len}; l++) {{
                float sum = READ_FLOAT(&{b_name}[out_c]);
                
                for (int in_c = 0; in_c < {in_channels}; in_c++) {{
                    for (int k = 0; k < {k_size}; k++) {{
                        int in_idx = l * {stride} + k;
                        int w_idx = out_c * ({in_channels} * {k_size}) + in_c * {k_size} + k;
                        sum += {current_var}[in_c * {in_len} + in_idx] * READ_FLOAT(&{w_name}[w_idx]);
                    }}
                }}
                {next_var}[out_c * {out_len} + l] = sum;
            }}
        }}"""
            inference_body.append(loop)
            
            current_var = next_var
            current_shape = [out_channels, out_len]
            current_flat_size = out_channels * out_len

        # > CAPA RELU
        elif 'ReLU' in layer_type:
            inference_body.append(f"        // --- ReLU ---")
            inference_body.append(f"        for(int i=0; i<{current_flat_size}; i++) {{")
            inference_body.append(f"            if({current_var}[i] < 0.0) {current_var}[i] = 0.0;")
            inference_body.append(f"        }}")

        # > CAPA SIGMOID
        elif 'Sigmoid' in layer_type:
            inference_body.append(f"        // --- Sigmoid ---")
            inference_body.append(f"        for(int i=0; i<{current_flat_size}; i++) {{")
            inference_body.append(f"            float val = {current_var}[i];")
            # Clip de seguridad idéntico al de Python para evitar Overflow en C++
            inference_body.append(f"            if(val > 100.0) val = 100.0;")
            inference_body.append(f"            if(val < -100.0) val = -100.0;")
            inference_body.append(f"            {current_var}[i] = 1.0 / (1.0 + exp(-val));")
            inference_body.append(f"        }}")

        # > CAPA FLATTEN
        elif 'Flatten' in layer_type:
            inference_body.append(f"        // --- Flatten ---")
            current_shape = [current_flat_size]

        # > CAPA MAXPOOL 1D
        elif 'MaxPool1d' in layer_type:
            # La forma esperada en 1D es [Canales, Longitud]
            in_channels = current_shape[-2] if len(current_shape) >= 2 else 1
            in_len = current_shape[-1]
            
            # Extraer hiperparámetros del JSON (por defecto kernel=2, stride=kernel)
            if is_quantized:
                k_size = data_src.get('kernel_size', 2)
                stride = data_src.get('stride', k_size)
            else:
                k_size = layer_obj.kernel_size
                stride = layer_obj.stride
            
            # Cálculo de la longitud de salida temporal
            out_len = (in_len - k_size) // stride + 1
            
            next_var = f"layer{i}_out"
            inference_body.append(f"        // --- MaxPool1D (K:{k_size}, S:{stride}) ---")
            inference_body.append(f"        static float {next_var}[{in_channels * out_len}];")
            
            loop = f"""        for (int c = 0; c < {in_channels}; c++) {{
            for (int l = 0; l < {out_len}; l++) {{
                float max_val = -999999.0f; // Evitar usar librerías externas para -Infinito
                for (int k = 0; k < {k_size}; k++) {{
                    int in_idx = l * {stride} + k;
                    if (in_idx < {in_len}) {{
                        // Indexación plana 1D: Canal Actual * Longitud Total + Índice Actual
                        float val = {current_var}[c * {in_len} + in_idx];
                        if (val > max_val) max_val = val;
                    }}
                }}
                {next_var}[c * {out_len} + l] = max_val;
            }}
        }}"""
            inference_body.append(loop)
            
            # Actualizar las variables de estado del generador
            current_var = next_var
            current_shape = [in_channels, out_len]
            current_flat_size = in_channels * out_len

        # > CAPA MAXPOOL
        elif 'MaxPool' in layer_type:
            # Recuperar Hiperparámetros (Soporte Híbrido Dict/Objeto)
            if is_quantized:
                # El Quantizer v3.0 estándar no guarda kernel/stride en capas sin pesos.
                # Intentamos recuperarlo si existe, si no, aplicamos el estándar industrial 2x2.
                k_val = data_src.get('kernel_size', 2) 
                kh, kw = (k_val, k_val) if isinstance(k_val, int) else (k_val[0], k_val[1])
                stride = data_src.get('stride', 2)
            else:
                kh, kw = _get_kernel_dims(layer_obj.kernel_size)
                stride = layer_obj.stride if layer_obj.stride else kh

            # Calcular Geometría de Entrada/Salida
            # Asumimos que current_shape viene correctamente seteado como [C, H, W]
            if len(current_shape) == 3:
                in_c, in_h, in_w = current_shape
            else:
                # Fallback de emergencia si venimos de un Flatten lógico inverso
                side = int(math.sqrt(current_flat_size / layer_obj.in_channels)) if not is_quantized else int(math.sqrt(current_flat_size)) # Riesgoso sin in_channels
                in_c, in_h, in_w = current_shape

            out_h = int((in_h - kh) / stride + 1)
            out_w = int((in_w - kw) / stride + 1)
            output_vol = in_c * out_h * out_w

            # Definir Buffer Estático C++
            next_var = f"layer{i}_out"
            inference_body.append(f"        // --- MaxPool ({kh}x{kw}, stride={stride}) ---")
            inference_body.append(f"        static float {next_var}[{output_vol}];")

            # Generar Algoritmo C++ Optimizado
            # Usamos -FLT_MAX para inicializar, asegurando que funcione con valores negativos (ReLU leak o pre-activación)
            loop = f"""        for(int c=0; c<{in_c}; c++) {{
            for(int oy=0; oy<{out_h}; oy++) {{
                for(int ox=0; ox<{out_w}; ox++) {{
                    float max_val = -3.40282e+38; // -FLT_MAX aprox
                    
                    int start_y = oy * {stride};
                    int start_x = ox * {stride};

                    for(int ky=0; ky<{kh}; ky++) {{
                        for(int kx=0; kx<{kw}; kx++) {{
                            int iy = start_y + ky;
                            int ix = start_x + kx;
                            
                            // Boundary Check
                            if(iy < {in_h} && ix < {in_w}) {{
                                int idx = (c * {in_h} * {in_w}) + (iy * {in_w}) + ix;
                                float val = {current_var}[idx];
                                if(val > max_val) max_val = val;
                            }}
                        }}
                    }}
                    
                    int out_idx = (c * {out_h} * {out_w}) + (oy * {out_w}) + ox;
                    {next_var}[out_idx] = max_val;
                }}
            }}
        }}"""
            inference_body.append(loop)

            # Actualizar Rastreo de Estado
            current_var = next_var
            current_shape = [in_c, out_h, out_w]
            current_flat_size = output_vol

        # > CAPA SEPARABLE CONV 2D (MobileNet Style Edge AI)
        elif 'SeparableConv2d' in layer_type:
            # Dimensiones de entrada 2D [Canales, Alto, Ancho]
            in_channels = current_shape[0] if len(current_shape) > 2 else 1
            in_h = current_shape[1] if len(current_shape) > 2 else current_shape[0]
            in_w = current_shape[2] if len(current_shape) > 2 else current_shape[1]

            # Extracción dinámica y segura de las sub-capas
            dw = layer_obj.depthwise
            pw = layer_obj.pointwise
        
            out_channels = pw.internal.out_channels if hasattr(pw, 'internal') else pw.weights.shape[0]
            
            raw_s = dw.internal.stride if hasattr(dw, 'internal') else getattr(dw, 'stride', 1)
            stride = int(raw_s[0] if isinstance(raw_s, tuple) else raw_s)
            
            raw_k = dw.internal.kernel_size if hasattr(dw, 'internal') else getattr(dw, 'kernel_size', 3)
            k_size = int(raw_k[0] if isinstance(raw_k, tuple) else raw_k)
            
            raw_p = dw.internal.padding if hasattr(dw, 'internal') else getattr(dw, 'padding', None)
            if raw_p is not None:
                pad = int(raw_p[0] if isinstance(raw_p, tuple) else raw_p)
            else:
                pad = (k_size - 1) // 2
            
            # Matemáticas de reducción espacial
            out_h = (in_h + 2 * pad - k_size) // stride + 1
            out_w = (in_w + 2 * pad - k_size) // stride + 1

            wdw_name, bdw_name = f"W_dw_{i}", f"B_dw_{i}"
            wpw_name, bpw_name = f"W_pw_{i}", f"B_pw_{i}"
            
            # Definiciones en Flash (PROGMEM)
            weight_definitions.append(f"    static const float {wdw_name}[{len(_flatten(dw.weights.data))}] MINITENSOR_MEM;")
            weight_definitions.append(f"    static const float {bdw_name}[{in_channels}] MINITENSOR_MEM;")
            weight_definitions.append(f"    static const float {wpw_name}[{len(_flatten(pw.weights.data))}] MINITENSOR_MEM;")
            weight_definitions.append(f"    static const float {bpw_name}[{out_channels}] MINITENSOR_MEM;")

            # Arreglos de SRAM
            next_var = f"layer{i}_out"
            mid_var = f"layer{i}_mid"

            inference_body.append(f"        // --- SeparableConv2D (In:{in_channels}, Out:{out_channels}, K:{k_size}, S:{stride}, P:{pad}) ---")
            inference_body.append(f"        static float {mid_var}[{in_channels * out_h * out_w}];")
            inference_body.append(f"        static float {next_var}[{out_channels * out_h * out_w}];")

            # Bucle C++ PROGMEM Safe
            loop = f"""
        // 1. Depthwise Spatial Convolution (Filtrado Espacial Per-Channel)
        for (int c = 0; c < {in_channels}; c++) {{
            for (int y = 0; y < {out_h}; y++) {{
                for (int x = 0; x < {out_w}; x++) {{
                    float sum = READ_FLOAT(&{bdw_name}[c]);
                    for (int ky = 0; ky < {k_size}; ky++) {{
                        for (int kx = 0; kx < {k_size}; kx++) {{
                            int in_y = y * {stride} - {pad} + ky;
                            int in_x = x * {stride} - {pad} + kx;
                            if (in_y >= 0 && in_y < {in_h} && in_x >= 0 && in_x < {in_w}) {{
                                // Mapeo de tensores 4D a 1D contiguo
                                int w_idx = c * ({k_size} * {k_size}) + ky * {k_size} + kx;
                                int in_idx = c * ({in_h} * {in_w}) + in_y * {in_w} + in_x;
                                sum += {current_var}[in_idx] * READ_FLOAT(&{wdw_name}[w_idx]);
                            }}
                        }}
                    }}
                    {mid_var}[c * ({out_h} * {out_w}) + y * {out_w} + x] = sum;
                }}
            }}
        }}

        // 2. Pointwise 1x1 Convolution (Mezcla de Canales)
        for (int out_c = 0; out_c < {out_channels}; out_c++) {{
            for (int y = 0; y < {out_h}; y++) {{
                for (int x = 0; x < {out_w}; x++) {{
                    float sum = READ_FLOAT(&{bpw_name}[out_c]);
                    for (int in_c = 0; in_c < {in_channels}; in_c++) {{
                        int w_idx = out_c * {in_channels} + in_c;
                        int mid_idx = in_c * ({out_h} * {out_w}) + y * {out_w} + x;
                        sum += {mid_var}[mid_idx] * READ_FLOAT(&{wpw_name}[w_idx]);
                    }}
                    {next_var}[out_c * ({out_h} * {out_w}) + y * {out_w} + x] = sum;
                }}
            }}
        }}
"""
            inference_body.append(loop)
            current_var = next_var
            current_shape = [out_channels, out_h, out_w]
            current_flat_size = out_channels * out_h * out_w

        # > BLOQUE RESIDUAL 1D
        elif 'ResidualBlock1D' in layer_type:
            in_channels = current_shape[-2] if len(current_shape) >= 2 else 1
            in_len = current_shape[-1]

            # [!] EXTRACCIÓN DINÁMICA
            c1 = layer_obj.conv1
            out_channels = c1.internal.out_channels if hasattr(c1, 'internal') else c1.weights.shape[0]
            stride = c1.internal.stride if hasattr(c1, 'internal') else 1
            
            if hasattr(c1, 'internal') and hasattr(c1.internal, 'kernel_size'):
                k_size = c1.internal.kernel_size[1]
            else:
                k_size = 3
                
            if hasattr(c1, 'internal') and hasattr(c1.internal, 'padding'):
                pad = c1.internal.padding[1] if isinstance(c1.internal.padding, tuple) else c1.internal.padding
            else:
                pad = (k_size - 1) // 2
            
            # Cálculo matemático exacto de la salida
            out_len = (in_len + 2 * pad - k_size) // stride + 1

            has_shortcut = len(layer_obj.shortcut.layers) > 0

            # Variables de C++
            w1_name, b1_name = f"W_res1_{i}", f"B_res1_{i}"
            w2_name, b2_name = f"W_res2_{i}", f"B_res2_{i}"
            ws_name, bs_name = f"W_ress_{i}", f"B_ress_{i}"

            # Definiciones de Memoria Flash (PROGMEM)
            weight_definitions.append(f"    static const float {w1_name}[{len(_flatten(layer_obj.conv1.weights.data))}] MINITENSOR_MEM;")
            weight_definitions.append(f"    static const float {b1_name}[{out_channels}] MINITENSOR_MEM;")
            weight_definitions.append(f"    static const float {w2_name}[{len(_flatten(layer_obj.conv2.weights.data))}] MINITENSOR_MEM;")
            weight_definitions.append(f"    static const float {b2_name}[{out_channels}] MINITENSOR_MEM;")
            if has_shortcut:
                weight_definitions.append(f"    static const float {ws_name}[{len(_flatten(layer_obj.shortcut.layers[0].weights.data))}] MINITENSOR_MEM;")
                weight_definitions.append(f"    static const float {bs_name}[{out_channels}] MINITENSOR_MEM;")

            # Arreglos de SRAM
            next_var = f"layer{i}_out"
            tmp1_var = f"layer{i}_t1"
            tmp2_var = f"layer{i}_t2"
            short_var = f"layer{i}_sh"

            inference_body.append(f"        // --- ResidualBlock1D (In:{in_channels}, Out:{out_channels}, K:{k_size}, P:{pad}) ---")
            inference_body.append(f"        static float {tmp1_var}[{out_channels * out_len}];")
            inference_body.append(f"        static float {tmp2_var}[{out_channels * out_len}];")
            if has_shortcut:
                inference_body.append(f"        static float {short_var}[{out_channels * out_len}];")
            inference_body.append(f"        static float {next_var}[{out_channels * out_len}];")

            # C++ Code Loop (Operator Fusion) - PROGMEM & Dynamic Math Safe
            loop = f"""
        // 1. Conv1 + ReLU
        for (int out_c = 0; out_c < {out_channels}; out_c++) {{
            for (int l = 0; l < {out_len}; l++) {{
                float sum = READ_FLOAT(&{b1_name}[out_c]);
                for (int in_c = 0; in_c < {in_channels}; in_c++) {{
                    for (int k = 0; k < {k_size}; k++) {{
                        int in_idx = l * {stride} - {pad} + k; // Dynamic Padding
                        if (in_idx >= 0 && in_idx < {in_len}) {{
                            int w_idx = out_c * ({in_channels} * {k_size}) + in_c * {k_size} + k;
                            sum += {current_var}[in_c * {in_len} + in_idx] * READ_FLOAT(&{w1_name}[w_idx]);
                        }}
                    }}
                }}
                if (sum < 0.0f) sum = 0.0f; // ReLU
                {tmp1_var}[out_c * {out_len} + l] = sum;
            }}
        }}

        // 2. Conv2
        for (int out_c = 0; out_c < {out_channels}; out_c++) {{
            for (int l = 0; l < {out_len}; l++) {{
                float sum = READ_FLOAT(&{b2_name}[out_c]);
                for (int in_c = 0; in_c < {out_channels}; in_c++) {{
                    for (int k = 0; k < {k_size}; k++) {{
                        int in_idx = l * 1 - {pad} + k; // Stride=1, Dynamic Padding
                        if (in_idx >= 0 && in_idx < {out_len}) {{
                            int w_idx = out_c * ({out_channels} * {k_size}) + in_c * {k_size} + k;
                            sum += {tmp1_var}[in_c * {out_len} + in_idx] * READ_FLOAT(&{w2_name}[w_idx]);
                        }}
                    }}
                }}
                {tmp2_var}[out_c * {out_len} + l] = sum;
            }}
        }}
"""
            if has_shortcut:
                loop += f"""
        // 3. Shortcut (1x1 Conv)
        for (int out_c = 0; out_c < {out_channels}; out_c++) {{
            for (int l = 0; l < {out_len}; l++) {{
                float sum = READ_FLOAT(&{bs_name}[out_c]);
                for (int in_c = 0; in_c < {in_channels}; in_c++) {{
                    int in_idx = l * {stride}; // Pad=0
                    if (in_idx < {in_len}) {{
                        int w_idx = out_c * {in_channels} + in_c;
                        sum += {current_var}[in_c * {in_len} + in_idx] * READ_FLOAT(&{ws_name}[w_idx]);
                    }}
                }}
                {short_var}[out_c * {out_len} + l] = sum;
            }}
        }}
"""
            
            loop += f"""
        // 4. Skip Connection + ReLU Final
        for (int out_c = 0; out_c < {out_channels}; out_c++) {{
            for (int l = 0; l < {out_len}; l++) {{
                float val = {tmp2_var}[out_c * {out_len} + l];"""
            
            if has_shortcut:
                loop += f"\n                float id_val = {short_var}[out_c * {out_len} + l];"
            else:
                loop += f"\n                float id_val = {current_var}[out_c * {out_len} + l];"
            
            loop += f"""
                val += id_val;
                if (val < 0.0f) val = 0.0f; // Final ReLU
                {next_var}[out_c * {out_len} + l] = val;
            }}
        }}
"""
            inference_body.append(loop)
            
            current_var = next_var
            current_shape = [out_channels, out_len]
            current_flat_size = out_channels * out_len

    # FINALIZAR CLASE
    code.extend(weight_definitions)

    code.append(f"    // Prediccion")
    code.append(f"    float* predict(float* input) {{")
    code.extend(inference_body)
    code.append(f"        return {current_var};")
    code.append(f"    }}")
    code.append("};") # Fin Class

    # ESCRIBIR DEFINICIONES ESTÁTICAS AL FINAL (.cpp style inside .h for header-only)
    code.append("")
    code.append("// --- Definicion de Pesos Estaticos ---")
    
    # Reiniciar iterador para escritura de datos
    if is_quantized:
        iterator = [(item[0], item[2]) for item in layers_data]
    else:
        iterator = enumerate(model.layers)

    for i, layer_obj in iterator:
        if is_quantized:
            layer_type = layer_obj['type']
            data_src = layer_obj
            layer_quantized = layer_obj.get('quantized', False)
        else:
            layer_type = layer_obj.__class__.__name__
            layer_quantized = False
            
        if 'Linear' in layer_type or ('Conv2d' in layer_type and 'Separable' not in layer_type) or 'Conv1d' in layer_type or 'ResidualBlock1D' in layer_type or 'SeparableConv2d' in layer_type:
            # Refrescar los nombres en cada iteración
            w_name = f"W_{i}"
            b_name = f"B_{i}"
            s_name = f"S_{i}"
            z_name = f"Z_{i}"

            # Obtener datos planos
            if is_quantized:
                w_flat = _flatten(data_src['weights'])
                b_flat = _flatten(data_src['bias']) if data_src.get('bias') else [0.0]
                if layer_quantized:
                    scales = _flatten(data_src['scale'])
                    zps = _flatten(data_src['zero_point'])
            else:
                if 'ResidualBlock1D' in layer_type or 'SeparableConv2d' in layer_type:
                    pass
                else:
                    w_flat = _flatten(layer_obj.weights.data)
                    b_flat = _flatten(layer_obj.bias.data) if layer_obj.bias else [0.0]

            if 'SeparableConv2d' in layer_type:
                dw = layer_obj.depthwise
                pw = layer_obj.pointwise
                
                wdw_flat = _flatten(dw.weights.data)
                bdw_flat = _flatten(dw.bias.data) if getattr(dw, 'bias', None) else [0.0] * dw.weights.shape[0]
                wpw_flat = _flatten(pw.weights.data)
                bpw_flat = _flatten(pw.bias.data) if getattr(pw, 'bias', None) else [0.0] * pw.weights.shape[0]
                
                # Escribir Depthwise
                wdw_str = ", ".join(f"{x:.6f}" for x in wdw_flat)
                bdw_str = ", ".join(f"{x:.6f}" for x in bdw_flat)
                code.append(f"const float {model_name}::W_dw_{i}[] MINITENSOR_MEM = {{ {wdw_str} }};")
                code.append(f"const float {model_name}::B_dw_{i}[] MINITENSOR_MEM = {{ {bdw_str} }};")
                
                # Escribir Pointwise
                wpw_str = ", ".join(f"{x:.6f}" for x in wpw_flat)
                bpw_str = ", ".join(f"{x:.6f}" for x in bpw_flat)
                code.append(f"const float {model_name}::W_pw_{i}[] MINITENSOR_MEM = {{ {wpw_str} }};")
                code.append(f"const float {model_name}::B_pw_{i}[] MINITENSOR_MEM = {{ {bpw_str} }};")
                
                continue

            if 'ResidualBlock1D' in layer_type:
                c1 = layer_obj.conv1
                c2 = layer_obj.conv2
                
                w1_flat = _flatten(c1.weights.data)
                b1_flat = _flatten(c1.bias.data) if c1.bias else [0.0] * c1.weights.shape[0]
                w2_flat = _flatten(c2.weights.data)
                b2_flat = _flatten(c2.bias.data) if c2.bias else [0.0] * c2.weights.shape[0]

                w1_str = ", ".join(f"{x:.6f}" for x in w1_flat)
                b1_str = ", ".join(f"{x:.6f}" for x in b1_flat)
                code.append(f"const float {model_name}::W_res1_{i}[] MINITENSOR_MEM = {{ {w1_str} }};")
                code.append(f"const float {model_name}::B_res1_{i}[] MINITENSOR_MEM = {{ {b1_str} }};")
                
                w2_str = ", ".join(f"{x:.6f}" for x in w2_flat)
                b2_str = ", ".join(f"{x:.6f}" for x in b2_flat)
                code.append(f"const float {model_name}::W_res2_{i}[] MINITENSOR_MEM = {{ {w2_str} }};")
                code.append(f"const float {model_name}::B_res2_{i}[] MINITENSOR_MEM = {{ {b2_str} }};")
                
                # Escribir Shortcut (Si existe)
                if len(layer_obj.shortcut.layers) > 0:
                    cs = layer_obj.shortcut.layers[0]
                    ws_flat = _flatten(cs.weights.data)
                    bs_flat = _flatten(cs.bias.data) if cs.bias else [0.0] * cs.weights.shape[0]
                    ws_str = ", ".join(f"{x:.6f}" for x in ws_flat)
                    bs_str = ", ".join(f"{x:.6f}" for x in bs_flat)
                    code.append(f"const float {model_name}::W_ress_{i}[] MINITENSOR_MEM = {{ {ws_str} }};")
                    code.append(f"const float {model_name}::B_ress_{i}[] MINITENSOR_MEM = {{ {bs_str} }};")
                    
                continue

            if layer_quantized:
                # Escribir array INT8
                w_str = ", ".join(str(int(x)) for x in w_flat)
                code.append(f"const int8_t {model_name}::{w_name}[] MINITENSOR_MEM = {{ {w_str} }};")
                
                # Escribir Scales/ZP
                s_name = f"S_{i}"
                z_name = f"Z_{i}"
                s_str = ", ".join(f"{x:.8f}" for x in scales)
                z_str = ", ".join(str(int(x)) for x in zps)
                
                code.append(f"const float {model_name}::{s_name}[] MINITENSOR_MEM = {{ {s_str} }};")
                code.append(f"const int8_t {model_name}::{z_name}[] MINITENSOR_MEM = {{ {z_str} }};")
            else:
                w_str = ", ".join(f"{x:.6f}" for x in w_flat)
                code.append(f"const float {model_name}::{w_name}[] MINITENSOR_MEM = {{ {w_str} }};")

            # Bias
            b_str = ", ".join(f"{x:.6f}" for x in b_flat)
            code.append(f"const float {model_name}::{b_name}[] MINITENSOR_MEM = {{ {b_str} }};")

    code.append("")
    code.append(f"#endif // {guard}")
    
    return "\n".join(code)