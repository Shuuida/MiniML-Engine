"""
CMSIS-NN adapter with Fixed-Point support.
Generates C code compatible with arm_fully_connected_s8 and a portable fallback.

Adaptador CMSIS-NN compatible con punto fijo. 
Genera código C compatible con arm_fully_connected_s8 y una alternativa portátil.
"""
from typing import Any, Tuple
import math

class CMSISAdapter:
    def __init__(self, model: Any):
        self.model = model

    def _quantize_multiplier(self, real_multiplier: float) -> Tuple[int, int]:
        if real_multiplier == 0: return 0, 0
        significand, shift = math.frexp(real_multiplier)
        q_mult = int(round(significand * (1 << 31)))
        if q_mult == (1 << 31):
            q_mult //= 2
            shift += 1
        return q_mult, shift

    def generate_c(self, out_path: str):
        # Validación básica
        if not hasattr(self.model, 'q_W1'):
            raise RuntimeError("Model not quantized. Run model.quantize() first.")

        # Obtener tipo de activación (default a sigmoid si no existe)
        hidden_act = getattr(self.model, 'hidden_activation', 'sigmoid').lower()
        output_act = getattr(self.model, 'output_activation', 'sigmoid').lower()

        # Datos
        rows1, cols1 = len(self.model.q_W1), len(self.model.q_W1[0])
        rows2, cols2 = len(self.model.q_W2), len(self.model.q_W2[0])

        # Pre-cálculo Multiplicadores
        mult1_int, shift1_int = zip(*[self._quantize_multiplier(m) for m in self.model.requant_mult1])
        mult2_int, shift2_int = zip(*[self._quantize_multiplier(m) for m in self.model.requant_mult2])

        with open(out_path, 'w', encoding='utf-8') as f:
            f.write('/* CMSIS-NN Adapter Generated Code */\n')
            f.write('#include <stdint.h>\n#include <math.h>\n\n')
            f.write('#define ALIGNED(x) __attribute__((aligned(x)))\n\n')
            f.write(f'// Model Architecture: {cols1}->{rows1} ({hidden_act}) -> {rows2} ({output_act})\n\n')

            # DATA ARRAYS
            def write_array(name, data, type_str):
                f.write(f'const {type_str} {name}[{len(data)}] ALIGNED(4) = {{ ')
                f.write(', '.join(str(x) for x in data))
                f.write(' };\n')

            # Layer 1
            flat_w1 = [w for row in self.model.q_W1 for w in row]
            write_array('W1', flat_w1, 'int8_t')
            write_array('B1', self.model.i32_B1, 'int32_t')
            write_array('MULT1', mult1_int, 'int32_t')
            write_array('SHIFT1', shift1_int, 'int32_t')
            write_array('MULT1_F', self.model.requant_mult1, 'float')

            # Layer 2
            flat_w2 = [w for row in self.model.q_W2 for w in row]
            write_array('W2', flat_w2, 'int8_t')
            write_array('B2', self.model.i32_B2, 'int32_t')
            write_array('MULT2', mult2_int, 'int32_t')
            write_array('SHIFT2', shift2_int, 'int32_t')
            write_array('MULT2_F', self.model.requant_mult2, 'float')
            f.write('\n')

            # CMSIS OPTIMIZED PATH
            f.write('#ifdef CMSISNN_ENABLED\n')
            f.write('#include "arm_nnfunctions.h"\n')
            f.write('static int8_t buf[128]; // Scratch buffer\n')
            f.write('void predict_int8(const int8_t *input, int8_t *output) {\n')
            f.write(f'    int8_t hidden[{rows1}];\n')
            
            # Layer 1 Call
            f.write(f'    arm_fully_connected_s8(input, NULL, {cols1}, {rows1}, 0, 0, -128, W1, B1, hidden, MULT1, SHIFT1, -128, 127, buf);\n')
            if hidden_act == 'relu':
                f.write(f'    arm_relu_q7(hidden, {rows1});\n')
            
            # Layer 2 Call
            f.write(f'    arm_fully_connected_s8(hidden, NULL, {cols2}, {rows2}, 0, 0, -128, W2, B2, output, MULT2, SHIFT2, -128, 127, buf);\n')
            if output_act == 'relu':
                f.write(f'    arm_relu_q7(output, {rows2});\n')
            f.write('}\n')
            
            # FALLBACK PATH
            f.write('#else\n')
            f.write('// Portable Fallback with float requantization\n')
            f.write('static int8_t clamp(int32_t v) { return (v > 127) ? 127 : (v < -128 ? -128 : (int8_t)v); }\n')
            
            f.write('void predict_int8(const int8_t *input, int8_t *output) {\n')
            f.write(f'    int8_t hidden[{rows1}];\n')
            
            # Layer 1 (Hidden)
            f.write(f'    for(int i=0; i<{rows1}; i++) {{\n')
            f.write(f'        int32_t acc = B1[i];\n')
            f.write(f'        for(int j=0; j<{cols1}; j++) acc += (int32_t)W1[i*{cols1}+j] * input[j];\n')
            f.write(f'        int32_t val = (int32_t)roundf(acc * MULT1_F[i]);\n')
            if hidden_act == 'relu':
                f.write(f'        if(val < 0) val = 0; // ReLU\n')
            f.write(f'        hidden[i] = clamp(val);\n')
            f.write('    }\n')

            # Layer 2 (Output)
            f.write(f'    for(int i=0; i<{rows2}; i++) {{\n')
            f.write(f'        int32_t acc = B2[i];\n')
            f.write(f'        for(int j=0; j<{cols2}; j++) acc += (int32_t)W2[i*{cols2}+j] * hidden[j];\n')
            f.write(f'        int32_t val = (int32_t)roundf(acc * MULT2_F[i]);\n')
            if output_act == 'relu':
                f.write(f'        if(val < 0) val = 0; // ReLU\n')
            f.write(f'        output[i] = clamp(val);\n')
            f.write('    }\n')
            f.write('}\n')
            f.write('#endif\n')