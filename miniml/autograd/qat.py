"""
MiniTensor - Módulo de Quantization Aware Training (QAT).
Soporta modos Per-Tensor (Global) y Per-Channel (Granular).
"""

from .tensor import Tensor

class FakeQuant:
    def __init__(self, mode='per_tensor', bit_width=8):
        self.mode = mode
        self.bit_width = bit_width
        self.scale = None
        self.zero_point = 0

    def forward(self, x):
        if not isinstance(x, Tensor):
            return x

        # Calcular Escala (Estadísticas)
        self.calc_scale(x)
        
        # Aplicar Cuantización (Delegar al Tensor)
        return x.fake_quantize(self.scale, self.zero_point, self.mode)

    def calc_scale(self, x):
        """Calcula min/max soportando per_channel (column-wise)."""
        q_max = (2 ** (self.bit_width - 1)) - 1
        
        if self.mode == 'per_channel':
            # Lógica Per-Channel: Una escala por COLUMNA
            # x.data es [[w1, w2], [w3, w4]]
            
            # Transponer para iterar por columnas (Truco Python puro)
            # cols será una lista de tuplas: [(w1, w3), (w2, w4)]
            if not x.data or not isinstance(x.data[0], list):
                # Fallback a per_tensor si es 1D
                self.mode = 'per_tensor' 
                return self.calc_scale(x)

            cols = list(zip(*x.data))
            scales = []

            for col in cols:
                # Calcular max abs para esta columna específica
                min_val = min(col)
                max_val = max(col)
                if abs(max_val - min_val) < 1e-5: max_val += 1e-5
                abs_max = max(abs(min_val), abs(max_val))
                
                scales.append(abs_max / q_max)
            
            # Guardar las escalas como un Tensor vector fila (1, N_Cols)
            self.scale = Tensor([scales])
            
        else:
            # Lógica Per-Tensor (Global): Una escala única
            flat_data = [val for row in x.data for val in row]
            if not flat_data: return

            min_val = min(flat_data)
            max_val = max(flat_data)
            if abs(max_val - min_val) < 1e-5: max_val += 1e-5
            abs_max = max(abs(min_val), abs(max_val))
            
            # Guardar como escalar simple
            self.scale = abs_max / q_max

    def __call__(self, x):
        return self.forward(x)