"""
MiniTensor Engine (Zero-Dependency Edition)
===========================================
Motor tensorial ligero escrito en Python puro.
ACTUALIZADO v1.6: Soporte Recursivo para N-Dimensiones (CNNs 4D).
"""

import math
import random

class Tensor:
    def __init__(self, data, _children=(), _op='', label='', requires_grad=False):
        if isinstance(data, Tensor):
            data = data.data
            
        self.data = self._ensure_list(data)
        self.grad = None 
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.requires_grad = requires_grad or any(c.requires_grad for c in _children)
        #self.shape = self._get_shape(self.data)

    # Utilidades Internas (Recursivas)

    def _ensure_list(self, data):
        """Asegura que todo sea una lista, manejo básico."""
        if isinstance(data, (int, float)):
            return [[float(data)]]
        if isinstance(data, list):
            if not data: return []
            # Si es lista plana [1,2] -> [[1,2]] (Solo para compatibilidad MLP legacy)
            if len(data) > 0 and not isinstance(data[0], list):
                return [data]
            return data
        return data

    def _get_shape(self, data):
        """Obtiene shape recursivo (Tupla N-Dimensional)."""
        shape = []
        d = data
        while isinstance(d, list) and len(d) > 0:
            shape.append(len(d))
            d = d[0]
        return tuple(shape)

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op='{self._op}')"

    # Operaciones Matemáticas (Forward)

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        # Forward con Broadcasting Recursivo
        def _recursive_add(a, b):
            if isinstance(a, list) and isinstance(b, list):
                # Caso Bias: Si 'b' tiene 1 fila y 'a' muchas, repetimos 'b'
                if len(b) == 1 and len(a) > 1:
                     return [_recursive_add(x, b[0]) for x in a]
                # Caso Normal: Elemento a elemento
                return [_recursive_add(x, y) for x, y in zip(a, b)]
            # Caso Base: Suma numérica
            return a + b

        out_data = _recursive_add(self.data, other.data)
        out = Tensor(out_data, (self, other), '+')

        # Backward con Corrección de Broadcasting
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad
            
            if other.requires_grad:
                other._init_grad()
                
                # Detección de Broadcasting
                # Si 'other' es un vector fila (Bias) y el gradiente es un Batch...
                if len(other.data) == 1 and len(out.grad.data) > 1:
                    # Debemos colapsar (sumar) todas las filas del gradiente en una sola
                    grad_batch = out.grad.data
                    
                    # Helper para sumar estructuras anidadas (listas de listas)
                    def _deep_add_accum(acc, val):
                        if isinstance(acc, list):
                            return [_deep_add_accum(a, v) for a, v in zip(acc, val)]
                        return acc + val

                    # Inicializamos el acumulador con ceros de la forma de UNA fila
                    accum = Tensor.zeros_like(grad_batch[0])
                    
                    # Sumamos todo el batch
                    for row in grad_batch:
                        accum = _deep_add_accum(accum, row)
                    
                    # Asignamos el resultado comprimido (envuelto en lista de 1 fila)
                    other.grad += Tensor([accum])
                else:
                    # Caso normal (Shapes coinciden)
                    other.grad += out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        """
        Implementa la negación (-a).
        Vital para convertir la resta en una suma: a - b = a + (-b).
        """
        return self * -1

    def __sub__(self, other):
        """
        Implementa la resta (a - b).
        Reutiliza la arquitectura de __add__ para garantizar
        que el broadcasting (bias) y la recursividad N-Dimensional funcionen
        exactamente igual que en la suma.
        """
        # Asegurar que 'other' sea un Tensor (igual que en __add__)
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        # Suma con el inverso aditivo
        # Esto invoca __add__, activando automáticamente:
        # - _recursive_add (Soporte recursivo)
        # - Lógica de colapso de gradientes para vectores Bias
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        # Helper específico para multiplicar una estructura por un flotante puro
        def _scalar_mul_structure(struct, val):
            if isinstance(struct, list):
                return [_scalar_mul_structure(x, val) for x in struct]
            return struct * val

        def _recursive_mul(a, b):
            # CASO A: 'b' es un Tensor Escalar 1x1 (ej: LR o Momentum)
            # Detectamos si b es [[val]]
            if isinstance(b, list) and len(b) == 1 and isinstance(b[0], list) and len(b[0]) == 1:
                val = b[0][0]
                # Multiplicamos 'a' por el float 'val' directamente, sin más recursión de listas
                return _scalar_mul_structure(a, val)

            # CASO B: 'a' es un Tensor Escalar (Inverso)
            if isinstance(a, list) and len(a) == 1 and isinstance(a[0], list) and len(a[0]) == 1:
                val = a[0][0]
                return _scalar_mul_structure(b, val)

            # CASO C: Multiplicación Elemento a Elemento (Zip)
            if isinstance(a, list) and isinstance(b, list):
                return [_recursive_mul(x, y) for x, y in zip(a, b)]
            
            # CASO D: Multiplicación Escalar Simple (Fallback)
            elif isinstance(a, list): 
                return [_scalar_mul_structure(a, b)] # b es float
            elif isinstance(b, list): 
                return [_scalar_mul_structure(b, a)] # a es float
            
            # Caso Base Numérico
            return a * b

        out_data = _recursive_mul(self.data, other.data)
        out = Tensor(out_data, (self, other), '*')

        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += other * out.grad
            if other.requires_grad:
                other._init_grad()
                
                # Cálculo de gradiente para 'other'
                self_times_grad = self * out.grad
                
                # Fix broadcasting en backward para el escalar
                # Si 'other' es escalar (1x1) y el resultado es matriz, sumamos todo
                if len(other.data) == 1 and len(other.data[0]) == 1 and len(self_times_grad.data) > 1:
                     other.grad += self_times_grad.sum()
                else:
                     other.grad += self_times_grad
                     
        out._backward = _backward
        return out

    def matmul(self, other):
        """MatMul sigue siendo estricto 2D para las capas Linear/Conv-Im2Col."""
        other = other if isinstance(other, Tensor) else Tensor(other)
        
        # Validación simple 2D
        if len(self.shape) != 2 or len(other.shape) != 2:
             # Si no son 2D, asumimos que son tensores aplanados o fallamos
             # Para CNNs, im2col ya devuelve 2D, así que esto está bien.
             pass

        rows_a = len(self.data)
        cols_a = len(self.data[0])
        cols_b = len(other.data[0])

        other_T = list(zip(*other.data)) 
        
        out_data = []
        for i in range(rows_a):
            row_result = []
            row_a = self.data[i]
            for j in range(cols_b):
                col_b = other_T[j]
                dot = sum(a * b for a, b in zip(row_a, col_b))
                row_result.append(dot)
            out_data.append(row_result)

        out = Tensor(out_data, (self, other), 'matmul')

        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad.matmul(other.T())
            if other.requires_grad:
                other._init_grad()
                other.grad += self.T().matmul(out.grad)
                
        out._backward = _backward
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def T(self):
        # Transpuesta solo para 2D
        transposed = list(map(list, zip(*self.data)))
        out = Tensor(transposed, (self,), 'T')
        def _backward():
            if self.requires_grad:
                self._init_grad()
                self.grad += out.grad.T()
        out._backward = _backward
        return out

    def sum(self):
        """Suma recursiva total (corrige error int + list)."""
        def _deep_sum(data):
            if isinstance(data, list):
                return sum(_deep_sum(x) for x in data)
            return data

        total = _deep_sum(self.data)
        out = Tensor([total], (self,), 'sum') # Envolver en lista para consistencia
        
        def _backward():
            if self.requires_grad:
                self._init_grad()
                
                # Extraer el valor escalar float
                # out.grad.data suele venir como [[val]], necesitamos 'val' limpio
                g = out.grad.data
                while isinstance(g, list):
                    if not g: # Protección lista vacía
                        g = 0.0 
                        break
                    g = g[0]
                grad_val = float(g)
                
                # Crear estructura de unos con la misma forma que los datos originales
                ones_data = Tensor.ones_like(self.data)
                
                # Función auxiliar para multiplicar la estructura de unos por el gradiente escalar
                def _mul_recursive(struct, val):
                    if isinstance(struct, list):
                        return [_mul_recursive(x, val) for x in struct]
                    return struct * val
                
                # El gradiente de una suma se distribuye igual a todos los elementos (1 * grad)
                grad_distributed = _mul_recursive(ones_data, grad_val)
                self.grad += Tensor(grad_distributed)
                
        out._backward = _backward
        return out

    def relu(self):
        """ReLU recursivo (corrige error > list)."""
        def _deep_relu(data):
            if isinstance(data, list):
                return [_deep_relu(x) for x in data]
            return max(0.0, data)

        out_data = _deep_relu(self.data)
        out = Tensor(out_data, (self,), 'ReLU')
        
        def _backward():
            if self.requires_grad:
                self._init_grad()
                
                def _deep_relu_grad(d):
                    if isinstance(d, list):
                        return [_deep_relu_grad(x) for x in d]
                    return 1.0 if d > 0 else 0.0
                
                grad_mask = Tensor(_deep_relu_grad(self.data))
                self.grad += grad_mask * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        """Sigmoid recursivo con Autograd (Estable numéricamente)."""
        def _deep_sigmoid(data):
            if isinstance(data, list):
                return [_deep_sigmoid(x) for x in data]
            # Clip para evitar desbordamiento (Overflow) en math.exp()
            val = max(-100.0, min(100.0, data))
            return 1.0 / (1.0 + math.exp(-val))

        out_data = _deep_sigmoid(self.data)
        out = Tensor(out_data, (self,), 'Sigmoid')
        
        def _backward():
            if self.requires_grad:
                self._init_grad()
                
                # Derivada de Sigmoid: S(x) * (1 - S(x))
                def _deep_sigmoid_grad(o):
                    if isinstance(o, list):
                        return [_deep_sigmoid_grad(x) for x in o]
                    return o * (1.0 - o)
                
                # Aprovechamos la sobrecarga de multiplicación * del Tensor
                grad_mask = Tensor(_deep_sigmoid_grad(out.data))
                self.grad += grad_mask * out.grad
                
        out._backward = _backward
        return out

    # Soporte QAT

    def fake_quantize(self, scale, zero_point=0, mode='per_tensor', qmin=-128, qmax=127):
        """
        Simula cuantización INT8. Soporta Per-Tensor (Global) y Per-Channel (Granular).
        Detecta automáticamente si es Linear (2D) o CNN (4D) para aplicar las escalas.
        """
        # Normalizar Escalas (puede ser Tensor, lista o escalar)
        if isinstance(scale, Tensor):
            scale_data = scale.data
        elif isinstance(scale, list):
            scale_data = scale
        else:
            scale_data = [scale]
            
        # Aplanar para obtener lista simple de floats [s1, s2...]
        def _flatten_scales(s):
            if isinstance(s, list):
                if not s: return [1.0]
                if isinstance(s[0], list): return _flatten_scales(s[0])
                return s
            return [s]
            
        scales = _flatten_scales(scale_data)
        is_per_channel = (mode == 'per_channel') and (len(scales) > 1)
        
        # Función Core de Cuantización
        def _quant_op(val, s):
            if s == 0: s = 1e-9
            x_q = round(val / s + zero_point)
            x_q = max(qmin, min(x_q, qmax))
            return (x_q - zero_point) * s

        out_data = []

        # Lógica de Aplicación según Dimensión
        
        # CNN 4D [Batch, Channel, Height, Width] -> Escala por Canal
        if is_per_channel and len(self.shape) == 4:
            B, C, H, W = self.shape
            # Protección: Si las escalas no coinciden con los canales, fallback a global
            if len(scales) != C:
                s_val = scales[0]
                is_per_channel = False
            else:
                for b in range(B):
                    batch_data = []
                    for c in range(C):
                        s = scales[c] # Escala específica para este canal
                        channel_data = []
                        for h in range(H):
                            # Aplicamos 's' a toda la fila
                            row_data = [_quant_op(val, s) for val in self.data[b][c][h]]
                            channel_data.append(row_data)
                        batch_data.append(channel_data)
                    out_data.append(batch_data)

        # Linear 2D [Batch, Features] -> Escala por Feature (Columna)
        elif is_per_channel and len(self.shape) == 2:
            B, F = self.shape
            if len(scales) != F:
                s_val = scales[0]
                is_per_channel = False
            else:
                for b in range(B):
                    row_data = []
                    for f in range(F):
                        # Aplicamos scales[f] a la columna f
                        row_data.append(_quant_op(self.data[b][f], scales[f]))
                    out_data.append(row_data)

        # Default / Per-Tensor (Recursivo Global)
        # Se ejecuta si no es per-channel O si hubo fallback
        if not out_data: 
            s_val = scales[0]
            def _deep_quant(data):
                if isinstance(data, list):
                    return [_deep_quant(x) for x in data]
                return _quant_op(data, s_val)
            out_data = _deep_quant(self.data)
            
        return Tensor(out_data, (self,), 'QAT')

    # Gestión de Gradientes

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        if self.grad is None:
            self.grad = Tensor([[1.0]])
            
        for node in reversed(topo):
            node._backward()

    def _init_grad(self):
        """
        Inicializa gradiente con la MISMA estructura profunda que data.
        Corrige el error de 'float object not subscriptable' en Flatten.
        """
        if self.grad is None:
            self.grad = Tensor(Tensor.zeros_like(self.data))

    def zero_grad(self):
        self.grad = None

    def reshape(self, new_shape):
        """
        Cambia la forma del Tensor sin alterar los datos subyacentes.
        Soporta Autograd: El gradiente fluye de regreso a la forma original.
        
        Args:
            new_shape: Tupla o Lista con las nuevas dimensiones (ej: (1, 64, 1, 10))
                       Soporta dimensión -1 (inferencia automática).
        """
        # Aplanar datos actuales (Flatten)
        flat_data = self._flatten_data(self.data)
        total_elements = len(flat_data)
        
        # Calcular/Validar nueva forma
        target_shape = list(new_shape)
        if -1 in target_shape:
            # Calcular la dimensión desconocida
            known_size = 1
            idx_neg = -1
            for i, s in enumerate(target_shape):
                if s == -1:
                    idx_neg = i
                else:
                    known_size *= s
            
            if known_size == 0 or total_elements % known_size != 0:
                raise ValueError(f"No se puede remodelar tamaño {total_elements} a {new_shape}")
                
            target_shape[idx_neg] = total_elements // known_size
        else:
            # Validar tamaño total
            target_size = 1
            for s in target_shape: target_size *= s
            if target_size != total_elements:
                raise ValueError(f"Tamaño incompatible: {total_elements} vs {target_size}")

        # Reconstruir estructura anidada (Unflatten)
        new_data = self._unflatten_data(flat_data, target_shape)
        
        # Crear nuevo Tensor conectado al grafo
        out = Tensor(new_data, (self,), f"Reshape{tuple(target_shape)}")
        
        # Definir Backward
        def _backward():
            if self.requires_grad:
                self._init_grad()
                # El gradiente viene con la 'new_shape', lo regresamos a 'self.shape'
                # Simplemente llamamos a reshape recursivamente sobre el gradiente
                grad_reshaped = out.grad.reshape(self.shape)
                self.grad += grad_reshaped

        out._backward = _backward
        return out

    def unsqueeze(self, dim):
        """
        Inserta una dimensión de tamaño 1 en la posición especificada.
        Ej: (B, C, T) -> unsqueeze(2) -> (B, C, 1, T)
        """
        current_shape = self.shape
        new_shape = list(current_shape)
        
        # Manejo de índice negativo
        if dim < 0: 
            dim += len(current_shape) + 1
            
        new_shape.insert(dim, 1)
        return self.reshape(new_shape)

    # Helpers internos (Para manejo de listas recursivas)
    
    def _flatten_data(self, data):
        """Convierte estructura anidada arbitraria en lista plana."""
        if not isinstance(data, list):
            return [data]
        flat = []
        for item in data:
            flat.extend(self._flatten_data(item))
        return flat

    def _unflatten_data(self, flat_data, shape):
        """Reconstruye lista anidada recursivamente desde lista plana."""
        if len(shape) == 0:
            return flat_data[0] # Escalar
            
        # Caso base: última dimensión
        if len(shape) == 1:
            return flat_data
            
        size_per_block = 1
        for s in shape[1:]: size_per_block *= s
        
        dims = shape[0]
        res = []
        for i in range(dims):
            start = i * size_per_block
            end = start + size_per_block
            chunk = flat_data[start:end]
            res.append(self._unflatten_data(chunk, shape[1:]))
            
        return res

    @property
    def shape(self):
        """Calcula la forma actual del tensor (suponiendo estructura rectangular)."""
        dims = []
        curr = self.data
        while isinstance(curr, list):
            dims.append(len(curr))
            if len(curr) > 0:
                curr = curr[0]
            else:
                break
        return tuple(dims)


    # Métodos Estáticos y Helpers Recursivos
    
    @staticmethod
    def sqrt(x):
        if isinstance(x, (int, float)): return math.sqrt(x)
        if isinstance(x, Tensor) and (x.shape == (1,1) or x.shape == (1,)):
             val = x.data[0] if isinstance(x.data[0], float) else x.data[0][0]
             return math.sqrt(val)
        raise ValueError("Tensor.sqrt solo soporta escalares")

    @staticmethod
    def uniform(low, high, shape):
        """Genera tensor con estructura N-Dimensional."""
        if not isinstance(shape, (list, tuple)):
            shape = (shape,)
            
        def _build_recursive(dims):
            if len(dims) == 1:
                return [random.uniform(low, high) for _ in range(dims[0])]
            return [_build_recursive(dims[1:]) for _ in range(dims[0])]
            
        return _build_recursive(shape)

    @staticmethod
    def zeros(shape):
        if not isinstance(shape, (list, tuple)):
             shape = (shape,)
        
        def _build_recursive(dims):
            if len(dims) == 1:
                return [0.0] * dims[0]
            return [_build_recursive(dims[1:]) for _ in range(dims[0])]
            
        return _build_recursive(shape)

    @staticmethod
    def zeros_like(data):
        """Clona la estructura de listas anidadas rellenándola con ceros."""
        if isinstance(data, list):
            return [Tensor.zeros_like(x) for x in data]
        return 0.0

    @staticmethod
    def ones_like(data):
        """Clona la estructura de listas anidadas rellenándola con unos."""
        if isinstance(data, list):
            return [Tensor.ones_like(x) for x in data]
        return 1.0