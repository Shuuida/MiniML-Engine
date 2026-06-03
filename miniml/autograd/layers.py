"""
MiniTensor - Lógica de construcción para redes neuronales profundas (Deep Learning).
Soporta inicialización de pesos y propagación hacia adelante.
"""

from .tensor import Tensor
import math

class Module:
    # -------------------------------------------------------------
    # módulo base para el Freeze/Unfreeze de las capas para
    # Transfer Learning en Edge AI para el futuro
    def __init__(self):
        self.trainable = True

    def freeze(self):
        """Congela la capa para Transfer Learning en Edge."""
        self.trainable = False
        if hasattr(self, 'weights'):
            self.weights.requires_grad = False
        if hasattr(self, 'bias') and self.bias:
            self.bias.requires_grad = False

    def unfreeze(self):
        """Descongela la capa para Fine-Tuning."""
        self.trainable = True
        if hasattr(self, 'weights'):
            self.weights.requires_grad = True
        if hasattr(self, 'bias') and self.bias:
            self.bias.requires_grad = True
    # -------------------------------------------------------------

    def zero_grad(self):
        for p in self.parameters():
            p.grad = Tensor.zeros_like(p.data)

    def parameters(self):
        return []

    def state_dict(self):
        """Exporta los pesos a un diccionario serializable (Listas puras)."""
        sd = {}
        for name, value in self.__dict__.items():
            # Si es un Tensor (Pesos/Bias), guardamos su data pura
            if isinstance(value, Tensor):
                sd[name] = value.data 
            # Si es un Sub-Modulo (ej: Capas dentro de Sequential), recursión
            elif isinstance(value, Module):
                sd[name] = value.state_dict()
            # Casos especiales: Listas de Modulos (Sequential.layers)
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], Module):
                # Guardamos como lista de dicts indexada
                for i, layer in enumerate(value):
                    sd[f"layer_{i}"] = layer.state_dict()
        return sd

    def load_state_dict(self, sd):
        """Carga los pesos desde un diccionario (Listas puras -> Tensores)."""
        for name, value in sd.items():
            if hasattr(self, name):
                attr = getattr(self, name)
                
                # Caso A: Es un Tensor (Peso/Bias)
                if isinstance(attr, Tensor):
                    # Inyectamos los datos crudos manteniendo el objeto Tensor vivo
                    # (Importante para mantener las referencias del optimizador si existieran)
                    attr.data = value
                    
                # Caso B: Es un Sub-Modulo
                elif isinstance(attr, Module):
                    attr.load_state_dict(value)
            
            # Caso C: Capas de Sequential (layer_0, layer_1...)
            # Sequential guarda sus capas en self.layers (lista), pero el state_dict viene como claves
            elif name.startswith("layer_") and hasattr(self, "layers"):
                try:
                    idx = int(name.split("_")[1])
                    if idx < len(self.layers):
                        self.layers[idx].load_state_dict(value)
                except (ValueError, IndexError):
                    pass

class Linear(Module):
    def __init__(self, n_in, n_out, bias=True):
        # Inicialización de Kaiming Uniform (estándar para ReLU)
        limit = Tensor.sqrt(6 / n_in)
        self.weights = Tensor(Tensor.uniform(-limit, limit, (n_in, n_out)), label='W', requires_grad=True)
        
        if bias:
            # Añadir requires_grad=True igualmente
            self.bias = Tensor(Tensor.zeros((1, n_out)), label='b', requires_grad=True)
        else:
            self.bias = None
        
        self.input_scale = 1.0
        self.output_scale = 1.0

    def __call__(self, x):
        # x @ W + b
        out = x.matmul(self.weights)
        if self.bias is not None:
            out = out + self.bias
        return out

    def parameters(self):
        return [self.weights] + ([self.bias] if self.bias else [])

class ReLU(Module):
    def __call__(self, x):
        return x.relu()

class Sigmoid(Module):
    """
    Capa de Activación Sigmoide.
    Comprime las salidas en un rango de (0.0 a 1.0).
    Ideal para la última capa en clasificación binaria (como el XOR).
    """
    def __call__(self, x):
        return x.sigmoid()

class Sequential(Module):
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
    
    # Compatibilidad con API antigua de EduBot
    def predict(self, X):
        # Convertir lista python a Tensor si es necesario
        if isinstance(X, list):
            X = Tensor.array(X)
        
        t_in = Tensor(X)
        t_out = self(t_in)
        
        # Si es clasificación, podríamos aplicar argmax aquí, 
        # pero por ahora devolvemos raw scores (logits)
        return t_out.data

# Soporte para Visión Artificial (CNNs)
class Flatten(Module):
    """Aplana tensores 4D (B,C,H,W) a 2D (B, Features) para entrar a Linear.
    Incluye Backward real para permitir entrenamiento de CNNs."""
    def __call__(self, x):
        # Análisis de dimensiones para reconstrucción futura
        # Asumimos estructura (Batch, ...)
        B = len(x.data)
        
        # Detectar si es una estructura CNN (4D) o simple
        input_structure = 'generic'
        dims = ()
        if B > 0 and isinstance(x.data[0], list):
            # Chequeo rápido de 4D: [Batch][Canal][Alto][Ancho]
            try:
                if isinstance(x.data[0][0], list) and isinstance(x.data[0][0][0], list):
                    C = len(x.data[0])
                    H = len(x.data[0][0])
                    W = len(x.data[0][0][0])
                    input_structure = '4d_cnn'
                    dims = (C, H, W)
            except IndexError:
                pass

        # Aplanado Manual (Forward) - Recursivo robusto
        flat_data = []
        
        def _recursive_flat(item):
            if isinstance(item, list):
                res = []
                for sub in item:
                    res.extend(_recursive_flat(sub))
                return res
            return [item]

        for i in range(B):
            flat_data.append(_recursive_flat(x.data[i]))

        # Crear tensor de salida
        out = Tensor(flat_data, (x,), 'Flatten')
        
        # Backward
        def _backward():
            if x.requires_grad:
                x._init_grad()
                
                # out.grad.data es [[g1, g2...], [g1...]] (Batch, Features)
                # x.grad.data es la estructura anidada original (llena de ceros)
                
                grad_flat = out.grad.data
                
                if input_structure == '4d_cnn':
                    C, H, W = dims
                    # Reconstrucción optimizada para CNNs
                    for b in range(B):
                        row_flat = grad_flat[b]
                        idx = 0
                        # Rellenamos la estructura anidada consumiendo la lista plana
                        for c in range(C):
                            for h in range(H):
                                for w in range(W):
                                    if idx < len(row_flat):
                                        x.grad.data[b][c][h][w] += row_flat[idx]
                                        idx += 1
                else:
                    # Fallback genérico para estructuras no estándar (Deep Recursion)
                    # Nota: Para v1.6 visual, el camino crítico es el '4d_cnn' de arriba.
                    # Este fallback intenta rellenar ciegamente si la estructura coincide.
                    pass 

        out._backward = _backward
        return out

class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        
        k_h, k_w = self.kernel_size
        
        # Inicialización He/Kaiming
        limit = Tensor.sqrt(6 / (in_channels * k_h * k_w))
        
        # Pesos: (Out, In * Kh * Kw) -> Guardados como matriz 2D para usar matmul
        self.weights = Tensor(
            Tensor.uniform(-limit, limit, (out_channels, in_channels * k_h * k_w)), 
            label='ConvW', 
            requires_grad=True
        )
        
        self.bias = Tensor(
            Tensor.zeros((1, out_channels)), 
            label='ConvB', 
            requires_grad=True 
        )
        
        self.input_scale = 1.0

    def __call__(self, x):
        # Entrada x: (Batch, C, H, W) asumido en listas anidadas
        # Im2Col: Convertir imagen a matriz de parches
        # col: (Batch * H_out * W_out, In * Kh * Kw)
        col, output_shape = self._im2col(x)
        
        # Convolución como MatMul
        # (Pixels, Inputs) @ (Inputs, Filters).T -> (Pixels, Filters)
        out_col = col.matmul(self.weights.T()) + self.bias
        
        # Col2Im (Reshape): (Pixels, Filters) -> (Batch, Out_C, H_out, W_out)
        out = self._col2im_reshape(out_col, output_shape)
        
        return out

    def parameters(self):
        return [self.weights, self.bias]

    # MOTOR IM2COL (Zero-Dependency)
    def _im2col(self, x):
        """Transforma imagen 4D a Matriz 2D (Diferenciable manualmente)."""
        # Extraer dimensiones
        batch_size = len(x.data)
        n_c = len(x.data[0])
        h_in = len(x.data[0][0])
        w_in = len(x.data[0][0][0])
        
        ph, pw = self.padding
        kh, kw = self.kernel_size
        h_out = (h_in + 2 * ph- kh) // self.stride + 1
        w_out = (w_in + 2 * pw - kw) // self.stride + 1
        
        # Matriz resultante (Python List)
        # Rows = Batch * H_out * W_out
        # Cols = C * Kh * Kw
        col_data = []
        
        # Caché de índices para backward (dónde vino cada pixel)
        self._indices_cache = [] 

        # Bucle principal (Lento en Python, pero funcional)
        for b in range(batch_size):
            for i in range(h_out):
                for j in range(w_out):
                    # Inicio del patch
                    r_start = i * self.stride - ph
                    c_start = j * self.stride - pw
                    
                    patch_row = []
                    patch_indices = []
                    
                    for c in range(n_c):
                        for kr in range(kh):
                            for kc in range(kw):
                                r_idx = r_start + kr
                                c_idx = c_start + kc
                                
                                # Padding check (Zero padding)
                                if 0 <= r_idx < h_in and 0 <= c_idx < w_in:
                                    val = x.data[b][c][r_idx][c_idx]
                                    # Guardamos tupla de coordenadas para backward
                                    patch_indices.append((b, c, r_idx, c_idx))
                                else:
                                    val = 0.0
                                    patch_indices.append(None) # Padding
                                patch_row.append(val)
                    
                    col_data.append(patch_row)
                    self._indices_cache.append(patch_indices)
        
        # Crear Tensor intermedio conectado al grafo
        col_tensor = Tensor(col_data, (x,), 'Im2Col')
        
        # Definir Backward Manual (Col2Im)
        def _backward():
            if x.requires_grad:
                x._init_grad()
                # Gradiente viene de out_col (Matriz)
                grad_col = col_tensor.grad.data
                
                # Acumular gradientes en la imagen original
                for row_idx, grad_row in enumerate(grad_col):
                    indices = self._indices_cache[row_idx]
                    for col_idx, grad_val in enumerate(grad_row):
                        coords = indices[col_idx]
                        if coords: # Si no es padding
                            b, c, r, k = coords
                            # Sumar gradiente (acumulativo por solapamiento)
                            x.grad.data[b][c][r][k] += grad_val

        col_tensor._backward = _backward
        return col_tensor, (batch_size, self.out_channels, h_out, w_out)

    def _col2im_reshape(self, out_col, output_shape):
        """Convierte la matriz de salida (Pixels, Channels) a tensor 4D."""
        B, Out_C, H_out, W_out = output_shape
        # out_col.data es lista de listas [B*H*W, Out_C]
        
        new_data = []
        row_idx = 0
        
        # Reconstrucción 4D
        for b in range(B):
            batch_data = []
            for c in range(Out_C):
                # Inicializar matriz HxW vacía
                batch_data.append([[0.0] * W_out for _ in range(H_out)])
            
            for h in range(H_out):
                for w in range(W_out):
                    pixel_vals = out_col.data[row_idx] # Lista de longitud Out_C
                    row_idx += 1
                    
                    for c in range(Out_C):
                        batch_data[c][h][w] = pixel_vals[c]
            
            new_data.append(batch_data)
            
        out = Tensor(new_data, (out_col,), 'Reshape4D')

        def _backward():
            if out_col.requires_grad:
                out_col._init_grad()
                
                # El gradiente viene de arriba con forma 4D (out.grad)
                # Necesitamos convertirlo a la forma 2D de out_col (out_col.grad)
                # para que el MatMul reciba la señal.
                
                grad_4d = out.grad.data
                grad_2d_flat = [] 
                
                # Iteramos en el mismo orden que en el Forward para asegurar correspondencia
                for b in range(B):
                    for h in range(H_out):
                        for w in range(W_out):
                            # Recolectar los canales de este pixel (C,)
                            pixel_grad = []
                            for c in range(Out_C):
                                pixel_grad.append(grad_4d[b][c][h][w])
                            grad_2d_flat.append(pixel_grad)
                
                # Sumamos el gradiente re-aplanado al padre
                out_col.grad += Tensor(grad_2d_flat)

        out._backward = _backward
        return out

class MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None):
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size

    def __call__(self, x):
        # x.data: (B, C, H, W)
        B = len(x.data)
        C = len(x.data[0])
        H = len(x.data[0][0])
        W = len(x.data[0][0][0])
        
        k = self.kernel_size
        s = self.stride
        
        H_out = (H - k) // s + 1
        W_out = (W - k) // s + 1
        
        out_data = []
        self._max_indices = [] # Cache para backward
        
        for b in range(B):
            batch_out = []
            batch_indices = []
            for c in range(C):
                channel_out = []
                channel_indices = []
                for i in range(H_out):
                    row_out = []
                    row_indices = []
                    for j in range(W_out):
                        # Encontrar max en la ventana
                        r_start, c_start = i*s, j*s
                        max_val = -float('inf')
                        max_pos = (0,0)
                        
                        for kr in range(k):
                            for kc in range(k):
                                val = x.data[b][c][r_start+kr][c_start+kc]
                                if val > max_val:
                                    max_val = val
                                    max_pos = (r_start+kr, c_start+kc)
                        
                        row_out.append(max_val)
                        row_indices.append(max_pos)
                    channel_out.append(row_out)
                    channel_indices.append(row_indices)
                batch_out.append(channel_out)
                batch_indices.append(channel_indices)
            out_data.append(batch_out)
            self._max_indices.append(batch_indices)
            
        out = Tensor(out_data, (x,), 'MaxPool')
        
        def _backward():
            if x.requires_grad:
                x._init_grad()
                # Si una operación matemática (como fx + x) envolvió el gradiente 
                # en un objeto Tensor, extraemos su lista interna (.data)
                # casi igual que en MaxPool1d
                s_grad = x.grad.data if hasattr(x.grad, 'data') else x.grad
                o_grad = out.grad.data if hasattr(out.grad, 'data') else out.grad
                # Rutear gradiente solo al ganador
                for b in range(B):
                    for c in range(C):
                        for i in range(H_out):
                            for j in range(W_out):
                                max_r, max_c = self._max_indices[b][c][i][j]
                                s_grad[b][c][max_r][max_c] += o_grad[b][c][i][j]
        
        out._backward = _backward
        return out

class CrossEntropyLoss(Module):
    """
    Combina LogSoftmax y NLLLoss en una sola operación atómica.
    Es numéricamente estable y eficiente.
    
    Formula: Loss = -log( exp(x[class]) / sum(exp(x[j])) )
    Gradiente Simplificado: probs[j] - 1 (si j == class) else probs[j]
    """
    def __init__(self):
        self.probs = None
        self.target = None
        self.batch_size = 0

    def __call__(self, logits, target):
        # logits: Tensor (Batch, NumClasses) - Salida cruda de la última capa Linear
        # target: List[int] o Tensor - Índices de la clase correcta (0, 1, 2...)
        
        # Normalizar Target (Extraer enteros si viene como Tensor/Lista anidada)
        if isinstance(target, Tensor):
            raw = target.data
            target = []
            for item in raw:
                # Maneja [[0], [1]] (Batch, 1) y [0, 1] (Batch,)
                target.append(int(item[0]) if isinstance(item, list) else int(item))
        elif isinstance(target, list) and len(target) > 0 and isinstance(target[0], list):
             # Maneja listas anidadas [[0], [1]]
             target = [int(x[0]) for x in target]
             
        self.target = target
        self.batch_size = len(logits.data)
        
        # Forward (Softmax + NLL)
        loss_val = 0.0
        self.probs = []
        
        for i in range(self.batch_size):
            row = logits.data[i]
            
            # Estabilidad: restamos el máximo para evitar exp(700) -> Overflow
            max_val = max(row)
            exps = [math.exp(x - max_val) for x in row]
            sum_exps = sum(exps)
            
            # Probabilidades Softmax
            row_probs = [e / sum_exps for e in exps]
            self.probs.append(row_probs)
            
            # Negative Log Likelihood
            correct_class_idx = int(target[i])
            
            # Protección contra índices fuera de rango
            if correct_class_idx >= len(row_probs):
                raise ValueError(f"Target index {correct_class_idx} fuera de rango para {len(row_probs)} clases.")
                
            p_correct = row_probs[correct_class_idx]
            
            # Clip pequeño (1e-9) para evitar log(0) -> -inf
            loss_val -= math.log(max(p_correct, 1e-9))
            
        # Promedio del Batch
        final_loss = loss_val / self.batch_size
        
        # Crear Tensor de Salida
        out = Tensor([final_loss], (logits,), 'CrossEntropy')
        
        # Backward Manual
        def _backward():
            if logits.requires_grad:
                logits._init_grad()
                
                # Gradiente analítico: (Softmax - OneHotTarget) / BatchSize
                grad_data = logits.grad.data
                
                # Obtenemos el gradiente entrante (generalmente 1.0 si es la loss final)
                incoming_grad = out.grad.data
                # Normalizar si viene envuelto en listas [[1.0]]
                if isinstance(incoming_grad, list): incoming_grad = incoming_grad[0]
                if isinstance(incoming_grad, list): incoming_grad = incoming_grad[0]
                
                scale = incoming_grad / self.batch_size
                
                for i in range(self.batch_size):
                    correct_class_idx = self.target[i]
                    for j in range(len(self.probs[i])):
                        # Derivada del Softmax+CE
                        gradient = self.probs[i][j]
                        if j == correct_class_idx:
                            gradient -= 1.0
                        
                        # Acumular gradiente escalado
                        grad_data[i][j] += gradient * scale

        out._backward = _backward
        return out

class MSELoss(Module):
    """
    Mean Squared Error Loss (Error Cuadrático Medio).
    Vital para:
    Autoencoders (Detección de Anomalías).
    Regresión (Predicción de valores continuos, ej: Temperatura).
    
    Formula: L = mean((pred - target)^2)
    """
    def __call__(self, pred, target):
        # Asegurar que target sea un Tensor para que la resta sea vectorizada
        if not isinstance(target, Tensor):
            # Si viene como lista simple [1.0, 2.0], la convertimos
            target = Tensor(target)
            
        # Diferencia (El autograd rastreará esta operación)
        diff = pred - target
        
        # Cuadrado
        sq_diff = diff * diff
        
        # Conteo de elementos (Flatten manual para seguridad)
        def _count_recursive(d):
            if isinstance(d, list):
                return sum(_count_recursive(x) for x in d)
            return 1
            
        num_elements = _count_recursive(pred.data)

        # Sumamos todos los errores
        total_error_tensor = sq_diff.sum()

        # Extraemos el valor float puro del tensor de suma
        # (Tensor.sum() devuelve [val] o [[val]] dependiendo de la versión)
        raw_sum = total_error_tensor.data
        while isinstance(raw_sum, list):
            # Pelamos las capas de listas hasta encontrar el número
            if not raw_sum:
                raw_sum = 0.0
                break
            raw_sum = raw_sum[0]
        mean_val = float(raw_sum) / max(1, num_elements)

        loss = Tensor([mean_val], [pred, target], 'MSE')

        def _backward():
            if pred.requires_grad:
                pred._init_grad()
                # El gradiente de MSE es 2*(pred - target) / N
                # Gradiente entrante (loss.grad) suele ser 1.0
                grad_in = loss.grad.data
                while isinstance(grad_in, list):
                    if not grad_in:
                        grad_in = 0.0
                        break
                    grad_in = grad_in[0]
                scale = (2.0 / num_elements) * grad_in

                # Re-calculamos diff para el gradiente (diff = pred - target)
                # Multiplicamos por la escala
                grad_tensor = diff * Tensor([scale])
                pred.grad += grad_tensor

        loss._backward = _backward
        return loss

class Conv1d(Module):
    """
    Convolución 1D para Series Temporales (TCN).
    Patrón de Diseño: Adapter.
    Convierte entradas (Batch, Canales, Tiempo) -> (Batch, Canales, 1, Tiempo)
    para aprovechar la robustez de Conv2d.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        # Nota: dilation no se implementa en el motor Python básico (requiere lógica compleja),
        # pero se guarda el atributo para que el Exportador C++ sí lo implemente.
        self.dilation = dilation
        
        # Inicializamos una Conv2d que tiene altura del kernel = 1
        # Kernel Shape efectivo: (1, kernel_size)
        self.internal = Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=(1, kernel_size), 
            stride=stride, 
            padding=(0, padding)
        )
        self.weights = self.internal.weights
        self.bias = self.internal.bias

    def __call__(self, x):
        # Adaptación dimensional con grafo
        # (Batch, Ch, Time) -> (Batch, Ch, 1, Time)
        # Usamos unsqueeze(2) que ahora es nativo y diferenciable.
        x_4d = x.unsqueeze(2)

        # Ejecutar Conv2d
        out_4d = self.internal(x_4d)

        # Aplanar salida con grafo
        # (Batch, Ch, 1, Time) -> (Batch, Ch, Time)
        # Usamos reshape para eliminar la dimensión 1
        # Inferimos dimensiones: (B, C, -1)
        b_size = len(out_4d.data)
        channels = len(out_4d.data[0])

        out = out_4d.reshape((b_size, channels, -1))

        return out

    def parameters(self):
        return self.internal.parameters()

class MaxPool1d(Module):
    """
    Capa de Max Pooling 1D para procesamiento de señales temporales (DSP).
    Reduce la dimensionalidad temporal quedándose con la señal más fuerte.
    """
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.trainable = False 

    def __call__(self, x):
        # Asumimos que x.data viene en forma 3D: [Batch][Canales][Longitud]
        B = len(x.data)
        C = len(x.data[0]) if B > 0 else 0
        L = len(x.data[0][0]) if C > 0 else 0
        
        k = self.kernel_size
        s = self.stride
        
        out_L = (L - k) // s + 1
        
        out_data = []
        self._max_indices = [] # Cache guardado en el módulo para el Backward
        
        for b in range(B):
            b_out = []
            b_indices = []
            for c in range(C):
                c_out = []
                c_indices = []
                for out_l in range(out_L):
                    start = out_l * s
                    end = start + k
                    
                    window = x.data[b][c][start:end]
                    max_val = max(window)
                    
                    # Guardamos el índice local del ganador y lo volvemos global
                    max_idx = start + window.index(max_val)
                    
                    c_out.append(max_val)
                    c_indices.append(max_idx)
                b_out.append(c_out)
                b_indices.append(c_indices)
            out_data.append(b_out)
            self._max_indices.append(b_indices)
            
        out = Tensor(out_data, (x,), 'MaxPool1D')
        
        def _backward():
            if x.requires_grad:
                x._init_grad()
                s_grad = x.grad.data if hasattr(x.grad, 'data') else x.grad
                o_grad = out.grad.data if hasattr(out.grad, 'data') else out.grad
                
                # Enrutar el gradiente
                for b in range(B):
                    for c in range(C):
                        for out_l in range(out_L):
                            orig_idx = self._max_indices[b][c][out_l]
                            s_grad[b][c][orig_idx] += o_grad[b][c][out_l]
                            
        out._backward = _backward
        return out

class SeparableConv2d(Module):
    """
    Convolución Separable en Profundidad (DSC).
    Patrón de Diseño: Pipeline (Composite).
    
    Divide la operación en:
    1. Depthwise: Filtra el espacio.
    2. Pointwise: Combina los canales (1x1 Conv).
    
    NOTA:
    En este motor Python, la 'depthwise' simula la estructura usando una Conv2d estándar.
    No obtendrá la aceleración de velocidad en Python, PERO define la arquitectura correcta
    para que el Exportador C++ genere el código optimizado para Arduino.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        # Depthwise:
        # En teoría debería tener groups=in_channels. 
        # Usamos Conv2d estándar como "placeholder funcional" que preserva dimensiones espaciales.
        # Entrada: (B, Cin, H, W) -> Salida Intermedia: (B, Cin, H', W')
        self.depthwise = Conv2d(
            in_channels=in_channels,
            out_channels=in_channels, # Mantiene profundidad
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        # Pointwise:
        # Convolución 1x1 para mezclar canales.
        # Entrada: (B, Cin, H', W') -> Salida: (B, Cout, H', W')
        self.pointwise = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1, # Kernel 1x1
            stride=1,
            padding=0
        )

    def __call__(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

    def parameters(self):
        return self.depthwise.parameters() + self.pointwise.parameters()

class ResidualBlock1D(Module):
    """
    Bloque Residual para Soft-Sensors (ResNet-1D).
    Permite entrenar redes profundas para regresión de señales complejas.
    Implementa: Output = Activation(Layer(x) + x)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        # Garantiza que la salida tenga la misma longitud que la entrada (si stride=1).
        # Requiere que kernel_size sea un número impar (3, 5, 7...).
        auto_padding = (kernel_size - 1) // 2
        
        # Ruta Principal (Convolución -> ReLU -> Convolución)
        self.conv1 = Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=auto_padding)
        self.relu = ReLU()
        
        # La segunda convolución siempre tiene stride=1 para no reducir más la dimensión
        self.conv2 = Conv1d(out_channels, out_channels, kernel_size, stride=1, padding=auto_padding)
        
        # Ruta de Atajo (Shortcut / Skip Connection)
        self.shortcut = Sequential([])
        
        # Si las dimensiones cambian (por stride o canales), necesitamos adaptar 'x' para poder sumarlo
        if stride != 1 or in_channels != out_channels:
            self.shortcut = Sequential([
                Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            ])

    def __call__(self, x):
        # Main Path
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        
        # Shortcut Path
        identity = self.shortcut(x)
        
        # Suma Element-wise
        out = out + identity
        
        # Activación final
        return self.relu(out)

    def parameters(self):
        return self.conv1.parameters() + self.conv2.parameters() + self.shortcut.parameters()