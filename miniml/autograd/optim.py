"""
MiniTensor - Optimizadores para el motor de Deep Learning.
Gestiona la actualización de pesos (W -= lr * grad).
"""

from .tensor import Tensor

class SGD:
    def __init__(self, parameters, lr=0.01, momentum=0.0):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        # Inicializamos velocidades como Tensores reales
        self.velocities = [Tensor(Tensor.zeros_like(p.data)) for p in self.parameters]

    def step(self):
        """
        Realiza un paso de optimización usando aritmética de Tensores.
        Esto asegura que se respeten las dimensiones y el broadcasting corregido.
        """
        for i, p in enumerate(self.parameters):
            if p.grad is None:
                continue
            
            # Calcular el término de gradiente (-lr * grad)
            # Usamos Tensor([[-self.lr]]) para broadcasting escalar seguro
            grad_term = p.grad * Tensor([[-self.lr]])
            
            # Calcular el término de momentum (momentum * velocity_old)
            momentum_term = self.velocities[i] * Tensor([[self.momentum]])
            
            # Actualizar Velocidad (v_new = m * v_old - lr * grad)
            new_velocity = momentum_term + grad_term
            self.velocities[i] = new_velocity
            
            # Actualizar Peso (w_new = w_old + v_new)
            # Usamos suma de tensores para aplicar la actualización estructuralmente correcta
            p_updated = p + new_velocity
            
            # Inyectamos los nuevos datos en el tensor de parámetros
            p.data = p_updated.data

    def zero_grad(self):
        for p in self.parameters:
            p.grad = Tensor(Tensor.zeros_like(p.data))