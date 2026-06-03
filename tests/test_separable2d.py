"""
===================================================
Test de QA: SeparableConv2D (Estilo MobileNet)
Evalúa: Operator Fusion en C++, Memoria PROGMEM y Same Padding.
"""

import os
from miniml import Tensor, nn, optim
from miniml.exporters import cpp_writer

# 1. Definición robusta de la capa (Para garantizar compatibilidad con el Exportador)
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # Depthwise: Filtra espacialmente canal por canal
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding)
        # Pointwise: Mezcla canales con un kernel 1x1
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def __call__(self, x):
        return self.pointwise(self.depthwise(x))

    def parameters(self):
        return self.depthwise.parameters() + self.pointwise.parameters()

print("1. Creando Dataset de Visión (4x4)...")
X_train = [
    # Imagen 1: Cruz Centrada (Clase 1)
    Tensor([[[
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ]]]),
    # Imagen 2: Cruz Desplazada (Clase 1)
    Tensor([[[
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 1.0, 0.0]
    ]]]),
    # Imagen 3: Ruido en esquinas (Clase 0)
    Tensor([[[
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 1.0]
    ]]])
]

Y_train = [Tensor([[1.0]]), Tensor([[1.0]]), Tensor([[0.0]])]

print("2. Construyendo Arquitectura Edge AI...")
model = nn.Sequential([
    # Entra: (1, 4, 4) -> Sale: (2, 4, 4) (Gracias al padding=1)
    SeparableConv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    
    # Entra: (2, 4, 4) -> Sale: (2, 2, 2)
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    nn.Flatten(), # (2 * 2 * 2 = 8)
    nn.Linear(8, 1),
    nn.Sigmoid()
])

print("3. Entrenando en PC (Separable Autograd)...")
optimizer = optim.SGD(model.parameters(), lr=0.1) 

for epoch in range(1200):
    total_loss = 0.0
    for x, y_true in zip(X_train, Y_train):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = (y_pred - y_true) * (y_pred - y_true)
        loss.backward()
        optimizer.step()
        total_loss += loss.data[0][0]
        
    if epoch % 300 == 0:
        print(f"  -> Época {epoch} | Pérdida: {total_loss:.4f}")

print("\n4. Evaluando IA:")
for i, (x, y_true) in enumerate(zip(X_train, Y_train)):
    pred = model(x).data[0][0]
    esperado = y_true.data[0][0]
    print(f"  Imagen {i+1} -> Predicción: {pred:.4f} | Esperado: {esperado}")

print("\n5. Exportando a C++...")
os.makedirs("exports", exist_ok=True)
# OJO: Input shape puramente geométrico para C++ -> (Canales=1, Alto=4, Ancho=4)
cpp_code = cpp_writer.generate_cpp_code(model, input_shape=(1, 4, 4))
with open("exports/miniml_model.h", "w") as f:
    f.write(cpp_code)
print("  [OK] Exportación exitosa. Listo para Wokwi.")