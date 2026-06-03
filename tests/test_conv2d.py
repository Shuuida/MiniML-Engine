"""
===================================================
Test de QA: Visión Artificial 2D en el Borde.
Evalúa: Conv2D -> MaxPool2D -> Flatten -> Linear
"""

import os
from miniml import Tensor, nn, optim
from estimators.memory_estimator import estimate_memory
from miniml.exporters import cpp_writer
from miniml import print_cli_summary

print("1. Creando Dataset de Visión 4x4...")
# Shape esperado: [Batch, Canales, Alto, Ancho] -> (1, 1, 4, 4)
X_train = [
    # Imagen 1: Línea Vertical central
    Tensor([[[
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0]
    ]]]),
    # Imagen 2: Línea Vertical desplazada
    Tensor([[[
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]
    ]]]),
    # Imagen 3: Línea Horizontal central
    Tensor([[[
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ]]]),
    # Imagen 4: Línea Horizontal desplazada
    Tensor([[[
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 0.0]
    ]]])
]

# Salidas: 1.0 (Vertical), 0.0 (Horizontal)
Y_train = [
    Tensor([[1.0]]), Tensor([[1.0]]),
    Tensor([[0.0]]), Tensor([[0.0]])
]

print("2. Construyendo Arquitectura de Visión...")
model = nn.Sequential([
    # Entra: (1, 4, 4). Sale: (2, 3, 3)
    nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2), 
    
    # Entra: (2, 3, 3). Sale: (2, 2, 2)
    nn.MaxPool2d(kernel_size=2, stride=1),
    
    nn.Flatten(),
    # Sale: (2 * 2 * 2) = 8
    
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
])

print_cli_summary(model)

print("3. Entrenando en PC (Buscando bordes espaciales)...")
optimizer = optim.SGD(model.parameters(), lr=0.1) 

for epoch in range(1500):
    total_loss = 0.0
    for x, y_true in zip(X_train, Y_train):
        optimizer.zero_grad()
        y_pred = model(x)
        
        diff = y_pred - y_true
        loss = diff * diff 
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.data[0][0]
        
    if epoch % 300 == 0:
        print(f"  -> Época {epoch} | Pérdida: {total_loss:.4f}")

print("\n4. Evaluando IA de Visión:")
for i, (x, y_true) in enumerate(zip(X_train, Y_train)):
    pred = model(x).data[0][0]
    esperado = y_true.data[0][0]
    print(f"  Imagen {i+1} -> Predicción: {pred:.4f} | Esperado: {esperado}")

print("\n5. Exportando a C++...")
os.makedirs("exports", exist_ok=True)
# OJO: Indicamos el shape (Canales=1, Alto=4, Ancho=4)
cpp_code = cpp_writer.generate_cpp_code(model, input_shape=(1, 4, 4))
with open("exports/miniml_model.h", "w") as f:
    f.write(cpp_code)
print("  [OK] Firmware de Visión Exportado.")