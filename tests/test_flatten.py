"""
===================================================
Test de QA: Transición Espacial (Multidimensional a Plano).
Evalúa la integridad de la memoria al usar Flatten()
"""

import os
from miniml import Tensor, nn, optim
from estimators.memory_estimator import estimate_memory
from miniml.exporters import cpp_writer
from miniml import print_cli_summary

print("1. Creando Dataset de Visión 2x2...")
# Entradas (Batch de 1, Matriz 2x2)
X_train = [
    Tensor([[[1.0, 0.0], [1.0, 0.0]]]), # Línea Vertical
    Tensor([[[0.0, 1.0], [0.0, 1.0]]]), # Línea Vertical desplazada
    Tensor([[[1.0, 1.0], [0.0, 0.0]]]), # Línea Horizontal
    Tensor([[[0.0, 0.0], [1.0, 1.0]]])  # Línea Horizontal desplazada
]

# Salidas: 1.0 para Vertical, 0.0 para Horizontal
Y_train = [
    Tensor([[1.0]]), Tensor([[1.0]]),
    Tensor([[0.0]]), Tensor([[0.0]])
]

print("2. Construyendo Arquitectura Híbrida...")
model = nn.Sequential([
    nn.Flatten(),      # Transforma (2, 2) -> (4,)
    nn.Linear(4, 16),   # Capa Oculta
    nn.ReLU(),
    nn.Linear(16, 1),   # Clasificador
    nn.Sigmoid()       # Salida binaria (0.0 o 1.0)
])

print_cli_summary(model)

print("3. Entrenando en PC...")
# Usamos un Learning Rate más alto para converger rápido en este problema lineal
optimizer = optim.SGD(model.parameters(), lr=0.5) 

for epoch in range(1000):
    total_loss = 0.0
    for x, y_true in zip(X_train, Y_train):
        optimizer.zero_grad()
        y_pred = model(x)
        
        # MSE Loss manual
        diff = y_pred - y_true
        loss = diff * diff 
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.data[0][0]
        
    if epoch % 200 == 0:
        print(f"  -> Época {epoch} | Pérdida: {total_loss:.4f}")

print("\n4. Evaluando IA:")
for x, y_true in zip(X_train, Y_train):
    pred = model(x).data[0][0]
    print(f"  Entrada:\n{x.data[0][0]}\n{x.data[0][1]}  -> Predicción: {pred:.4f} | Esperado: {y_true.data[0][0]}\n")

print("5. Exportando a C++...")
os.makedirs("exports", exist_ok=True)
# OJO: Aquí le indicamos al estimador y al exportador que el input es espacial
cpp_code = cpp_writer.generate_cpp_code(model, input_shape=(1, 2, 2))
with open("exports/miniml_model.h", "w") as f:
    f.write(cpp_code)
print("  [OK] C++ Exportado exitosamente.")