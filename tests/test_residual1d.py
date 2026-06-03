"""
===================================================
Test de QA: Bloques Residuales en el Borde.
Evalúa: Ramificación del Grafo Computacional (x + F(x))
"""

import os
from miniml import Tensor, nn, optim
from estimators.memory_estimator import estimate_memory
from miniml.exporters import cpp_writer
from miniml import print_cli_summary

print("1. Creando Dataset de Series Temporales (8 timesteps)...")
# Shape: (Batch=1, Canales=1, Longitud=8)
X_train = [
    Tensor([[[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]]]), # Oscilación estable (Clase 1)
    Tensor([[[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]]]), 
    Tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]), # Silencio (Clase 0)
    Tensor([[[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]]])  # Saturación inicial (Clase 0)
]

Y_train = [
    Tensor([[1.0]]), Tensor([[1.0]]),
    Tensor([[0.0]]), Tensor([[0.0]])
]

print("2. Construyendo Arquitectura Residual...")
model = nn.Sequential([
    # Entra: (1, 8). Sale (2, 8)
    nn.Conv1d(in_channels=1, out_channels=2, kernel_size=1),
    
    # El Bloque Residual (Entra 2 canales, Salen 2 canales)
    nn.ResidualBlock1D(in_channels=2, out_channels=2),
    
    # Reducción espacial
    nn.MaxPool1d(kernel_size=2), # Sale (2, 4)
    
    nn.Flatten(), # Sale (8)
    nn.Linear(8, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
])

try:
    print_cli_summary(model)
except Exception as e:
    print(f"\n[Aviso CLI] El formateador no pudo imprimir el bloque residual: {e}\n")

print("3. Entrenando en PC (Autograd con Skip Connections)...")
optimizer = optim.SGD(model.parameters(), lr=0.1) 

for epoch in range(1000):
    total_loss = 0.0
    for x, y_true in zip(X_train, Y_train):
        optimizer.zero_grad()
        y_pred = model(x)
        
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
    print(f"  Entrada: {x.data[0][0]} -> Predicción: {pred:.4f} | Esperado: {y_true.data[0][0]}")

print("\n5. Exportando a C++...")
os.makedirs("exports", exist_ok=True)
try:
    cpp_code = cpp_writer.generate_cpp_code(model, input_shape=(1, 8))
    with open("exports/miniml_model.h", "w") as f:
        f.write(cpp_code)
    print("  [OK] C++ Exportado exitosamente.")
except Exception as e:
    print(f"  [ERROR C++] El exportador falló al procesar el bloque residual: {e}")