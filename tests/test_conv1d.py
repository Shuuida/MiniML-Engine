"""
===================================================
Test de QA: Procesamiento de Señales (DSP) en el Borde.
Evalúa la integridad de memoria temporal: Conv1D + MaxPool1D -> Flatten
"""

import os
from miniml import Tensor, nn, optim
from estimators.memory_estimator import estimate_memory
from miniml.exporters import cpp_writer
from miniml import print_cli_summary

print("1. Creando Dataset de Vibración (6 timesteps)...")
# Entradas: Shape (Batch=1, Canales=1, Longitud=6)
X_train = [
    Tensor([[[0.0, 1.0, 1.0, 1.0, 0.0, 0.0]]]), # Impacto sostenido
    Tensor([[[0.0, 0.0, 1.0, 1.0, 1.0, 0.0]]]), # Impacto sostenido desplazado
    Tensor([[[1.0, 0.0, 1.0, 0.0, 1.0, 0.0]]]), # Ruido alterno (Normal)
    Tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])  # Motor apagado (Normal)
]

# Salidas: 1.0 (Anomalía), 0.0 (Normal)
Y_train = [
    Tensor([[1.0]]), Tensor([[1.0]]),
    Tensor([[0.0]]), Tensor([[0.0]])
]

print("2. Construyendo Arquitectura DSP...")
model = nn.Sequential([
    # Entra: (1, 6)
    nn.Conv1d(in_channels=1, out_channels=2, kernel_size=3), 
    # Sale: (2, 4) -> 2 filtros aprendiendo patrones de 3 pasos
    
    nn.MaxPool1d(kernel_size=2),
    # Sale: (2, 2) -> Se queda con la señal más fuerte
    
    nn.Flatten(),
    # Sale: (4,) -> Puente unidimensional
    
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
])

print_cli_summary(model)

print("3. Entrenando en PC (Buscando el patrón de impacto)...")
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

print("\n4. Evaluando IA DSP:")
for x, y_true in zip(X_train, Y_train):
    pred = model(x).data[0][0]
    print(f"  Señal: {x.data[0][0]} -> Predicción: {pred:.4f} | Esperado: {y_true.data[0][0]}")

print("\n5. Exportando a C++...")
os.makedirs("exports", exist_ok=True)
# OJO: Indicamos el shape (Canales=1, Longitud=6)
cpp_code = cpp_writer.generate_cpp_code(model, input_shape=(1, 6))
with open("exports/miniml_model.h", "w") as f:
    f.write(cpp_code)
print("  [OK] Firmware DSP Exportado.")