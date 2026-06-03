"""
===================================================
Script de entrenamiento MiniTensor y exportación a C++/Rust.
Ideal para pruebas Hardware-in-the-Loop en Wokwi.
"""

import os
from miniml import Tensor, nn, optim
from estimators.memory_estimator import estimate_memory
from miniml.exporters import cpp_writer, rust_writer
from miniml import print_cli_summary

print("1. Inicializando Dataset XOR...")
# Entradas (Batch de 1 para simplificar Autograd didáctico)
X_train = [
    Tensor([[0.0, 0.0]]),
    Tensor([[0.0, 1.0]]),
    Tensor([[1.0, 0.0]]),
    Tensor([[1.0, 1.0]])
]
# Salidas esperadas
Y_train = [
    Tensor([[0.0]]),
    Tensor([[1.0]]),
    Tensor([[1.0]]),
    Tensor([[0.0]])
]

print("2. Construyendo Arquitectura Deep Learning...")
model = nn.Sequential([
    nn.Linear(2, 8),   # Capa de entrada a oculta
    nn.ReLU(),         # Activación no lineal
    nn.Linear(8, 1),    # Capa oculta a salida
    nn.Sigmoid()     # Activación sigmoide
])

# Resumen de arquitectura CLI
print_cli_summary(model)

print("3. Entrenando en PC (Autograd activado)...")
optimizer = optim.SGD(model.parameters(), lr=0.5)

epochs = 3000
for epoch in range(epochs):
    total_loss = 0.0
    for x, y_true in zip(X_train, Y_train):
        optimizer.zero_grad()
        
        # Forward Pass
        y_pred = model(x)
        
        # Mean Squared Error (MSE) Loss manual usando tensores
        diff = y_pred - y_true
        loss = diff * diff 
        
        # Backward Pass (Calcula gradientes mágicamente)
        loss.backward()
        
        # Actualiza pesos
        optimizer.step()
        
        total_loss += loss.data[0][0]
        
    if epoch % 200 == 0:
        print(f"  -> Época {epoch} | Pérdida (Loss): {total_loss:.4f}")

print("\n4. Evaluando Modelo Entrenado:")
for x, y_true in zip(X_train, Y_train):
    pred = model(x).data[0][0]
    esperado = y_true.data[0][0]
    print(f"  Entrada: {x.data[0]} | Predicción: {pred:.4f} | Esperado: {esperado}")

print("\n5. Auditoría de Memoria para Hardware Embebido:")
mem_report = estimate_memory(
    model=model,
    quantized=False, # Float32 estándar para esta prueba
    target_flash=32256,
    target_sram=2048,
    language="C++",
    input_shape=(1, 2)
)
print(f"  -> SRAM requerida (Peak Arena): {mem_report['sram_bytes']} bytes")
print(f"  -> Flash requerida (Pesos + Lógica): {mem_report['flash_bytes']} bytes")

print("\n6. Exportando Firmware...")
os.makedirs("exports", exist_ok=True)

# Exportar a C++ (Para Arduino en Wokwi)
cpp_code = cpp_writer.generate_cpp_code(model, input_shape=(1, 2))
with open("exports/miniml_model.h", "w") as f:
    f.write(cpp_code)
print("  [OK] C++ Exportado: exports/miniml_model.h")

# Exportar a Rust (Para ecosistemas no_std)
rust_code = rust_writer.generate_rust_code(model, input_shape=(1, 2))
with open("exports/miniml_model.rs", "w") as f:
    f.write(rust_code)
print("  [OK] Rust Exportado: exports/miniml_model.rs")

print("\n=== Pipeline Finalizado Exitosamente ===")