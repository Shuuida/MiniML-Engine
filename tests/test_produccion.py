from miniml import Tensor, nn, optim
from estimators.memory_estimator import estimate_memory
from miniml import print_cli_summary

print("=== Iniciando Prueba de MiniTensor ===")

# 1. Crear una red neuronal secuencial
modelo = nn.Sequential([
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
])

# 2. Hacer una predicción (Forward pass)
entrada = Tensor([[1.5, -0.5]])
salida = modelo(entrada)
print(f"Salida del modelo: {salida.data}")

# 3. Imprimir el resumen CLI MLOps
print_cli_summary(modelo)

# 4. Estimar consumo para el microcontrolador
memoria = estimate_memory(
    model=modelo,
    quantized=False,
    target_flash=32256,
    target_sram=2048,
    language="C++",
    input_shape=(1, 2)
)

print(f"Uso de RAM estimado: {memoria['sram_bytes']} bytes")
print("=== Prueba Finalizada con Éxito ===")