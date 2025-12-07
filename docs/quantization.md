# Cuantificación en MiniML: Guía Completa

## Tabla de Contenidos
1. [Introducción](#introducción)
2. [Arquitectura de Cuantificación](#arquitectura-de-cuantificación)
3. [Métodos de Cuantificación](#métodos-de-cuantificación)
4. [Proceso de Cuantificación](#proceso-de-cuantificación)
5. [Exportación a Firmware C](#exportación-a-firmware-c)
6. [Tabla Comparativa](#tabla-comparativa)
7. [Limitaciones Actuales](#limitaciones-actuales)
8. [Recomendaciones para Proyectos Embebidos](#recomendaciones-para-proyectos-embebidos)
9. [Ejemplos de Uso](#ejemplos-de-uso)

---

## Introducción

MiniML implementa **Post-Training Quantization (PTQ)** para redes neuronales, permitiendo reducir el tamaño del modelo y acelerar la inferencia en microcontroladores de bajo costo mediante el uso de aritmética de enteros de 8 bits (int8) en lugar de punto flotante de 32 bits.

### Beneficios Clave
- **Reducción de memoria**: ~75% menos espacio (int8 vs float32)
- **Aceleración**: Operaciones enteras más rápidas en MCUs sin FPU
- **Compatibilidad CMSIS-NN**: Integración con kernels optimizados de ARM
- **Precisión preservada**: Pérdida típica < 2% en accuracy

---

## Arquitectura de Cuantificación

### Componentes Principales

#### 1. **MiniNeuralNetwork** (`ml_runtime.py`)
Clase base que implementa:
- Calibración de activaciones (`calibrate()`)
- Cuantificación de pesos y biases (`quantize()`)
- Exportación nativa a C (`to_arduino_code()`)

#### 2. **CMSISAdapter** (`adapters/cmsis_nn/adapter.py`)
Adaptador avanzado que genera código compatible con:
- **CMSIS-NN**: Kernels optimizados de ARM para Cortex-M
- **Fallback portátil**: Implementación en C estándar sin dependencias

### Flujo de Cuantificación

```
Entrenamiento (Float32)
    ↓
Calibración (calibrate())
    ↓ [Calcula act_scales: input, hidden, output]
Cuantificación (quantize())
    ↓ [Convierte pesos a int8, biases a int32]
Exportación (export_to_c())
    ↓ [Genera código C optimizado]
Firmware C (int8 inference)
```

---

## Métodos de Cuantificación

### 1. Cuantificación Simétrica por Capa (Per-Layer Symmetric)

**Implementación**: Método por defecto en `MiniNeuralNetwork.quantize()`

#### Características:
- **Pesos**: Cuantificados a `int8` con rango `[-127, 127]`
- **Zero-point**: Implícitamente 0 (symmetric quantization)
- **Escala por capa**: Una escala por matriz de pesos
- **Biases**: Cuantificados a `int32` para preservar precisión

#### Fórmulas:

**Para Pesos (W):**
```
abs_max = max(|min(W)|, |max(W)|)
scale_w = abs_max / 127.0
q_w = round(w / scale_w)  # Clipped to [-127, 127]
```

**Para Biases (B):**
```
effective_scale = input_scale * scale_w
b_int32 = round(b / effective_scale)  # Clipped to int32 range
```

**Multiplicador de Requantización:**
```
requant_mult = effective_scale / output_scale
```

#### Ventajas:
- ✅ Implementación simple y rápida
- ✅ Bajo overhead computacional
- ✅ Compatible con CMSIS-NN
- ✅ Buena precisión para redes pequeñas-medianas

#### Desventajas:
- ❌ Menor precisión que per-channel para redes grandes
- ❌ Sensible a outliers en distribución de pesos

---

### 2. Cuantificación con CMSIS-NN (Fixed-Point)

**Implementación**: `CMSISAdapter.generate_c()`

#### Características:
- **Pesos**: `int8_t` almacenados en arrays alineados
- **Biases**: `int32_t` pre-cuantificados
- **Multiplicadores**: Convertidos a formato Q31 (significand + shift)
- **Activaciones**: `int8_t` en inferencia completa

#### Formato Q31 para Multiplicadores:
```python
def _quantize_multiplier(real_multiplier: float) -> Tuple[int, int]:
    significand, shift = math.frexp(real_multiplier)
    q_mult = int(round(significand * (1 << 31)))
    # Ajuste para evitar overflow
    if q_mult == (1 << 31):
        q_mult //= 2
        shift += 1
    return q_mult, shift
```

#### Ventajas:
- ✅ Máxima velocidad en Cortex-M (kernels SIMD)
- ✅ Inferencia completamente en int8 (sin float)
- ✅ Bajo consumo de energía
- ✅ Compatible con ARM CMSIS-NN

#### Desventajas:
- ❌ Requiere librería CMSIS-NN (no portátil)
- ❌ Limitado a arquitecturas ARM Cortex-M
- ❌ Mayor complejidad de implementación

---

### 3. Modo Híbrido AVR (Arduino 8-bit)

**Implementación**: `MiniNeuralNetwork.to_arduino_code()`

#### Características:
- **Pesos**: `int8_t` almacenados en `PROGMEM` (Flash)
- **Escalas**: `float` almacenadas en `PROGMEM`
- **Cálculo**: Híbrido (int8 storage, float compute)
- **Biases**: `float` originales en `PROGMEM`

#### Estrategia:
```c
// Lectura de peso cuantificado desde Flash
int8_t w = pgm_read_byte(&W1[i][j]);
// Descuantificación on-the-fly
float dequantized = (float)w * scale * input;
```

#### Ventajas:
- ✅ Ahorra SRAM (pesos en Flash, no RAM)
- ✅ Ideal para AVR 8-bit (Arduino Uno/Nano)
- ✅ No requiere FPU (usa float solo para escalas)
- ✅ Portátil (sin dependencias externas)

#### Desventajas:
- ❌ Más lento que int8 puro (conversiones float)
- ❌ Requiere FPU o emulación de float
- ❌ Mayor uso de Flash que int8 puro

---

## Proceso de Cuantificación

### Paso 1: Calibración (`calibrate()`)

**Propósito**: Determinar los rangos de activación para cada capa.

```python
def calibrate(self, dataset: List[List[float]]):
    """
    Calcula rangos de activación (min/max) para Input, Hidden y Output.
    Esencial para cuantificación int8 (Post-Training Quantization).
    """
    # Itera sobre el dataset y encuentra:
    # - max_in: máximo absoluto de inputs
    # - max_hidden: máximo absoluto de activaciones ocultas
    # - max_out: máximo absoluto de outputs
    
    self.act_scales = {
        'input': max_in / 127.0,
        'hidden': max_hidden / 127.0,
        'output': max_out / 127.0
    }
```

**Notas Importantes**:
- Se ejecuta automáticamente después de `fit()`
- Requiere dataset de calibración (típicamente el de entrenamiento)
- Los escalas se guardan en `act_scales` para uso posterior

### Paso 2: Cuantificación (`quantize()`)

**Propósito**: Convertir pesos y biases de float32 a int8/int32.

```python
def quantize(self, per_channel: bool = True):
    """
    Cuantifica pesos a int8 y biases a int32.
    Requiere act_scales calculados previamente.
    """
    # Para cada capa:
    # 1. Calcula escala por fila de pesos
    # 2. Cuantifica pesos a int8
    # 3. Cuantifica biases a int32
    # 4. Calcula multiplicadores de requantización
    
    self.q_W1, self.i32_B1, self.requant_mult1, self.s_W1_list = ...
    self.q_W2, self.i32_B2, self.requant_mult2, self.s_W2_list = ...
    self.quantized = True
```

**Atributos Generados**:
- `q_W1`, `q_W2`: Matrices de pesos cuantificadas (int8)
- `i32_B1`, `i32_B2`: Vectores de bias cuantificados (int32)
- `requant_mult1`, `requant_mult2`: Multiplicadores de requantización (float)
- `s_W1_list`, `s_W2_list`: Escalas por fila de pesos (float)

### Paso 3: Exportación (`export_to_c()`)

**Propósito**: Generar código C optimizado para el firmware.

**Detección Automática**:
- Si el modelo tiene `q_W1` → Usa `CMSISAdapter` (preferido)
- Si falla → Usa `to_arduino_code()` (fallback nativo)
- Si tiene scaler → Incluye código de preprocesamiento

---

## Exportación a Firmware C

### Opción 1: CMSIS-NN (Recomendado para ARM Cortex-M)

**Generado por**: `CMSISAdapter.generate_c()`

**Características**:
- Código optimizado para `arm_fully_connected_s8()`
- Inferencia completamente en int8
- Fallback portátil si CMSIS-NN no está disponible

**Estructura del Código**:
```c
// Arrays de datos (alineados para SIMD)
const int8_t W1[N] ALIGNED(4) = {...};
const int32_t B1[M] ALIGNED(4) = {...};
const int32_t MULT1[M] ALIGNED(4) = {...};
const int32_t SHIFT1[M] ALIGNED(4) = {...};

#ifdef CMSISNN_ENABLED
    // Usa kernels optimizados de ARM
    arm_fully_connected_s8(...);
#else
    // Fallback portátil en C estándar
    // Implementación manual con loops
#endif
```

**Ventajas**:
- Máxima velocidad en Cortex-M4/M7
- Bajo consumo de energía
- Inferencia pura int8 (sin float)

### Opción 2: Modo Híbrido AVR (Arduino 8-bit)

**Generado por**: `MiniNeuralNetwork.to_arduino_code()`

**Características**:
- Pesos en `PROGMEM` (Flash)
- Cálculo híbrido (int8 storage, float compute)
- Ideal para AVR sin FPU

**Estructura del Código**:
```c
#include <avr/pgmspace.h>

// Pesos cuantificados en Flash
const int8_t W1[N][M] PROGMEM = {...};
// Escalas en Flash
const float sW1[N] PROGMEM = {...};
// Biases originales en Flash
const float B1[N] PROGMEM = {...};

void predict(float *row, float *out) {
    // Descuantificación on-the-fly
    float w = (float)pgm_read_byte(&W1[i][j]) * pgm_read_float(&sW1[i]);
    // Cálculo en float
    sum += w * input[j];
}
```

**Ventajas**:
- Ahorra SRAM (datos en Flash)
- Portátil (sin dependencias)
- Compatible con Arduino Uno/Nano

---

## Tabla Comparativa

| Característica | Per-Layer Symmetric | CMSIS-NN Fixed-Point | Modo Híbrido AVR |
|----------------|---------------------|----------------------|-------------------|
| **Precisión de Pesos** | int8 | int8 | int8 |
| **Precisión de Biases** | int32 | int32 | float32 |
| **Precisión de Activaciones** | float32 (ref) | int8 | float32 |
| **Escala** | Por capa | Por capa | Por capa |
| **Zero-Point** | 0 (symmetric) | 0 (symmetric) | 0 (symmetric) |
| **Almacenamiento Pesos** | RAM/Flash | RAM (alineado) | PROGMEM (Flash) |
| **Velocidad Inferencia** | Media | Muy Alta | Baja-Media |
| **Consumo Energía** | Medio | Bajo | Medio |
| **Memoria Requerida** | ~75% menos | ~75% menos | ~75% menos (pesos) |
| **Compatibilidad** | Universal | ARM Cortex-M | AVR 8-bit |
| **Dependencias** | Ninguna | CMSIS-NN | Ninguna |
| **FPU Requerido** | Opcional | No | Sí (emulación OK) |
| **SIMD Optimizado** | No | Sí | No |
| **Recomendado para** | General | Cortex-M4/M7 | Arduino Uno/Nano |
| **Overhead de Código** | Bajo | Medio | Bajo |
| **Complejidad** | Baja | Media | Baja |

### Métricas de Rendimiento Típicas

| Métrica | Float32 Original | Per-Layer (int8) | CMSIS-NN | Híbrido AVR |
|---------|------------------|------------------|----------|-------------|
| **Tamaño Modelo** | 100% | ~25% | ~25% | ~25% (pesos) |
| **Velocidad Inferencia** | 1x | 2-3x | 5-10x | 1.5-2x |
| **Pérdida Accuracy** | 0% | 0.5-2% | 0.5-2% | 0.5-2% |
| **RAM Usada** | Alta | Media | Media | Baja |
| **Flash Usada** | Baja | Media | Media | Alta |

---

## Limitaciones Actuales

### Limitaciones Técnicas

1. **Solo Redes Neuronales**
   - ✅ `MiniNeuralNetwork` soporta cuantificación completa
   - ❌ Otros modelos (DecisionTree, RandomForest, etc.) no soportan cuantificación

2. **Cuantificación Post-Entrenamiento**
   - ❌ No hay Quantization-Aware Training (QAT)
   - ❌ La cuantificación ocurre después del entrenamiento
   - ⚠️ Puede haber pérdida de precisión en modelos sensibles

3. **Per-Layer (No Per-Channel)**
   - ❌ Una escala por capa, no por canal
   - ⚠️ Menor precisión que per-channel para redes grandes
   - ✅ Suficiente para redes pequeñas-medianas

4. **Activaciones en Float (Modo Nativo)**
   - ⚠️ En `to_arduino_code()`, las activaciones se calculan en float
   - ✅ Solo CMSIS-NN usa activaciones int8 puras
   - ⚠️ Requiere FPU o emulación para modo híbrido

5. **Limitado a MLP (2 Capas)**
   - ❌ Solo soporta arquitecturas de 2 capas ocultas
   - ❌ No soporta redes profundas (3+ capas)
   - ✅ Suficiente para la mayoría de casos embebidos

6. **Calibración Requiere Dataset**
   - ⚠️ `calibrate()` necesita dataset completo
   - ⚠️ No hay calibración con dataset reducido
   - ✅ Se ejecuta automáticamente después de `fit()`

### Limitaciones de Hardware

1. **CMSIS-NN**: Solo ARM Cortex-M
2. **Modo Híbrido**: Requiere FPU o emulación de float
3. **PROGMEM**: Solo disponible en AVR

---

## Recomendaciones para Proyectos Embebidos

### ¿Qué Método Usar?

#### 🏆 **Recomendación Principal: CMSIS-NN (si está disponible)**

**Para**: ARM Cortex-M4, M7, M33, M55
- ✅ Máxima velocidad y eficiencia
- ✅ Inferencia completamente en int8
- ✅ Bajo consumo de energía
- ✅ Kernels optimizados con SIMD

**Ejemplo de Uso**:
```python
# Entrenar y exportar
model = MiniNeuralNetwork(n_inputs=2, n_hidden=4, n_outputs=1)
model.fit(dataset)
# Cuantificación automática en export_to_c()
code = ml_manager.export_to_c("my_model")
```

#### 🥈 **Segunda Opción: Modo Híbrido AVR**

**Para**: Arduino Uno, Nano, y otros AVR 8-bit
- ✅ Ahorra SRAM (datos en Flash)
- ✅ Portátil (sin dependencias)
- ✅ Compatible con hardware limitado

**Ejemplo de Uso**:
```python
# Similar al anterior, pero export_to_c() usará to_arduino_code()
# si CMSISAdapter no está disponible
code = ml_manager.export_to_c("my_model")
```

#### 🥉 **Tercera Opción: Per-Layer Nativo**

**Para**: Proyectos que requieren máxima portabilidad
- ✅ Sin dependencias externas
- ✅ Funciona en cualquier plataforma
- ⚠️ Menor velocidad que CMSIS-NN

### Guía de Selección por Hardware

| Hardware | MCU | RAM | Flash | FPU | Recomendación |
|----------|-----|-----|-------|-----|---------------|
| **STM32F4** | Cortex-M4 | 192KB | 1MB | Sí | CMSIS-NN |
| **STM32F7** | Cortex-M7 | 512KB | 2MB | Sí | CMSIS-NN |
| **Arduino Uno** | AVR | 2KB | 32KB | No | Modo Híbrido |
| **ESP32** | Xtensa | 520KB | 4MB | Sí | CMSIS-NN (si portado) |
| **Raspberry Pi Pico** | Cortex-M0+ | 264KB | 2MB | No | Modo Híbrido |

### Mejores Prácticas

1. **Siempre Calibrar con Dataset Representativo**
   ```python
   # Usar el mismo dataset de entrenamiento
   model.fit(training_dataset)  # Calibra automáticamente
   ```

2. **Validar Precisión Post-Cuantificación**
   ```python
   # Comparar accuracy antes y después
   accuracy_before = evaluate(model, test_set)
   model.quantize()
   accuracy_after = evaluate_quantized(model, test_set)
   assert accuracy_after >= accuracy_before - 0.02  # Tolerancia 2%
   ```

3. **Usar Escalado de Inputs**
   ```python
   # El scaler ayuda a mantener rangos consistentes
   ml_manager.train_pipeline(
       model_name="model",
       dataset=data,
       model_type="neural_network",
       scaling="minmax"  # Recomendado
   )
   ```

4. **Optimizar Arquitectura para Cuantificación**
   - Usar activaciones ReLU (más amigables a cuantificación)
   - Evitar capas muy profundas
   - Limitar rango de pesos durante entrenamiento

---

## Ejemplos de Uso

### Ejemplo 1: Entrenamiento y Exportación Completa

```python
from miniml import ml_manager

# Dataset de ejemplo (XOR)
dataset = [
    [0.0, 0.0, 0],
    [0.0, 1.0, 1],
    [1.0, 0.0, 1],
    [1.0, 1.0, 0]
]

# Entrenar con escalado
result = ml_manager.train_pipeline(
    model_name="xor_nn",
    dataset=dataset,
    model_type="neural_network",
    params={
        "n_inputs": 2,
        "n_hidden": 4,
        "n_outputs": 1,
        "epochs": 2000,
        "learning_rate": 0.1
    },
    scaling="minmax"
)

# Exportar a C (cuantificación automática)
c_code = ml_manager.export_to_c("xor_nn")

# Guardar código
with open("xor_model.h", "w") as f:
    f.write(c_code)
```

### Ejemplo 2: Cuantificación Manual

```python
from miniml.ml_runtime import MiniNeuralNetwork

# Crear y entrenar modelo
model = MiniNeuralNetwork(n_inputs=2, n_hidden=4, n_outputs=1)
model.fit(dataset)

# Calibración (automática después de fit, pero se puede hacer manual)
model.calibrate(dataset)

# Cuantificación explícita
model.quantize(per_channel=True)

# Verificar cuantificación
print(f"Quantized: {model.quantized}")
print(f"Act scales: {model.act_scales}")
print(f"W1 shape: {len(model.q_W1)}x{len(model.q_W1[0])}")
```

### Ejemplo 3: Uso de CMSISAdapter Directamente

```python
from miniml.ml_runtime import MiniNeuralNetwork
from adapters.cmsis_nn.adapter import CMSISAdapter

# Entrenar modelo
model = MiniNeuralNetwork(n_inputs=2, n_hidden=4, n_outputs=1)
model.fit(dataset)
model.quantize()

# Generar código CMSIS-NN
adapter = CMSISAdapter(model)
adapter.generate_c("model_cmsis.h")
```

### Ejemplo 4: Guardar y Cargar Modelo Cuantificado

```python
# Guardar modelo (incluye act_scales y pesos cuantificados)
ml_manager.save_model("xor_nn", "xor_nn.json")

# Cargar modelo (restaura act_scales)
ml_manager.load_model("xor_nn", "xor_nn.json")

# Re-cuantificar si es necesario
model = ml_manager.get_model("xor_nn")
if not model.quantized:
    model.quantize()

# Exportar
c_code = ml_manager.export_to_c("xor_nn")
```

---

## Conclusión

MiniML ofrece un sistema de cuantificación robusto y flexible para redes neuronales, con tres modos principales adaptados a diferentes plataformas:

1. **CMSIS-NN**: Máxima velocidad para ARM Cortex-M
2. **Modo Híbrido AVR**: Ideal para Arduino 8-bit
3. **Per-Layer Nativo**: Portátil y universal

La cuantificación reduce el tamaño del modelo en ~75% y acelera la inferencia 2-10x, con una pérdida típica de precisión < 2%, haciéndola ideal para aplicaciones embebidas de IA.

---

**Última actualización**: 2024
**Versión MiniML**: 1.0.0
