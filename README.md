<p align="center">
  <img src="assets/miniml_banner.png" alt="MiniML Engine Banner" width="850">
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.7%2B-blue.svg?style=flat-square&logo=python&logoColor=white" alt="Python Version"></a>
  <a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-red.svg?style=flat-square" alt="License: Apache 2.0"></a>
  <img src="https://img.shields.io/badge/Dependencies-Zero-brightgreen.svg?style=flat-square" alt="Zero Dependencies">
  <img src="https://img.shields.io/badge/C%2B%2B-PROGMEM%20Ready-orange.svg?style=flat-square&logo=c%2B%2B&logoColor=white" alt="C++ Ready">
  <img src="https://img.shields.io/badge/Platform-Arduino%20%7C%20ESP32%20%7C%20STM32-lightgrey.svg?style=flat-square" alt="Supported Platforms">
  <img src="https://img.shields.io/badge/Edge%20AI-Deep%20Learning-purple.svg?style=flat-square" alt="Edge AI Deep Learning">
</p>

---

# **🧠 MiniML Engine (Powered by MiniTensor)**

**Version:** 1.1.0 - *Industrial Edge AI Release* 
**Architecture:** Zero-Dependency Python Core + Autograd Engine + Native C++ Export (PROGMEM)

**Philosophy:** "Train on PC, Run on Metal"

**Author:** Wilner Manzanares (Michego Takoro 'Shuuida')

---

## **📋 Overview**

**MiniML Engine** is an industrial-grade Machine Learning and Deep Learning framework explicitly designed for **extreme low-cost, resource-constrained embedded systems** (Arduino, ESP32, STM32, 8-bit AVR microcontrollers).

Defying industry standards, MiniML operates with **zero external dependencies**. No NumPy, no SciPy, no PyTorch, and no TensorFlow. The entire ecosystem—from linear algebra and decision trees to the N-Dimensional computational graph and Backpropagation—is written in pure Python.

This architecture guarantees absolute transparency and enables a mathematically perfect (1:1) translation into static, optimized C++ code tailored to run on microcontrollers with **less than 2KB of RAM**.

### **Core Value Proposition (USP)**

* **🚫 Zero-Dependency Core & Autograd:** Train everything from simple linear regressions to deep neural networks using only the Python standard library.
* **⚡ Extreme Optimization (SRAM < 2KB):** Algorithms are reverse-engineered to flatten dynamic structures and avoid recursion. Aggressive use of Flash memory (`PROGMEM`).
* **🧠 Hybrid INT8 Quantization:** Converts 32-bit float weights into 8-bit integers for storage, de-quantizing on-the-fly during inference to maintain mathematical precision.
* **📦 Industrial Packaging:** Exports native libraries with dual manifests (`library.json` and `library.properties`), ready to be compiled in **PlatformIO** and **Arduino IDE**.

---

## **📂 Model Ecosystem: Legacy ML & Deep Learning**

MiniML Engine features two distinct inference engines, allowing the software architect to choose the perfect approach based on the task's complexity and the hardware's constraints.

### **1. Classic MiniML (Legacy Machine Learning)**

Located in `ml_runtime.py`, this engine is the core for tabular data and simple analog signals. It is ideal for low-frequency sensors where a neural network would be a waste of clock cycles.

| Model Class | Algorithm & Edge Optimization (C++) |
| --- | --- |
| **`DecisionTreeClassifier`** | *CART (Gini Impurity)*. Trees are flattened into linear arrays (`feature_index`, `threshold`) to allow **O(1)** stack usage via continuous `while` loops. |
| **`RandomForestClassifier`** | *Bagging*. Generates independent C functions stored in Flash memory for each tree and a lightweight **Majority Vote** function in SRAM. |
| **`MiniLinearModel`** | *SGD*. Uses iterative updates. Exports a simple float array for ultra-fast dot-product inference. |
| **`MiniSVM`** | *Hinge Loss*. Implements a linear decision boundary, perfect for binary classification on hardware lacking a dedicated Floating-Point Unit (FPU). |
| **`KNearestNeighbors`** | *Lazy Learning*. Exports the entire dataset to Flash memory. Implements an in-place *Insertion Sort* simulating a priority queue to avoid exhausting dynamic RAM. |
| **`MiniScaler`** | *Preprocessing*. Generates MinMax/Standard normalization routines in C++ (`preprocess_data()`) to stabilize sensor signals prior to inference. |

### **2. MiniTensor (Embedded Deep Learning)**

Located in the `tensor.py` and `layers.py` modules, this is the automatic differentiation (*Autograd*) engine for extracting complex features from time-series, audio, and basic computer vision.

| Layer / Module | Description & C++ Inference Optimization |
| --- | --- |
| **`Conv1D` & `Conv2D`** | Spatial/temporal convolutions. The exporter dynamically indexes the geometry and generates nested loops with safe memory reads using `READ_FLOAT` macros. |
| **`SeparableConv2D`** | *MobileNet-Style*. Splits the convolution into *Depthwise* and *Pointwise* steps. Supports native **Operator Fusion** in C++ to minimize memory access and accelerate inference. |
| **`ResidualBlock1D`** | *ResNet-Style*. Enables deep networks without vanishing gradients by adding the identity ($y = \mathcal{F}(x) + x$). Implements strict geometric indexing to align dimensions in C++. |
| **`MaxPool1D/2D`** | Mathematical dimensionality reduction with sliding window management (Kernel/Stride). |
| **`Flatten` & `Linear`** | Multilayer Perceptron. Supports recursive flattening of dynamic tensors (up to 4D) while maintaining the gradient flow. |
| **Activations** | `ReLU`, `Sigmoid`, `MSELoss`, `CrossEntropyLoss`. Implemented with *Clip* barriers to prevent mathematical *Overflows* in 8-bit architectures. |

---

### **⚙️ System Architecture (The Pipeline Legacy)**

The framework ensures the Separation of Concerns (SoC) principle through its internal managers:

### **\. ml\_manager.py (The Orchestrator)**

The high-level API that unifies the workflow. It acts as a bridge between the user and the raw algorithms.

* **Intelligent Dual-Core:** Checks for sklearn. If present, it uses it for high-speed training on the PC. If not, it seamlessly switches to ml\_runtime.  
* **Automated Pipeline:**  
  1. **Imputation:** Fills missing values (NaN) to prevent crashes.  
  2. **Scaling:** Normalizes data using MiniScaler.  
  3. **Training:** Fits the selected model.  
* **predict() Polymorphism:** Automatically handles raw input, applies the saved scaler, and runs inference.

### **\. ml\_compat.py (Safety & Compatibility)**

The data guardian. It ensures that the dynamic nature of Python does not break the strict static nature of C.

* **\_flatten\_tree\_to\_arrays():** The most critical function for tree-based models. It traverses the Python dictionary tree structure and serializes it into parallel arrays (C-style), enabling the iterative execution logic required for microcontrollers.  
* **check\_dims():** strictly validates input dimensions before prediction, preventing index out-of-bounds errors in the generated C code.  
* **impute\_missing\_values():** Ensures data integrity before it reaches the mathematical core.

### **\. ml\_factory.py (The Factory Pattern)**

Decouples model instantiation from the logic flow.

* **Function:** create\_model(type\_string, params\_dict)  
* **Purpose:** Allows the system to instantiate complex objects (like RandomForestRegressor) from simple JSON strings. This is vital for the Save/Load system and prevents circular dependencies between modules.

### *5\. ml\_exporter.py (Serialization & Export)**

Handles the persistence and translation of models.

* **Structure Extraction:** Instead of using Python's pickle (which is insecure and Python-specific), this module extracts the pure mathematical structure (weights, thresholds, topology) into a language-agnostic JSON format.  
* **Sklearn Interop:** If a model was trained using scikit-learn, this module extracts the internal NumPy arrays (tree\_.value, coef\_) and converts them into the MiniML standard format, allowing you to **export Sklearn models to Arduino C**.

---

## **🛠️ Tooling & MLOps for Edge**

* **Architecture Analysis (CLI):** Prints detailed summaries of the model topology, output dimensions, and trainable parameter counts in the terminal.
* **Memory Estimator:** Mathematically projects the exact **SRAM** and **Flash memory** footprint before compilation, preventing physical microcontroller crashes.
* **Deterministic Simulation:** The generated C++ code provides perfect mathematical parity (1:1) with the Python engine, exhaustively validated against instruction-level hardware emulators (like Wokwi).

---

## **🚀 Expanded Use Cases (Industrial IoT & Robotics)**

1. **Predictive Maintenance (Acoustics/Vibration):** Using `Conv1D` and `ResidualBlock1D`, a microcontroller can analyze time-windows from an accelerometer to detect anomalies in industrial motors locally (Edge Computing).
2. **Sensor Fusion (Soft-Sensors):** By combining low-cost sensors (e.g., DHT11, LDR) and processing them through a **Random Forest** or a **Quantized MLP**, the MCU can predict complex variables without requiring an internet connection.
3. **Tiny Vision:** Employing `SeparableConv2D`, it is possible to train pattern classifiers for low-resolution arrays (e.g., thermal or small optical cameras), drastically reducing computational load.

---

### **The fit() Difference**

**Crucial:** MiniML uses a unified dataset format for fit(), unlike Scikit-learn.

* **Sklearn:** fit(X, y) (Two separate arrays).  
* **MiniML:** fit(dataset) (One list of lists, where the **last column** is the target).

### **Real-World Workflow Example (Sensor to Arduino)**

import miniml

\# 1\. Dataset (3 features from sensors, last column is class)  
\# \[Temperature, Humidity, Light\_Level, CLASS\]  
data \= \[  
    \[25.0, 60.0, 100, 0\], \# Normal  
    \[26.0, 62.0, 150, 0\],  
    \[80.0, 20.0, 800, 1\], \# Fire Danger  
    \[85.0, 15.0, 900, 1\]  
\]

\# 2\. Train Pipeline (Handles scaling automatically)  
print("Training model...")  
result \= miniml.train\_pipeline(  
    model\_name="fire\_detector",  
    dataset=data,  
    model\_type="DecisionTreeClassifier",  
    params={"max\_depth": 3},  
    scaling="minmax" \# Crucial for sensor data normalization  
)

\# 3\. Predict on PC (Sanity Check)  
\# Input is raw sensor data. MiniML scales it automatically before prediction.  
sensor\_input \= \[\[82.0, 18.0, 850\]\]   
prediction \= miniml.predict("fire\_detector", sensor\_input)  
print(f"Prediction (0=Safe, 1=Danger): {prediction}") 

\# 4\. Export to Firmware  
print("Generating C code...")  
c\_code \= miniml.export\_to\_c("fire\_detector")

\# 5\. Save to file  
with open("model.h", "w") as f:  
    f.write(c\_code)

## **💾 Generated C Code (Artifact)**

The output is standard C99 code, ready to be included in an Arduino sketch (\#include "model.h").

// MiniML Export: fire\_detector  
// Preprocessing (MinMax Scaler baked in)  
void preprocess\_data(float row\[\]) {  
  // Hardcoded values from training phase  
  row\[0\] \= (row\[0\] \- 25.0) / 60.0;   
  row\[1\] \= (row\[1\] \- 15.0) / 47.0;   
  row\[2\] \= (row\[2\] \- 100.0) / 800.0;  
}

// Model Arrays (Flattened Tree)  
const int tree\_feature\_index\[\] \= {0, 2, \-1, \-1, \-1};  
const float tree\_threshold\[\] \= {0.5, 0.8, 0.0, 0.0, 0.0};  
const int tree\_left\[\] \= {1, 3, \-1, \-1, \-1};  
const int tree\_right\[\] \= {2, 4, \-1, \-1, \-1};  
const int tree\_value\[\] \= {0, 0, 0, 1, 0}; // 0=Safe, 1=Danger

// Inference Function (Iterative \- Stack Safe)  
int predict\_model(float row\[\]) {  
  int node\_index \= 0;  
  while (tree\_feature\_index\[node\_index\] \!= \-1) {  
     if (row\[tree\_feature\_index\[node\_index\]\] \<= tree\_threshold\[node\_index\]) {  
        node\_index \= tree\_left\[node\_index\];  
     } else {  
        node\_index \= tree\_right\[node\_index\];  
     }  
  }  
  return tree\_value\[node\_index\];  
}

// Unified Entry Point  
float predict(float inputs\[\]) {  
  preprocess\_data(inputs); // Modifies in-place  
  return (float)predict\_model(inputs);  
}

---

## **💻 MiniTensor End-to-End Workflow Example**

From Python training to PlatformIO packaging.

```python
import miniml
from miniml import Tensor, nn, optim
from miniml.exporters.library_packer import LibraryPackager

# 1. Define Architecture (MiniTensor Deep Learning)
model = nn.Sequential([
    nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
    nn.MaxPool1d(kernel_size=2),
    nn.ResidualBlock1D(in_channels=4, out_channels=4),
    nn.Flatten(),
    nn.Linear(16, 1),
    nn.Sigmoid()
])

# 2. Train on PC (Zero-Dependency Autograd Engine)
optimizer = optim.SGD(model.parameters(), lr=0.01)
# ... standard training loop ...

# 3. Edge Optimization (Hybrid INT8 Quantization)
model.quantize()

# 4. Export to Native C++
from miniml.exporters import cpp_writer
cpp_code = cpp_writer.generate_cpp_code(model, input_shape=(1, 1, 8))

# 5. Package Industrial Library (PlatformIO / Arduino)
LibraryPackager.create_arduino_zip(
    model_name="ResNet_Vibration_Detector",
    cpp_code=cpp_code,
    version="2.0.0",
    quantized=True
)
print("ZIP library successfully generated, structured, and ready to compile on hardware!")

```

---

## **🛠️ Installation & Usage**

### **Installation**

Since MiniML is a pure Python package, installation is straightforward:

pip install miniml

*(Optional: Install scikit-learn for faster training on PC, but it is NOT required).*

---

## **🤝 Contributing**

Contributions are highly welcome. MiniML's mission is to strictly maintain its "zero dependencies" philosophy and low-level optimization.

1. Fork the Project.
2. Create your Feature Branch (`git checkout -b feature/NewOptimization`).
3. Commit your Changes (`git commit -m 'Added new feature'`).
4. Push to the Branch (`git push origin feature/NewOptimization`).
5. Open a Pull Request.

## **📄 License and Documentation**

Distributed under the **Apache License 2.0**. Free for academic, commercial, and industrial use, guaranteeing open-source traceability. See the `LICENSE` file for more information.

The documentation is under development. Documentation on how to use the library, how to take advantage of Arduino's memory safety features, and its syntax will be uploaded routinely to the docs folder. Thank you for your patience!

---

# **🧠 MiniML Engine (Powered by MiniTensor)**

**Versión:** 1.1.0 - *Industrial Edge AI Release* 
**Arquitectura:** Núcleo de Python (Zero-Dependency) + Motor Autograd + Exportación a C++ Nativo (PROGMEM)

**Filosofía:** "Train on PC, Run on Metal"

**Autor:** Wilner Manzanares (Michego Takoro 'Shuuida')

---

## **📋 Resumen (Overview)**

**MiniML Engine** es un framework de Machine Learning y Deep Learning diseñado explícitamente para **sistemas embebidos de bajo costo y recursos extremos** (Arduino, ESP32, STM32, AVR de 8-bits).

Desafiando los estándares de la industria, MiniML opera con **cero dependencias externas**. Sin NumPy, sin SciPy, sin PyTorch ni TensorFlow. Todo el ecosistema —desde el álgebra lineal y los árboles de decisión, hasta el grafo computacional N-Dimensional y la propagación hacia atrás (*Backpropagation*)— está escrito en Python puro.

Esta arquitectura garantiza una transparencia absoluta y permite una traducción matemática perfecta (1:1) hacia código C++ estático, optimizado para ejecutarse en microcontroladores con **menos de 2KB de RAM**.

### **Propuesta de Valor Principal (USP)**

* **🚫 Zero-Dependency Core & Autograd:** Entrena desde regresiones simples hasta redes neuronales profundas usando solo la biblioteca estándar de Python.
* **⚡ Optimización Extrema (SRAM < 2KB):** Los algoritmos son diseñados a la inversa (*reverse-engineered*) para aplanar estructuras dinámicas y evitar la recursividad. Uso agresivo de memoria Flash (`PROGMEM`).
* **🧠 Cuantificación INT8 Híbrida:** Convierte pesos de flotantes de 32 bits a enteros de 8 bits para almacenamiento, de-cuantizando al vuelo durante la inferencia para mantener la precisión matemática.
* **📦 Empaquetado Industrial:** Exporta librerías nativas con manifiestos duales (`library.json` y `library.properties`) listas para compilar en **PlatformIO** y **Arduino IDE**.

---

## **📂 Ecosistema de Modelos: Legacy ML & Deep Learning**

MiniML Engine ofrece dos motores de inferencia, permitiendo al arquitecto de software elegir el enfoque perfecto según la tarea y el hardware.

### **1. MiniML Clásico (Legacy Machine Learning)**

Ubicado en `ml_runtime.py`, este motor es el corazón para datos tabulares y señales analógicas simples. Ideal para sensores de baja frecuencia donde una red neuronal sería un desperdicio de ciclos de reloj.

| Clase de Modelo | Algoritmo y Optimización en Edge (C++) |
| --- | --- |
| **`DecisionTreeClassifier`** | *CART (Impureza Gini)*. Los árboles se aplanan en arreglos lineales (`feature_index`, `threshold`) para permitir un uso de pila de **O(1)** mediante bucles `while` continuos. |
| **`RandomForestClassifier`** | *Bagging*. Genera funciones C independientes guardadas en Flash para cada árbol y una función ligera de **Voto Mayoritario** en SRAM. |
| **`MiniLinearModel`** | *SGD*. Utiliza actualizaciones iterativas. Exporta un arreglo simple de flotantes para una inferencia ultra-rápida por producto punto. |
| **`MiniSVM`** | *Hinge Loss*. Implementa un límite de decisión lineal, perfecto para clasificación binaria en hardware sin unidad de coma flotante (FPU) dedicada. |
| **`KNearestNeighbors`** | *Lazy Learning*. Exporta el dataset completo a Flash. Implementa un *Insertion Sort* in-place simulando una cola de prioridad para no agotar la RAM dinámica. |
| **`MiniScaler`** | *Preprocesamiento*. Genera rutinas de normalización MinMax/Standard en C++ (`preprocess_data()`) para estabilizar las señales de los sensores antes de la inferencia. |

### **2. MiniTensor (Embedded Deep Learning)**

Ubicado en los módulos `tensor.py` y `layers.py`, este es el motor de diferenciación automática (*Autograd*) para la extracción de características complejas en series temporales, audio y visión artificial básica.

| Capa / Módulo | Descripción y Optimización de Inferencia en C++ |
| --- | --- |
| **`Conv1D` & `Conv2D`** | Convoluciones espaciales/temporales. El exportador indexa la geometría dinámicamente y genera bucles anidados con lectura segura mediante macros `READ_FLOAT`. |
| **`SeparableConv2D`** | *MobileNet-Style*. Divide la convolución en *Depthwise* y *Pointwise*. Soporta **Operator Fusion** nativo en C++ para minimizar el acceso a memoria y acelerar la inferencia. |
| **`ResidualBlock1D`** | *ResNet-Style*. Permite redes profundas sin desvanecimiento de gradiente sumando la identidad ($y = \mathcal{F}(x) + x$). Implementa indexación geométrica estricta para alinear las dimensiones en C++. |
| **`MaxPool1D/2D`** | Reducción de dimensionalidad matemática con gestión de ventanas deslizantes (Kernel/Stride). |
| **`Flatten` & `Linear`** | Perceptrón Multicapa. Soporta el aplanado recursivo de tensores dinámicos (hasta 4D) manteniendo el flujo del gradiente. |
| **Activaciones** | `ReLU`, `Sigmoid`, `MSELoss`, `CrossEntropyLoss`. Implementadas con barreras de *Clip* para evitar *Overflows* matemáticos en arquitecturas de 8-bits. |

---

## **⚙️ Arquitectura del Sistema (El Pipeline)**

El framework asegura el principio de Separación de Responsabilidades a través de sus gestores internos:

### **\. ml\_manager.py (El Orquestador)**

La API de alto nivel que unifica el flujo de trabajo. Actúa como un puente entre el usuario y los algoritmos base.

* **Doble Núcleo Inteligente:** Verifica la presencia de **sklearn**. Si está, lo usa para un entrenamiento de alta velocidad en el PC. Si no, cambia sin problemas a **ml\_runtime**.  
* **Pipeline Automatizado:**  
  1. **Imputación:** Rellena los valores faltantes (NaN) para prevenir fallos.  
  2. **Escalado:** Normaliza los datos usando **MiniScaler**.  
  3. **Entrenamiento:** Ajusta el modelo seleccionado.  
* **Polimorfismo de predict():** Maneja automáticamente la entrada cruda, aplica el escalador guardado y ejecuta la inferencia.

---

### **\. ml\_compat.py (Seguridad y Compatibilidad)**

El guardián de los datos. Asegura que la naturaleza dinámica de Python no rompa la estricta naturaleza estática de C.

* **\_flatten\_tree\_to\_arrays():** La función más crítica para modelos basados en árboles. Recorre la estructura de árbol del diccionario de Python y la serializa en arrays paralelos (estilo C), habilitando la lógica de ejecución iterativa requerida para microcontroladores.  
* **check\_dims():** Valida estrictamente las dimensiones de entrada antes de la predicción, previniendo errores de índice fuera de límites en el código C generado.  
* **impute\_missing\_values():** Asegura la integridad de los datos antes de que lleguen al núcleo matemático.

---

### **\. ml\_factory.py (El Patrón Factory)**

Desacopla la instanciación del modelo del flujo de lógica.

* **Función:** $\\text{create\\\_model}(\\text{type\\\_string}, \\text{params\\\_dict})$  
* **Propósito:** Permite al sistema instanciar objetos complejos (como RandomForestRegressor) a partir de simples cadenas JSON. Esto es vital para el sistema de Guardar/Cargar y previene dependencias circulares entre módulos.

---

### **\. ml\_exporter.py (Serialización y Exportación)**

Maneja la persistencia y traducción de modelos.

* **Extracción de Estructura:** En lugar de usar **pickle** de Python (que es inseguro y específico de Python), este módulo extrae la estructura matemática pura (pesos, umbrales, topología) a un **formato JSON agnóstico al lenguaje**.  
* **Interoperabilidad con Sklearn:** Si un modelo fue entrenado usando **scikit-learn**, este módulo extrae los arrays internos de NumPy ($\\text{tree\\\_value, coef\\\_}$) y los convierte al formato estándar de MiniML, permitiendo **exportar modelos de Sklearn a C de Arduino**.

---

## **🛠️ Tooling & MLOps para Edge**

* **Análisis de Arquitectura (CLI):** Imprime resúmenes detallados de la topología del modelo, dimensiones de salida y conteo de parámetros entrenables.
* **Estimador de Memoria:** Proyecta matemáticamente el consumo exacto de **SRAM** y **Memoria Flash** antes de compilar, previniendo reinicios (crashes) en el microcontrolador físico.
* **Simulación Determinista:** El código generado ofrece paridad matemática perfecta (1:1) con el motor de Python, validado contra emuladores de hardware a nivel de instrucciones (como Wokwi).

---

## **🚀 Casos de Uso Expandidos (IoT Industrial & Robótica)**

1. **Mantenimiento Predictivo (Acústica/Vibración):** Utilizando `Conv1D` y `ResidualBlock1D`, un microcontrolador puede analizar ventanas temporales de un acelerómetro y detectar anomalías en motores industriales de forma local (Edge Computing).
2. **Fusión de Sensores (Soft-Sensors):** Combinando sensores de bajo costo (ej. DTH11, LDR) y procesándolos mediante un **Random Forest** o un **MLP Cuantizado**, el MCU puede predecir variables complejas sin requerir conexión a internet.
3. **Tiny Vision:** Empleando `SeparableConv2D`, es posible entrenar clasificadores de patrones para matrices de baja resolución (ej. cámaras térmicas u ópticas pequeñas) reduciendo drásticamente la carga computacional.

---

### **La Diferencia de fit()**

**Crucial:** MiniML utiliza un formato de conjunto de datos unificado para $\\text{fit}()$, a diferencia de Scikit-learn.

* **Sklearn:** $\\text{fit}(X, y)$ (Dos arrays separados).  
* **MiniML:** $\\text{fit}(\\text{dataset})$ (Una lista de listas, donde la **última columna** es el objetivo).

### **Ejemplo de Flujo de Trabajo en el Mundo Real (Sensor a Arduino)**

Python

import miniml

\# 1\. Conjunto de Datos (3 características de sensores, la última columna es la clase)

\# \[Temperatura, Humedad, Nivel\_Luz, CLASE\]

data \= \[

    \[25.0, 60.0, 100, 0\], \# Normal

    \[26.0, 62.0, 150, 0\],

    \[80.0, 20.0, 800, 1\], \# Peligro de Incendio

    \[85.0, 15.0, 900, 1\]

\]

\# 2\. Pipeline de Entrenamiento (Maneja el escalado automáticamente)

print("Entrenando modelo...")

result \= miniml.train\_pipeline(

    model\_name="fire\_detector",

    dataset=data,

    model\_type="DecisionTreeClassifier",

    params={"max\_depth": 3},

    scaling="minmax" \# Crucial para la normalización de datos de sensores

)

\# 3\. Predicción en PC (Verificación de Sanidad)

\# La entrada son datos de sensor crudos. MiniML los escala automáticamente antes de la predicción.

sensor\_input \= \[\[82.0, 18.0, 850\]\]

prediction \= miniml.predict("fire\_detector", sensor\_input)

print(f"Predicción (0=Seguro, 1=Peligro): {prediction}")

\# 4\. Exportar al Firmware

print("Generando código C...")

c\_code \= miniml.export\_to\_c("fire\_detector")

\# 5\. Guardar en archivo

with open("model.h", "w") as f:

    f.write(c\_code)

---

## **💾 Código C Generado (Artifacto)**

La salida es código C99 estándar, listo para ser incluido en un sketch de Arduino (\#include "model.h").

C

// Exportación MiniML: fire\_detector

// Preprocesamiento (Escalador MinMax incorporado)

void preprocess\_data(float row\[\]) {

  // Valores codificados (Hardcoded) de la fase de entrenamiento

  row\[0\] \= (row\[0\] \- 25.0) / 60.0;

  row\[1\] \= (row\[1\] \- 15.0) / 47.0;

  row\[2\] \= (row\[2\] \- 100.0) / 800.0;

}

// Arrays del Modelo (Árbol Aplanado)

const int tree\_feature\_index\[\] \= {0, 2, \-1, \-1, \-1};

const float tree\_threshold\[\] \= {0.5, 0.8, 0.0, 0.0, 0.0};

const int tree\_left\[\] \= {1, 3, \-1, \-1, \-1};

const int tree\_right\[\] \= {2, 4, \-1, \-1, \-1};

const int tree\_value\[\] \= {0, 0, 0, 1, 0}; // 0=Seguro, 1=Peligro

// Función de Inferencia (Iterativa \- Segura para la Pila)

int predict\_model(float row\[\]) {

  int node\_index \= 0;

  while (tree\_feature\_index\[node\_index\] \!= \-1) {

     if (row\[tree\_feature\_index\[node\_index\]\] \<= tree\_threshold\[node\_index\]) {

        node\_index \= tree\_left\[node\_index\];

     } else {

        node\_index \= tree\_right\[node\_index\];

     }

  }

  return tree\_value\[node\_index\];

}

// Punto de Entrada Unificado

float predict(float inputs\[\]) {

  preprocess\_data(inputs); // Modifica in-place (en el mismo lugar)

  return (float)predict\_model(inputs);

}

---

## **💻 Ejemplo de Flujo de Trabajo para MiniTensor (End-to-End)**

Desde el entrenamiento en Python hasta el empaquetado para PlatformIO.

```python
import miniml
from miniml import Tensor, nn, optim
from miniml.exporters.library_packer import LibraryPackager

# 1. Definir Arquitectura (MiniTensor Deep Learning)
model = nn.Sequential([
    nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
    nn.MaxPool1d(kernel_size=2),
    nn.ResidualBlock1D(in_channels=4, out_channels=4),
    nn.Flatten(),
    nn.Linear(16, 1),
    nn.Sigmoid()
])

# 2. Entrenar en PC (Motor Autograd)
optimizer = optim.SGD(model.parameters(), lr=0.01)
# ... bucle de entrenamiento estándar ...

# 3. Optimización para Edge (Cuantificación Híbrida INT8)
model.quantize()

# 4. Exportar a C++ Nativo
from miniml.exporters import cpp_writer
cpp_code = cpp_writer.generate_cpp_code(model, input_shape=(1, 1, 8))

# 5. Empaquetar Librería Industrial (PlatformIO / Arduino)
LibraryPackager.create_arduino_zip(
    model_name="ResNet_Vibration_Detector",
    cpp_code=cpp_code,
    version="2.0.0",
    quantized=True
)
print("¡Librería ZIP generada, estructurada y lista para compilar en hardware!")

```

---

## **🛠️ Instalación y Uso**

### **Instalación**

Dado que MiniML es un paquete de Python puro, la instalación es sencilla:

Bash

pip install miniml

*(Opcional: Instalar scikit-learn para un entrenamiento más rápido en PC, pero NO es un requisito).*

---

## **🤝 Contribuciones**

Las contribuciones son bienvenidas. MiniML tiene como misión mantener su filosofía estricta de "cero dependencias" y optimización a bajo nivel.

1. Haz un *Fork* del Proyecto.
2. Crea tu rama (`git checkout -b feature/NuevaOptimizacion`).
3. Confirma tus cambios (`git commit -m 'Añadida nueva característica'`).
4. Sube la rama (`git push origin feature/NuevaOptimizacion`).
5. Abre un *Pull Request*.

## **📄 Licencia y Documentación**

Distribuido bajo la Licencia **Apache License 2.0**. Libre para uso académico, comercial e industrial garantizando la trazabilidad open-source. Consulta el archivo `LICENSE` para más información.

La documentación está en desarrollo, dentro de la carpeta docs se irán subiendo de forma rutinaria la documentación sobre como usar la librería, como aprovechar el uso de seguridad de memoria en Arduino y su sintaxis. ¡Muchas gracias por la espera!