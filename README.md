# **🧠 MiniML Framework Documentation**

**Version:** 1.0.0

**Architecture:** Zero-Dependency Python Core \+ C Export for Embedded Systems

**Philosophy:** "Train on PC, Run on Metal"

**Author:** "Wilner Manzanares (Michego Takoro 'Shuuida')"

---

## **📋 Overview**

**MiniML** is a lightweight Machine Learning framework explicitly engineered for low-cost embedded systems (Arduino, ESP32, STM32). Unlike traditional frameworks that rely on heavy libraries like NumPy or Pandas, MiniML is built from scratch using pure Python to ensure total transparency and compatibility with the C code it generates.

### **Core Value Proposition (USP)**

* **🚫 Zero Dependencies:** No numpy, scipy, or pandas. Runs on any standard Python interpreter (including legacy systems).  
* **⚡ Embedded Optimization:** Algorithms are reverse-engineered to run on hardware with \< 2KB RAM.  
* **🔄 Dual-Core Engine:** Automatically accelerates training using scikit-learn if installed on the host PC, falling back to the pure Python ml\_runtime otherwise.

## **📂 Modular Architecture Analysis**

The framework operates through five cohesive modules, each respecting the Separation of Concerns (SoC) principle.

### **1\. ml\_runtime.py (The Mathematical Core)**

This is the engine room. It contains the pure Python implementations of ML algorithms. Every line of code here is written to mathematically mirror the C code that will run on the microcontroller.

**Key Features:**

* **MiniMatrixOps:** A custom linear algebra class that replaces NumPy. It handles dot products, transpositions, and matrix multiplications using native Python lists.  
* **Iterative Design:** Algorithms are designed to avoid deep recursion, preventing Stack Overflow on microcontrollers.

**Supported Models & Algorithms:**

| Model Class | Algorithm | Embedded Optimization (Reverse Engineering) |
| :---- | :---- | :---- |
| **DecisionTreeClassifier** | CART (Gini Impurity) | Trees are flattened into linear arrays (feature\_index\[\], threshold\[\]) to allow O(1) stack usage during inference via a while loop. |
| **RandomForestClassifier** | Bagging (Bootstrap Aggregation) | Generates independent C functions for each tree and a lightweight "Majority Vote" function in C. |
| **MiniLinearModel** | Stochastic Gradient Descent (SGD) | Uses iterative weight updates. Exports a simple float array weights\[\] for dot-product inference. |
| **MiniSVM** | Linear SVM (Hinge Loss) | Implements a linear decision boundary perfect for binary classification on hardware with limited FPU. |
| **MiniNeuralNetwork** | MLP (Backpropagation) | Supports **Quantization**. Can convert 32-bit float weights to 8-bit integers, reducing model size by \~75% for Flash memory storage. |
| **KNearestNeighbors** | Euclidean Distance (Lazy) | **⚠️ Warning:** Exports the *entire* training dataset as a const C array. High Flash memory consumption. |
| **MiniScaler** | MinMax / Standard Scaling | Records statistics (min/max/mean/std) during training to generate a C function preprocess\_data() that normalizes sensor inputs in real-time. |

### **2\. ml\_manager.py (The Orchestrator)**

The high-level API that unifies the workflow. It acts as a bridge between the user and the raw algorithms.

* **Intelligent Dual-Core:** Checks for sklearn. If present, it uses it for high-speed training on the PC. If not, it seamlessly switches to ml\_runtime.  
* **Automated Pipeline:**  
  1. **Imputation:** Fills missing values (NaN) to prevent crashes.  
  2. **Scaling:** Normalizes data using MiniScaler.  
  3. **Training:** Fits the selected model.  
* **predict() Polymorphism:** Automatically handles raw input, applies the saved scaler, and runs inference.

### **3\. ml\_compat.py (Safety & Compatibility)**

The data guardian. It ensures that the dynamic nature of Python does not break the strict static nature of C.

* **\_flatten\_tree\_to\_arrays():** The most critical function for tree-based models. It traverses the Python dictionary tree structure and serializes it into parallel arrays (C-style), enabling the iterative execution logic required for microcontrollers.  
* **check\_dims():** strictly validates input dimensions before prediction, preventing index out-of-bounds errors in the generated C code.  
* **impute\_missing\_values():** Ensures data integrity before it reaches the mathematical core.

### **4\. ml\_factory.py (The Factory Pattern)**

Decouples model instantiation from the logic flow.

* **Function:** create\_model(type\_string, params\_dict)  
* **Purpose:** Allows the system to instantiate complex objects (like RandomForestRegressor) from simple JSON strings. This is vital for the Save/Load system and prevents circular dependencies between modules.

### **5\. ml\_exporter.py (Serialization & Export)**

Handles the persistence and translation of models.

* **Structure Extraction:** Instead of using Python's pickle (which is insecure and Python-specific), this module extracts the pure mathematical structure (weights, thresholds, topology) into a language-agnostic JSON format.  
* **Sklearn Interop:** If a model was trained using scikit-learn, this module extracts the internal NumPy arrays (tree\_.value, coef\_) and converts them into the MiniML standard format, allowing you to **export Sklearn models to Arduino C**.

## **🛠️ Installation & Usage**

### **Installation**

Since MiniML is a pure Python package, installation is straightforward:

pip install miniml

*(Optional: Install scikit-learn for faster training on PC, but it is NOT required).*

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

## **🤝 Contributing**

Contributions are welcome\! MiniML aims to maintain its "zero dependency" philosophy.

1. Fork the Project.  
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`).  
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`).  
4. Push to the Branch (`git push origin feature/AmazingFeature`).  
5. Open a Pull Request.

---

## **📄 License**

Distributed under the MIT License. See `LICENSE` for more information.

---

# **🧠 Documentación del Framework MiniML**

**Versión:** 1.0.0

**Arquitectura:** Núcleo de Python con Cero Dependencias \+ Exportación a C para Sistemas Embebidos

**Filosofía:** "Entrenar en PC, Ejecutar en el Hardware (Run on Metal)"

**Autor:** "Wilner Manzanares (Michego Takoro 'Shuuida')"

---

## **📋 Resumen (Overview)**

**MiniML** es un framework de Machine Learning ligero, diseñado explícitamente para **sistemas embebidos de bajo costo** (Arduino, ESP32, STM32). A diferencia de los frameworks tradicionales que dependen de librerías pesadas como NumPy o Pandas, MiniML se construye desde cero utilizando **Python puro** para asegurar total transparencia y compatibilidad con el código C que genera.

### **Propuesta de Valor Principal (USP)**

* **🚫 Cero Dependencias:** No utiliza numpy, scipy o pandas. Se ejecuta en cualquier intérprete de Python estándar (incluyendo sistemas heredados/legacy).  
* **⚡ Optimización Embebida:** Los algoritmos son diseñados a la inversa (*reverse-engineered*) para funcionar en hardware con **menos de 2KB de RAM**.  
* **🔄 Motor de Doble Núcleo:** Acelera automáticamente el entrenamiento utilizando **scikit-learn** si está instalado en el PC anfitrión, volviendo al **ml\_runtime** de Python puro en caso contrario.

---

## **📂 Análisis de la Arquitectura Modular**

El framework opera a través de cinco módulos cohesionados, cada uno respetando el principio de la **Separación de Responsabilidades (SoC)**.

### **1\. ml\_runtime.py (El Núcleo Matemático)**

Es la sala de máquinas. Contiene las implementaciones de algoritmos de ML en Python puro. Cada línea de código aquí está escrita para reflejar matemáticamente el código C que se ejecutará en el microcontrolador.

**Características Clave:**

* **MiniMatrixOps:** Una clase de álgebra lineal personalizada que reemplaza a NumPy. Maneja productos de puntos, traspuestas y multiplicaciones de matrices usando **listas nativas de Python**.  
* **Diseño Iterativo:** Los algoritmos están diseñados para evitar la recursión profunda, previniendo el **Stack Overflow** en microcontroladores.

**Modelos y Algoritmos Soportados:**

| Clase de Modelo | Algoritmo | Optimización Embebida (Diseño Inverso) |
| :---- | :---- | :---- |
| **DecisionTreeClassifier** | CART (Impureza Gini) | Los árboles se aplanan en arrays lineales (feature\_index , threshold ) para permitir un uso de pila de **O(1)** durante la inferencia mediante un ciclo **while**. |
| **RandomForestClassifier** | Bagging (Agregación Bootstrap) | Genera funciones C independientes para cada árbol y una función ligera de **"Voto Mayoritario"** en C. |
| **MiniLinearModel** | Descenso de Gradiente Estocástico (SGD) | Utiliza actualizaciones de pesos iterativas. Exporta un array simple de flotantes weights para la inferencia mediante producto de puntos. |
| **MiniSVM** | SVM Lineal (Pérdida Hinge) | Implementa un límite de decisión lineal, perfecto para clasificación binaria en hardware con **FPU** limitado. |
| **MiniNeuralNetwork** | MLP (Backpropagation) | Soporta **Cuantificación**. Puede convertir pesos de **float de 32 bits** a **enteros de 8 bits**, reduciendo el tamaño del modelo en $\\sim$75% para el almacenamiento en memoria Flash. |
| **KNearestNeighbors** | Distancia Euclidiana (Lazy) | **⚠️ Advertencia:** Exporta el *conjunto de datos completo* de entrenamiento como un **array C const**. Alto consumo de memoria Flash. |
| **MiniScaler** | Escalado MinMax / Estándar | Registra estadísticas (min/max/mean/std) durante el entrenamiento para generar una función C **preprocess\_data()** que normaliza las entradas del sensor en tiempo real. |

---

### **2\. ml\_manager.py (El Orquestador)**

La API de alto nivel que unifica el flujo de trabajo. Actúa como un puente entre el usuario y los algoritmos base.

* **Doble Núcleo Inteligente:** Verifica la presencia de **sklearn**. Si está, lo usa para un entrenamiento de alta velocidad en el PC. Si no, cambia sin problemas a **ml\_runtime**.  
* **Pipeline Automatizado:**  
  1. **Imputación:** Rellena los valores faltantes (NaN) para prevenir fallos.  
  2. **Escalado:** Normaliza los datos usando **MiniScaler**.  
  3. **Entrenamiento:** Ajusta el modelo seleccionado.  
* **Polimorfismo de predict():** Maneja automáticamente la entrada cruda, aplica el escalador guardado y ejecuta la inferencia.

---

### **3\. ml\_compat.py (Seguridad y Compatibilidad)**

El guardián de los datos. Asegura que la naturaleza dinámica de Python no rompa la estricta naturaleza estática de C.

* **\_flatten\_tree\_to\_arrays():** La función más crítica para modelos basados en árboles. Recorre la estructura de árbol del diccionario de Python y la serializa en arrays paralelos (estilo C), habilitando la lógica de ejecución iterativa requerida para microcontroladores.  
* **check\_dims():** Valida estrictamente las dimensiones de entrada antes de la predicción, previniendo errores de índice fuera de límites en el código C generado.  
* **impute\_missing\_values():** Asegura la integridad de los datos antes de que lleguen al núcleo matemático.

---

### **4\. ml\_factory.py (El Patrón Factory)**

Desacopla la instanciación del modelo del flujo de lógica.

* **Función:** $\\text{create\\\_model}(\\text{type\\\_string}, \\text{params\\\_dict})$  
* **Propósito:** Permite al sistema instanciar objetos complejos (como RandomForestRegressor) a partir de simples cadenas JSON. Esto es vital para el sistema de Guardar/Cargar y previene dependencias circulares entre módulos.

---

### **5\. ml\_exporter.py (Serialización y Exportación)**

Maneja la persistencia y traducción de modelos.

* **Extracción de Estructura:** En lugar de usar **pickle** de Python (que es inseguro y específico de Python), este módulo extrae la estructura matemática pura (pesos, umbrales, topología) a un **formato JSON agnóstico al lenguaje**.  
* **Interoperabilidad con Sklearn:** Si un modelo fue entrenado usando **scikit-learn**, este módulo extrae los arrays internos de NumPy ($\\text{tree\\\_value, coef\\\_}$) y los convierte al formato estándar de MiniML, permitiendo **exportar modelos de Sklearn a C de Arduino**.

---

## **🛠️ Instalación y Uso**

### **Instalación**

Dado que MiniML es un paquete de Python puro, la instalación es sencilla:

Bash

pip install miniml

*(Opcional: Instalar scikit-learn para un entrenamiento más rápido en PC, pero NO es un requisito).*

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

La salida es código C99 estándar, listo para ser incluido en un sketch de Arduino ($\\text{\#include "model.h"}$).

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

## **🤝 Contribuciones**

¡Las contribuciones son bienvenidas\! MiniML tiene como objetivo mantener su filosofía de "cero dependencias".

1. Bifurca (Fork) el Proyecto.  
2. Crea tu Rama de Característica (git checkout \-b feature/AmazingFeature).  
3. Confirma tus Cambios (git commit \-m 'Add some AmazingFeature').  
4. Empuja a la Rama (git push origin feature/AmazingFeature).  
5. Abre un Pull Request.

---

## **📄 Licencia**

Distribuido bajo la Licencia MIT. Consulta $\\text{LICENSE}$ para más información.