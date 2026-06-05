# MiniML Engine: Documentación Técnica Oficial

### "Train on PC, Run on Metal."

**Versión:** 1.1.0

**Autor:** Michego Takoro "Wilner Manzanares"

**Licencia:** Apache 2.0

**Framework:** Deep Learning y Machine Learning para Sistemas Embebidos (Edge AI)

---

## Índice de Contenidos

* **Introducción**
* ¿Qué es y para qué sirve MiniML?
* Filosofía "Train on PC, Run on Metal"
* Casos de uso principales


* **Capítulo 1: Introducción Técnica**
* Visión general del ecosistema
* Filosofía de Cero Dependencias


* **Capítulo 2: Pipeline del Sistema**
* Fase de Ingesta y Preprocesamiento
* Entornos de ejecución: Legacy vs. MiniTensor
* Fase de Cuantificación y Optimización
* Proceso de Exportación a Bare-Metal


* **Capítulo 3: Modelos Legacy y Cuantificación**
* Modelos soportados (DT, RF, Linear, SVM, KNN, MLP)
* Tipos de cuantificación (INT8 Híbrido, Per-Channel, Per-Tensor)
* Métodos de cuantificación matemática
* CMSIS-NN y Fixed-Point
* Métricas de rendimiento y limitaciones técnicas
* Recomendaciones para proyectos embebidos


* **Capítulo 4: Modelos y Capas de Deep Learning (MiniTensor)**
* Detalles técnicos por capa (`Conv1D/2D`, `SeparableConv2D`, `ResidualBlock1D`, etc.)
* Funciones de Activación (`ReLU`, `Sigmoid`) y Pérdida (`MSE`, `CrossEntropy`)
* Prácticas de diseño y manejo de capas
* Garantía de cómputo y condiciones de uso


* **Capítulo 5: Módulo de Hardware**
* Arquitectura del `serial_manager.py`
* Simulación con `virtual_sensor.py`
* Limitaciones y separación lógica: Software vs. Hardware físico


* **Capítulo 6: Exportación y Empaquetado C++**
* Arquitectura del transpilador y generación de código
* Seguridad de memoria: `PROGMEM` y SRAM Estática
* Seguridad del modelo: Inferencia protegida
* Limitaciones técnicas actuales (Estado del exportador a Rust)


* **Capítulo 7: CLI de MiniML**
* Auditoría con `inspect`
* Perfilado de memoria con `estimate`
* Recolección de datos con `sensor`
* Entorno de simulación REPL con `simulate`


* **Conclusión**
* Manifiesto del creador



---

*Nota: Esta documentación está sujeta a cambios conforme evolucione el framework. Se recomienda consultar el repositorio oficial para las actualizaciones más recientes del motor.*


---

# Introducción: ¿Qué es y para qué sirve MiniML Engine?

**MiniML Engine** es un framework de Machine Learning y Deep Learning embebido de grado industrial, diseñado explícitamente para operar en sistemas con restricciones extremas de recursos (hardware con menos de 2KB de RAM, como microcontroladores AVR de 8-bits, ESP32 o STM32).

A diferencia de las arquitecturas tradicionales de Inteligencia Artificial que dependen de ecosistemas pesados, MiniML Engine se fundamenta en un principio de **Cero Dependencias (Zero-Dependency)**. Todo su núcleo matemático, desde el álgebra lineal básica hasta el motor tensorial de diferenciación automática (*Autograd*) conocido como **MiniTensor**, está escrito íntegramente en Python puro.

### ¿Para qué sirve realmente?

En esencia, MiniML Engine sirve como un puente determinista entre el entorno de desarrollo de alto nivel y el silicio físico. Su propósito es permitir a ingenieros, investigadores y desarrolladores de hardware:

1. **Entrenar y Diseñar localmente:** Construir arquitecturas de IA (desde árboles de decisión hasta redes neuronales convolucionales complejas) utilizando un entorno Python estándar en una PC, sin la sobrecarga de instalar librerías masivas de terceros.
2. **Exportar a "Bare Metal":** Traducir ese modelo matemático entrenado en código **C++ nativo, estático y determinista**.
3. **Ejecutar Inferencia en el Borde (Edge AI):** Implementar el modelo directamente en el hardware final para que procese señales de sensores en tiempo real, de manera offline, sin latencia de red y con un consumo energético mínimo.

### La Filosofía Core: *"Train on PC, Run on Metal"*

El framework divide el ciclo de vida del Machine Learning en dos fases estrictamente separadas:

* **Fase de Entrenamiento (Host):** Se realiza en hardware con abundantes recursos (PC/Servidor) aprovechando la flexibilidad de Python para calcular gradientes, optimizar pesos y estructurar la topología del modelo.
* **Fase de Inferencia (Edge):** Se ejecuta en el hardware objetivo. El exportador de MiniML no "interpreta" el modelo en el microcontrolador; en su lugar, hace ingeniería inversa de la estructura matemática y la compila directamente en instrucciones C++ planas.

### ¿Por qué utilizar MiniML Engine? (Casos de Uso)

El ecosistema de MiniML y MiniTensor no está pensado para ejecutar Modelos de Lenguaje Masivos (LLMs) en servidores en la nube. Su dominio absoluto es el **Internet de las Cosas (IoT) y la Robótica**:

* **Mantenimiento Predictivo:** Ingerir datos de acelerómetros para detectar vibraciones anómalas en motores industriales en la misma línea de ensamblaje.
* **Soft-Sensors (Fusión de Sensores):** Combinar datos analógicos simples (temperatura, humedad, voltaje) para predecir variables físicas complejas en tiempo real sin requerir sensores costosos.
* **Tiny Vision & Audio:** Implementar convoluciones optimizadas espacialmente para clasificar patrones de audio o matrices de imágenes térmicas de baja resolución directamente en la placa.
* **Seguridad de la Información:** Al procesar todo de manera local en el chip, la información sensible nunca abandona el dispositivo, garantizando privacidad total por diseño.

MiniML Engine asume el trabajo pesado de la gestión de memoria (uso de memoria Flash `PROGMEM` y SRAM), la cuantificación y la arquitectura de software, dejando en manos del integrador únicamente la responsabilidad del hardware y el acondicionamiento de la señal física.

---

# Capítulo 1. Pipeline del Sistema (Flujo de Funcionamiento)

El pipeline de **MiniML Engine** está diseñado bajo una arquitectura de "Línea de Ensamblaje". El ciclo de vida de los datos pasa por etapas estrictamente aisladas: desde la ingesta de la matriz cruda hasta la generación del firmware final en C++.

Este flujo se bifurca internamente en dos motores distintos, dependiendo de si el usuario invoca un modelo de Machine Learning Clásico (Legacy) o una arquitectura de Deep Learning (MiniTensor). A continuación, se detalla el funcionamiento técnico paso a paso de este ecosistema.



### Fase 1: Ingesta y Preprocesamiento de Datos

Antes de que cualquier algoritmo ejecute operaciones matemáticas, el framework asegura la integridad de la información a través del módulo orquestador (`ml_manager.py` y `ml_compat.py`).

* **Validación Estructural:** El motor verifica que las dimensiones de entrada sean consistentes y que no existan irregularidades que puedan provocar desbordamientos de memoria (*Buffer Overflow*) posteriormente en el microcontrolador.
* **Imputación de Datos:** Los valores faltantes (`NaN`) en la matriz de entrada son detectados y neutralizados mediante técnicas estadísticas básicas para evitar la propagación de errores en el cálculo de gradientes o divisiones por cero.
* **Acondicionamiento (MiniScaler):** Las señales de los sensores suelen tener magnitudes dispares (ej. humedad de 0 a 100, y presión en miles). El `MiniScaler` ajusta estos valores a rangos manejables (MinMax o Standard) y, crucialmente, guarda estos parámetros de escalado para inyectarlos más adelante en el código C++.



### Fase 2: Entornos de Ejecución y Entrenamiento

Una vez que los datos están limpios, el orquestador enruta el flujo hacia uno de los dos motores de cálculo del framework.

#### Ruta A: El Pipeline Legacy (Machine Learning Clásico)

Ubicado principalmente en `ml_runtime.py`, este pipeline maneja algoritmos como Árboles de Decisión, Random Forest, SVM y KNN.

* **Estructuras Planas:** A diferencia de las implementaciones tradicionales que usan objetos recursivos complejos, este motor entrena los modelos y simultáneamente aplana sus estructuras en memoria.
* **Diseño Inverso:** Durante el `fit()`, un árbol de decisión no se guarda como nodos anidados, sino que se serializa directamente en arreglos paralelos unidimensionales (índices de características, umbrales, nodos hijos). Esto prepara el modelo para una inferencia iterativa de memoria constante en el dispositivo físico.

#### Ruta B: El Pipeline MiniTensor (Deep Learning & Autograd)

Ubicado en los módulos de `tensor.py` y `layers.py`, este es el motor de diferenciación automática.

* **Construcción del Grafo Computacional:** Al definir una red mediante `nn.Sequential()`, MiniML construye un grafo acíclico dirigido (DAG) en la memoria de la PC. Cada operación matemática realizada sobre un `Tensor` registra su propio historial.
* **Forward Pass:** Los tensores de datos atraviesan las capas (ej. `Conv1D`, `Linear`). El motor calcula activaciones y extrae características manteniendo un registro de las transformaciones geométricas (especialmente crucial en capas como `Flatten` y `ResidualBlock1D`).
* **Backward Pass (Autograd):** Al invocar la función de pérdida y ejecutar `backward()`, el motor aplica la regla de la cadena del cálculo diferencial, derivando el error a través de la topología de la red para actualizar los pesos.
* **Actualización de Pesos:** Optimizadores iterativos (como SGD) ajustan los tensores paramétricos ciclo tras ciclo hasta la convergencia.



### Fase 3: Cuantificación y Optimización (Post-Training)

Una vez que el modelo ha convergido en el PC (donde los pesos son flotantes de 32 bits y ocupan grandes bloques de memoria), el pipeline entra en la fase de optimización para Edge AI.

* **Mapeo de Precisión:** El usuario puede invocar la cuantificación híbrida. El motor escanea los tensores, calcula factores de escala y puntos cero, y comprime los parámetros a enteros de 8 bits (`INT8`).
* **Preparación de Operator Fusion:** Para modelos Deep Learning con topologías específicas (como `SeparableConv2D`), el motor identifica patrones de convolución secuencial y fusiona los bucles computacionales. Esto elimina la necesidad de crear tensores intermedios en la RAM estática (SRAM) del microcontrolador.



### Fase 4: Exportación a Bare-Metal (Transpilación a C++)

La etapa final y más crítica del pipeline. El modelo, ahora optimizado y/o cuantizado, abandona el entorno de Python.

* **Generación de Archivos:** Los módulos `ml_exporter.py` y `cpp_writer.py` extraen los pesos paramétricos y la topología guardada.
* **Mapeo de Memoria (PROGMEM):** El exportador traduce los tensores directamente a arreglos de C++ etiquetados estáticamente. Aplica directivas específicas de hardware (`PROGMEM`) para obligar al compilador del microcontrolador a alojar estos pesos masivos en la memoria Flash (ROM) y no en la SRAM dinámica.
* **Inyección de Lógica:** Se escriben las rutinas de inferencia exactas que coinciden con la topología entrenada (bucles for anidados para convoluciones, iteraciones while para árboles).
* **Empaquetado:** El módulo `LibraryPackager` toma todo este código en crudo y lo estructura en un archivo `.zip` de grado industrial con manifiestos (`library.json`, `library.properties`), listo para ser compilado en cualquier IDE de sistemas embebidos.

---

# Capítulo 2. Modelos Base (Legacy & MLP)

El ecosistema alojado en el módulo `ml_runtime.py` contiene los algoritmos fundacionales del framework. A diferencia del motor Autograd (MiniTensor) que construye grafos dinámicos, estos modelos están programados usando listas nativas de Python y álgebra lineal cruda (`MiniMatrixOps`). Esta simplicidad arquitectónica permite exportaciones a C++ extremadamente compactas, ideales para hardware con memoria SRAM hiper-reducida.

A continuación, se detallan los algoritmos soportados, su funcionamiento interno a nivel de hardware, casos de uso ideales y la sintaxis para invocarlos a través del orquestador unificado (`ml_manager.py`).



### 1. DecisionTree (Clasificación y Regresión)

* **¿Qué es y cómo funciona?**
Utiliza el algoritmo CART evaluando la Impureza de Gini (para clasificación) o el Error Cuadrático Medio (para regresión) para crear ramificaciones lógicas.
* **Optimización Edge (C++):**
En lugar de generar estructuras recursivas complejas en C++ que podrían causar un *Stack Overflow*, el framework aplana la topología del árbol. Exporta arreglos paralelos unidimensionales (`feature_index`, `threshold`, `left`, `right`, `value`) hacia la memoria Flash (`PROGMEM`). La inferencia se ejecuta mediante un simple bucle `while`, garantizando un uso de memoria dinámica (RAM) de **O(1)**.
* **Casos de uso:**
Alarmas basadas en umbrales físicos (ej. sistemas de detección de incendios evaluando temperatura y gas), donde se requiere auditar exactamente *por qué* el modelo tomó una decisión.
* **Sintaxis de Entrenamiento:**
```python
import miniml

# Entrenar Árbol de Decisión con profundidad máxima de 5
modelo = miniml.train_pipeline(
    model_name="detector_incendios",
    dataset=datos_entrenamiento,
    model_type="DecisionTreeClassifier", # o "DecisionTreeRegressor"
    params={"max_depth": 5, "min_size": 1},
    scaling="minmax"
)

```





### 2. RandomForest (Clasificación y Regresión)

* **¿Qué es y cómo funciona?**
Implementa *Bagging* (Agregación Bootstrap), entrenando múltiples árboles de decisión independientes sobre subconjuntos de los datos y tomando un promedio o voto mayoritario para reducir el sobreajuste (*overfitting*).
* **Optimización Edge (C++):**
Genera múltiples matrices planas en `PROGMEM` y una función `predict()` independiente para cada árbol. Durante la inferencia, una función maestra ejecuta el "Voto Mayoritario" (o promedio para regresión) en la SRAM para decidir la salida final.
* **Casos de uso:**
Fusión de sensores ambientales complejos (ej. predecir la ocupación de una sala combinando sensores LDR, PIR y CO2).
* **Sintaxis de Entrenamiento:**
```python
# Entrenar Random Forest con 10 árboles
modelo = miniml.train_pipeline(
    model_name="sensor_fusion",
    dataset=datos_entrenamiento,
    model_type="RandomForestClassifier", # o "RandomForestRegressor"
    params={"n_trees": 10, "max_depth": 5},
    scaling="standard"
)

```




### 3. MiniLinearModel (Regresión Lineal)

* **¿Qué es y cómo funciona?**
Un modelo base optimizado mediante Descenso de Gradiente Estocástico (SGD) para predecir variables continuas.
* **Optimización Edge (C++):**
Es el modelo más rápido de todo el framework. Exporta un único arreglo unidimensional de números en punto flotante (`weights`) a `PROGMEM`. La predicción se reduce a una operación aritmética de producto escalar (*dot product*) más un sesgo (*bias*).
* **Casos de uso:**
Calibración algorítmica de sensores analógicos (ej. predecir el porcentaje de vida útil restante de una batería basándose en la curva de caída de voltaje).
* **Sintaxis de Entrenamiento:**
```python
modelo = miniml.train_pipeline(
    model_name="calibrador_bateria",
    dataset=datos_entrenamiento,
    model_type="linear_regression",
    params={"learning_rate": 0.01, "epochs": 1000},
    scaling="minmax"
)

```



### 4. MiniSVM (Máquina de Vectores de Soporte Lineal)

* **¿Qué es y cómo funciona?**
Implementa un clasificador lineal maximizando el margen entre dos clases utilizando la función de pérdida *Hinge Loss*.
* **Optimización Edge (C++):**
Exporta un hiperplano ligero. Al ser un límite de decisión puramente lineal, evita operaciones matemáticas no lineales costosas, siendo ideal para microcontroladores AVR que carecen de Unidad de Coma Flotante (FPU). Evalúa si el producto escalar es mayor o menor a cero para emitir un `1` o `-1`.
* **Casos de uso:**
Clasificación estrictamente binaria en líneas de ensamblaje (ej. control de calidad "Pasa / Falla").
* **Sintaxis de Entrenamiento:**
```python
modelo = miniml.train_pipeline(
    model_name="qa_tester_svm",
    dataset=datos_entrenamiento,
    model_type="MiniSVM",
    params={"learning_rate": 0.01, "n_iters": 1000},
    scaling="standard"
)

```



### 5. K-Nearest Neighbors (KNN)

* **¿Qué es y cómo funciona?**
Algoritmo de aprendizaje perezoso (*Lazy Learning*) que clasifica una nueva muestra calculando la distancia euclidiana hacia los 'K' puntos más cercanos en el conjunto de entrenamiento.
* **Optimización Edge (C++):**
Exporta **todo el conjunto de datos como arreglos constantes en la memoria Flash (`PROGMEM`)**. Para evitar colapsar la RAM durante el cálculo de distancias, implementa un algoritmo iterativo de *Insertion Sort* in-place, simulando una cola de prioridad que solo retiene los 'K' vecinos más cercanos.
* **Casos de uso:**
Reconocimiento de patrones espaciales muy simples donde la calibración debe ser explicable estrictamente por proximidad.
* **⚠️ Limitación Técnica:**
Consume memoria Flash de manera proporcional al tamaño del dataset ($O(N)$). Solo debe usarse con conjuntos de entrenamiento diminutos (< 200 muestras) para evitar desbordar el almacenamiento de la placa.
* **Sintaxis de Entrenamiento:**
```python
modelo = miniml.train_pipeline(
    model_name="knn_clasificador",
    dataset=datos_entrenamiento_reducido,
    model_type="knn",
    params={"k": 3, "task": "classification"},
    scaling="minmax"
)

```



### 6. MiniNeuralNetwork (Perceptrón Multicapa - MLP)

* **¿Qué es y cómo funciona?**
Es el puente del framework hacia el Deep Learning. Consiste en una red neuronal *Feed-Forward* multicapa, entrenada desde cero con una implementación cruda de propagación hacia atrás (*Backpropagation*) y optimizador SGD. Soporta múltiples salidas y activaciones (`sigmoid`, `relu`, `linear`).
* **Optimización Edge (C++):**
Este modelo implementa de forma nativa la **Cuantificación Híbrida INT8 (Post-Training Quantization)**. Los pesos (Float32) se comprimen a enteros de 8 bits y se almacenan en `PROGMEM`. Durante la inferencia, C++ lee los bytes y los multiplica por factores de escala (`s_W1`, `s_W2`) para de-cuantizar "al vuelo", protegiendo la SRAM sin sacrificar precisión.
* **Casos de uso:**
Resolución de problemas no lineales complejos en microcontroladores de 8-bits donde MiniTensor sería excesivo, como sensores de gas multicomponente o reconocimiento de gestos a través de IMUs.
* **Sintaxis de Entrenamiento:**
```python
# Entrenar MLP con 1 capa oculta de 8 neuronas
modelo = miniml.train_pipeline(
    model_name="sensor_gestos_mlp",
    dataset=datos_entrenamiento,
    model_type="neural_network",
    params={
        "n_inputs": 3, 
        "n_hidden": 8, 
        "n_outputs": 2, 
        "learning_rate": 0.1, 
        "epochs": 2000
    },
    scaling="minmax"
)

```

---

# Capítulo 3. Cuantización de Modelos para Edge AI (Para enriquecer la lectura, puede leerse la guia de cuantificación dentro de la misma carpeta)

Uno de los mayores desafíos al llevar la Inteligencia Artificial a microcontroladores (como los AVR de 8 bits o los ARM Cortex-M) es la estricta limitación de memoria RAM (SRAM) y almacenamiento (Flash). Los modelos entrenados en PC utilizan tensores en punto flotante de 32 bits (`Float32`), los cuales consumen rápidamente los recursos del hardware embebido.

Para resolver esto, **MiniML Engine** y **MiniTensor** implementan un ecosistema de cuantificación avanzado. La cuantificación es el proceso de mapear números continuos de alta precisión (32 bits) a enteros de menor precisión (usualmente 8 bits), reduciendo el tamaño del modelo drásticamente sin perder la fidelidad matemática.

A continuación, se desglosan los enfoques de cuantificación soportados por el framework.



### 1. Cuantización Híbrida INT8

La Cuantificación Híbrida es la estrategia principal utilizada por el exportador nativo de C++ para redes neuronales (MLP).

* **¿Cómo funciona?** Los pesos matemáticos de la red neuronal se comprimen de `Float32` a `INT8` (`int8_t`) durante la exportación. Estos enteros se almacenan forzosamente en la memoria Flash (`PROGMEM`). Sin embargo, la entrada del sensor y las variables temporales dentro del microcontrolador se mantienen en formato `Float32` para evitar desbordamientos aritméticos.
* **Ventaja Edge:** Reduce el peso físico del modelo en un ~75% dentro del almacenamiento de la placa, manteniendo la robustez de las operaciones aritméticas en coma flotante.

### 2. Cuantización "Al Vuelo" (On-the-Fly Dequantization) en MLP

Estrechamente ligada a la estrategia híbrida, esta técnica ocurre milisegundo a milisegundo durante la inferencia física en la placa.

* **El Proceso:** En lugar de cargar todos los pesos cuantizados a la SRAM y de-cuantizarlos de golpe (lo que colapsaría la memoria), el código C++ generado por MiniML lee un único byte (`int8_t`) directamente desde la memoria Flash, lo multiplica por un factor de escala flotante precalculado (`s_W1` o `s_W2`), y realiza la multiplicación con el dato de entrada.
* **Impacto:** Permite ejecutar redes neuronales profundas con un consumo de SRAM dinámico de apenas unos pocos bytes (limitado a las matrices de activación interactivas).

### 3. Post-Training Quantization (PTQ)

El PTQ es la técnica estándar para cuantizar un modelo *después* de que ha sido entrenado por completo. Es el enfoque principal del `MiniNeuralNetwork` en el módulo de ML Clásico.

* **Calibración:** Antes de cuantizar, el modelo debe "observar" datos reales para entender los rangos dinámicos (mínimos y máximos) de las activaciones en cada capa. El método `calibrate(dataset)` escanea los tensores y establece los factores de escala (`act_scales`).
* **Compresión:** Posteriormente, el método `quantize()` utiliza estos rangos para mapear de forma segura los pesos a enteros entre -127 y 127. El framework soporta de forma automática la cuantificación por canal (*Per-Channel*), calculando una escala independiente para cada fila de la matriz de pesos, lo que minimiza drásticamente el error de redondeo.

### 4. Quantization-Aware Training (QAT) en MiniTensor

Mientras que PTQ cuantiza *después* del entrenamiento, el QAT (*Entrenamiento Consciente de la Cuantificación*) se implementa en el motor Autograd de MiniTensor para modelos de Deep Learning más sensibles (como CNNs complejas).

* **Simulación durante el Forward Pass:** Durante la etapa de entrenamiento en el PC, el motor MiniTensor inserta nodos falsos de cuantificación/de-cuantificación en el grafo computacional. Esto "engaña" a la red neuronal, forzándola a experimentar el error de redondeo INT8 en tiempo real.
* **Ajuste de Gradientes:** Al utilizar un estimador de gradiente directo (*Straight-Through Estimator*), el optimizador SGD ajusta los pesos de la red para que se vuelvan matemáticamente resistentes a la pérdida de precisión de los 8 bits.
* **Ventaja:** Produce modelos INT8 mucho más precisos que el PTQ tradicional, ideal para topologías como `SeparableConv2D` o arquitecturas residuales.



### ¿Cómo invocar la Cuantización en el Código? (Sintaxis y API)

El framework está diseñado para que la cuantificación sea transparente y automatizada. Aquí se muestra el flujo de trabajo tanto para llamadas directas como a través del orquestador unificado.

#### A. Invocación Directa (Modo Manual PTQ)

Si estás manipulando la red neuronal directamente, debes seguir el orden estricto de: Entrenar -> Calibrar -> Cuantizar.

```python
from miniml import ml_runtime

# 1. Definir y entrenar el modelo
nn = ml_runtime.MiniNeuralNetwork(n_inputs=3, n_hidden=8, n_outputs=1)
nn.fit(dataset_entrenamiento)

# 2. Calibración (CRÍTICO para PTQ)
# Pasa un subconjunto de datos para encontrar los rangos min/max de activación
nn.calibrate(dataset_calibracion)

# 3. Cuantizar (Aplica compresión INT8 Per-Channel)
nn.quantize(per_channel=True)

# 4. Exportar el C++ Cuantizado
codigo_cpp = nn.to_arduino_code(fn_name="prediccion_cuantizada")

```

#### B. Invocación mediante el Orquestador (`ml_manager`)

El orquestador automatiza el proceso de calibración y cuantificación durante la exportación a C++. Si invocas la exportación de una red neuronal, el `ml_manager` detectará si el modelo requiere optimización INT8.

```python
import miniml

# 1. Entrenar a través del pipeline (maneja el escalado automáticamente)
modelo = miniml.train_pipeline(
    model_name="sensor_vibracion",
    dataset=datos_entrenamiento,
    model_type="neural_network",
    params={"n_inputs": 3, "n_hidden": 8, "n_outputs": 1}
)

# 2. Exportación Directa (Aplica PTQ automáticamente si es posible)
codigo_cpp = miniml.export_to_c("sensor_vibracion")

```


### 5. Flujo de Cuantificación en MiniML Engine

El proceso de llevar un modelo desde su estado matemático puro en coma flotante (Float32) hasta un binario de enteros (INT8) optimizado para microcontroladores no ocurre por arte de magia. Sigue un pipeline algorítmico estricto para asegurar que la pérdida de información sea estadísticamente insignificante.

Este es el flujo técnico exacto que sigue el framework internamente:

1. **Entrenamiento en Coma Flotante (Float32):** El modelo (ej. `MiniNeuralNetwork`) se entrena normalmente utilizando el motor Autograd o el descenso de gradiente estándar. Durante esta fase, los pesos y sesgos se ajustan libremente con alta precisión matemática para encontrar el mínimo global.
2. **Calibración Dinámica:** Al invocar `calibrate()`, el modelo hace un *Forward Pass* "silencioso" utilizando un subconjunto de datos representativos. El motor registra los valores absolutos máximos de las activaciones en cada capa (Entrada, Oculta, Salida).
3. **Cálculo de Escalas (Scale Factors):** Con los valores máximos capturados, el motor calcula el factor de escala ($S$) necesario para mapear ese rango dinámico dentro del límite de un entero con signo de 8 bits ($-127$ a $127$).
4. **Compresión (Quantization):** Al invocar `quantize()`, se aplica la transformación matemática a las matrices. Los pesos originales se dividen por el factor de escala y se redondean al entero más cercano.
5. **Exportación Híbrida C++:** El generador de código extrae los pesos comprimidos y los escribe como arreglos `int8_t` etiquetados con `PROGMEM` para que el compilador de Arduino/C++ los aloje en la memoria Flash. Los factores de escala se exportan como flotantes para permitir la de-cuantización durante la inferencia.



### 6. Métodos de Cuantificación Matemática en MiniML

MiniML Engine utiliza un esquema de **Cuantificación Uniforme Asimétrica/Simétrica**. La transformación de los tensores se rige por la siguiente lógica matemática integrada en el código fuente:

**Para Cuantizar (De Float32 a INT8):**
La fórmula base para comprimir un peso $W$ a su versión cuantizada $Q_w$ es:


$$Q_w = \text{clamp}\left( \text{round}\left( \frac{W}{S} \right), -127, 127 \right)$$


Donde $S$ es el factor de escala calculado como $S = \frac{\max(|W|)}{127.0}$.

**Manejo de Sesgos (Biases):**
A diferencia de los pesos de conexión, los sesgos ($B$) son extremadamente sensibles a los errores de redondeo. En el modo de cuantificación nativo de MiniML, el motor realiza un ajuste de escala efectivo ($S_{in} \times S_w$) y comprime el sesgo a un entero de 32 bits (`INT32`) para evitar pérdidas catastróficas de precisión, o bien, en el modo exportador C++ ultraligero, los mantiene como variables Float nativas en `PROGMEM` dado que su huella de memoria es dimensionalmente minúscula ($O(N)$) en comparación con las matrices de pesos ($O(N \times M)$).

**De-Cuantización Al Vuelo (De INT8 a Float32):**
Durante la ejecución en el microcontrolador, la capa matemática en C++ reconstruye la señal aproximada antes de aplicarle la función de activación (ej. ReLU o Sigmoid):


$$V_{approx} = (Q_w \times S_w \times X_{in}) + B$$


Este enfoque garantiza que las activaciones no sufran desbordamientos aritméticos (Overflows), un problema común en arquitecturas de hardware de 8-bits.



### 7. Soporte Per-Channel vs. Per-Tensor

Una de las características de grado industrial del motor de cuantificación de MiniML es su capacidad para gestionar la granularidad (*Granularity*) del mapeo de escalas. El rendimiento del modelo en el Edge depende críticamente de cómo se aplique esta compresión.

#### Cuantización Per-Tensor (Por Tensor Completo)

* **Concepto:** Se calcula **un único factor de escala** global para toda la matriz de pesos de una capa (ej. un único $S$ para todos los pesos que conectan la capa de entrada con la capa oculta).
* **Ventaja:** Genera un código C++ ligeramente más corto y ahorra unos pocos bytes de memoria Flash, ya que solo debe almacenar un factor de escala flotante.
* **Desventaja técnica:** Si un solo peso en la matriz tiene un valor anormalmente grande (un *outlier*), obligará a que el factor de escala sea gigante. Esto aplastará a todos los demás pesos pequeños hacia el cero (`0`), destruyendo la precisión de la red neuronal.

#### Cuantización Per-Channel (Por Canal / Por Neurona)

* **Concepto:** Se calcula **un factor de escala independiente para cada fila** (canal de salida o neurona) de la matriz de pesos.
* **Ventaja:** Es el **estándar por defecto activado en MiniML Engine** (`quantize(per_channel=True)`). Al tener escalas independientes, el motor aísla los pesos atípicos de una neurona sin afectar la precisión de las demás. Cada canal aprovecha al máximo los 256 valores posibles del rango `INT8` ($-127$ a $127$).
* **Implementación en C++:** En lugar de exportar un solo flotante, el exportador de MiniML genera un arreglo unidimensional de escalas en la memoria Flash (`s_W1[n_hidden]`). Durante el bucle de inferencia, el microcontrolador lee la escala específica para la neurona que está evaluando en ese ciclo exacto de reloj.

Esta implementación *Per-Channel* es el secreto arquitectónico que permite que las redes `MiniNeuralNetwork` generadas por el framework presenten un error de cuantificación estadísticamente nulo frente a su contraparte entrenada en PC.



### 8. Cuantificación y Compatibilidad con CMSIS-NN (Fixed-Point)

Aunque la filosofía principal de MiniML Engine es **Cero Dependencias** (evitando obligar al usuario a instalar librerías externas de fabricantes), el motor de exportación está diseñado con una arquitectura matemáticamente compatible con los estándares de la industria, específicamente con la aritmética de punto fijo (*Fixed-Point Arithmetic*) utilizada por **ARM CMSIS-NN**.

* **Aritmética de Punto Fijo (Fixed-Point):** En lugar de realizar la de-cuantificación multiplicando por un flotante (lo cual consume ciclos de reloj si el microcontrolador no tiene FPU), el código C++ generado puede ser optimizado para utilizar desplazamientos de bits (*Bit-shifting*). Las operaciones se transforman al formato $Q_m.n$, donde las multiplicaciones se resuelven mediante un desplazamiento a la derecha (`>>`).
* **Paridad de Rendimiento:** Si el código exportado por MiniML se compila en un chip ARM Cortex-M (ej. STM32), el compilador de C++ (`-O3`) optimizará las instrucciones SIMD para procesar los arreglos `int8_t` empaquetados, logrando una velocidad de inferencia casi idéntica a la que se obtendría integrando manualmente la librería CMSIS-NN, pero sin la pesadilla de configurar sus dependencias.



### 9. Limitaciones Técnicas de la Cuantificación

La cuantificación INT8 no es una solución mágica universal. Funciona comprimiendo la entropía matemática, lo que significa que solo es viable en arquitecturas con alta redundancia paramétrica. Es vital que el arquitecto de software entienda qué modelos soportan esta compresión y cuáles colapsarían si se les aplica.

#### ✅ Modelos que SOPORTAN y requieren Cuantificación

* **Deep Learning (MiniTensor):** Todas las capas paramétricas (`Conv1D`, `Conv2D`, `SeparableConv2D`, `Linear`, `ResidualBlock1D`). Al tener miles o millones de pesos, la redundancia es alta y la pérdida de precisión de un solo peso (por el paso a 8 bits) se diluye en la suma total del tensor.
* **Perceptrón Multicapa (MiniNeuralNetwork - Legacy):** Soporta PTQ de forma nativa. Indispensable para capas ocultas de más de 16 neuronas si se despliega en hardware AVR de 8-bits.

#### ❌ Modelos que NO SOPORTAN o no se benefician de la Cuantificación

* **DecisionTree & RandomForest:** **No soportados.** Los árboles basan sus decisiones en umbrales estrictos de corte (ej. `if temperatura > 25.43`). Si cuantizamos $25.43$ a un entero, el límite de decisión se deforma, destruyendo la precisión lógica del árbol. Además, su huella de memoria es de por sí mínima.
* **MiniLinearModel & MiniSVM:** **Innecesario.** Al ser modelos puramente lineales, constan de un único arreglo de pesos (tantos pesos como variables de entrada). Cuantizar un arreglo de 5 valores flotantes a enteros ahorraría apenas 15 bytes, pero añadiría un costo computacional injustificado al forzar la de-cuantificación al vuelo.
* **K-Nearest Neighbors (KNN):** **Limitado.** Aunque el dataset podría comprimirse a `INT8` para ahorrar memoria Flash, el cálculo de la Distancia Euclidiana ($d = \sqrt{\sum (q_i - p_i)^2}$) elevaría los enteros al cuadrado. En hardware de 8 bits, $127^2 = 16129$, lo que provocaría un desbordamiento inmediato (*Integer Overflow*) arruinando la predicción.



### 10. Proceso de Cuantificación (El Ciclo de Vida)

Para el desarrollador que utiliza MiniML Engine, el proceso de cuantificación se resume en cuatro etapas secuenciales claras dentro del pipeline:

1. **Entrenamiento (*Training Phase*):** El modelo se instancia y se entrena en la PC usando tensores en alta precisión (`Float32`). El optimizador (SGD) busca la convergencia matemática sin restricciones de memoria.
2. **Calibración (*Calibration Phase*):** Solo aplicable si se usa Post-Training Quantization (PTQ). Se inyecta un lote de datos reales (no de validación, sino un subconjunto representativo del entorno físico) para que el modelo registre los límites numéricos (mínimos y máximos) de las activaciones internas.
3. **Compresión (*Quantization Phase*):** Se invoca el método `.quantize()`. El framework convierte todas las matrices de pesos a `INT8`, calculando y almacenando los factores de escala (`Scale`) y, si corresponde, los puntos cero (`Zero-Point`).
4. **Generación de C++ (*Export Phase*):** El módulo `ml_exporter` transcribe el modelo a C++. Envuelve las matrices comprimidas en directivas `PROGMEM` y genera el bucle `predict()` con las matemáticas de de-cuantificación al vuelo incorporadas, listo para el empaquetado final.



### 11. Tabla Comparativa y Métricas de Rendimiento

Para ilustrar el impacto arquitectónico de la cuantificación en entornos con recursos extremos, el siguiente benchmark presenta el rendimiento simulado de distintos modelos de **MiniML Engine**.

Las métricas de consumo de memoria y latencia están calculadas asumiendo una compilación estándar con optimización `-O3` en dos placas representativas del sector Edge: un microcontrolador de 8-bits sin Unidad de Coma Flotante (Arduino Nano / ATmega328P) y un procesador de 32-bits con FPU (ESP32).

#### Benchmark: Impacto de la Cuantificación en Almacenamiento y Precisión

*Nota: Las pruebas se basan en un Perceptrón Multicapa (MLP) de topología `[16, 16, 4]` (320 parámetros) y una red convolucional pequeña basada en `SeparableConv2D` (~5000 parámetros).*

---

| Modelo / Topología | Estrategia de Cuantificación | Memoria Flash (ROM) | SRAM Dinámica | Latencia (ESP32) | Pérdida de Precisión (Accuracy Drop) |
| --- | --- | --- | --- | --- | --- |
| **MiniNeuralNetwork (MLP)** | **Ninguna (Float32 Nativo)** | 1.28 KB | 144 Bytes | ~0.8 ms | **0.0%** (Línea Base) |
| **MiniNeuralNetwork (MLP)** | **PTQ (Per-Tensor INT8)** | 0.32 KB | 144 Bytes | ~1.1 ms | **-4.5%** a **-8.0%** |
| **MiniNeuralNetwork (MLP)** | **PTQ (Per-Channel INT8)** | 0.40 KB | 144 Bytes | ~1.2 ms | **< 1.0%** (Recomendado) |
| **MiniTensor (SeparableConv2D)** | **Ninguna (Float32 Nativo)** | 20.00 KB | 2.50 KB | ~12.5 ms | **0.0%** (Línea Base) |
| **MiniTensor (SeparableConv2D)** | **QAT (INT8 + Operator Fusion)** | 5.20 KB | 2.50 KB | ~8.0 ms* | **< 0.5%** |

---

**La latencia en el modelo INT8 QAT es menor gracias al Operator Fusion y al uso de instrucciones SIMD (si se compila con soporte CMSIS-NN en ARM/ESP32), lo que compensa el costo de de-cuantización.*



### Análisis Técnico de los Resultados

Al analizar las métricas generadas por el empaquetador de MiniML, el arquitecto de software debe tomar decisiones basadas en los siguientes compromisos (*trade-offs*):

* **El Ahorro Drástico de ROM (Memoria Flash):** Como se observa en la tabla, pasar de `Float32` a `INT8` reduce la huella física de las matrices de pesos en un **75% exacto** (de 4 bytes por parámetro a 1 byte). El ligero aumento entre *Per-Tensor* (0.32 KB) y *Per-Channel* (0.40 KB) se debe a que este último debe almacenar un arreglo de flotantes con los factores de escala (uno por cada neurona), un costo mínimo que vale totalmente la pena por la precisión ganada.
* **Estabilidad de la SRAM:** Notarás que el consumo de RAM dinámica (SRAM) no varía entre los modelos flotantes y los cuantizados. Esto es un triunfo de la arquitectura de MiniML: la de-cuantización ocurre "al vuelo" leyendo byte por byte desde `PROGMEM`. Los tensores enteros nunca se vuelcan en la SRAM de forma masiva.
* **El Costo Oculto de la Latencia en 8-bits:** En la estrategia híbrida (donde los pesos son INT8 pero las matemáticas de inferencia y la escala se calculan en Float32), el microcontrolador debe convertir el número entero a flotante antes de multiplicarlo por la entrada. Si la placa física no tiene FPU (como un Arduino clásico de 8-bits), esta conversión por software puede hacer que la inferencia INT8 sea fraccionalmente *más lenta* que la Float32 nativa.
* **El Triunfo del QAT en Deep Learning:** Para arquitecturas profundas (MiniTensor), el Entrenamiento Consciente de la Cuantificación (QAT) mantiene la caída de precisión por debajo del **0.5%**. Esto permite implementar visión artificial básica en el borde con un riesgo estadísticamente nulo de que el modelo pierda su capacidad de generalización.



### 12. Limitaciones de Hardware (El Choque con la Realidad Física)

Por más optimizado que esté el código C++ generado por **MiniML Engine**, el silicio físico impone barreras inquebrantables. Al desplegar Inteligencia Artificial en microcontroladores (Edge Computing), el arquitecto de software debe diseñar asumiendo restricciones severas. A continuación, se detallan los cuellos de botella del hardware y cómo impactan en los modelos.

#### A. SRAM (Memoria Dinámica) - El Cuello de Botella Crítico

La SRAM es donde el microcontrolador guarda las variables temporales durante la ejecución. Es el recurso más escaso (ej. un Arduino Uno / ATmega328P tiene apenas **2 KB** de SRAM).

* **El Límite:** Si la red neuronal de MiniTensor requiere aplanar un tensor intermedio masivo (ej. el resultado de una convolución `Conv2D` antes de pasar a la capa `Linear`), esa matriz temporal debe existir en la SRAM.
* **El Riesgo:** Si el tamaño del tensor intermedio supera la memoria disponible, el microcontrolador sufrirá un *Heap/Stack Collision*, resultando en un reinicio silencioso (crash) o comportamiento errático.
* **Solución de MiniML:** El uso de topologías como `SeparableConv2D` (Operator Fusion) y la de-cuantización al vuelo previenen el agotamiento rápido de la RAM, además, el CLI tiene un estimador de memoria que permite ver cuanto consumirá el modelo entrenado en el microcontrolador (Tanto en SRAM como en Flash).

#### B. Memoria Flash / ROM (Almacenamiento)

La memoria Flash aloja el programa compilado y, gracias a la directiva `PROGMEM`, también almacena los pesos del modelo. Aunque es más abundante que la SRAM (ej. 32 KB en Arduino Uno, 4 MB en ESP32), es finita.

* **El Límite:** Algoritmos como K-Nearest Neighbors (KNN) o redes convolucionales no cuantizadas (`Float32`) devoran el espacio de la Flash linealmente con su tamaño.
* **El Riesgo:** El compilador del IDE (PlatformIO/Arduino) arrojará un error de *Oversize* impidiendo el flasheo si el modelo supera la capacidad de la placa.

#### C. FPU (Unidad de Coma Flotante) y Ciclos de Reloj

Microcontroladores de gama baja (8-bits) no poseen hardware dedicado para matemáticas con decimales.

* **El Límite:** Una multiplicación en punto flotante (`3.14 * 2.5`) debe ser resuelta por software, lo que toma cientos de ciclos de reloj comparado con el único ciclo que toma una multiplicación de enteros.
* **El Riesgo:** Redes neuronales profundas sin cuantizar en arquitecturas de 8-bits presentarán una latencia altísima, haciendo imposible la inferencia en tiempo real para señales rápidas (como vibración o audio).

#### D. Resolución del ADC (Conversor Analógico-Digital)

El modelo de ML asume que los datos de entrada son perfectos, pero el hardware rara vez lo es.

* **El Límite:** Si un sensor se conecta a un ADC de 10 bits, la señal tendrá ruido eléctrico, picos parásitos (glitches) y fluctuaciones térmicas.
* **El Riesgo:** Si el modelo fue entrenado en la PC con un dataset "limpio" y sin *Data Augmentation* (ruido artificial), fracasará al intentar predecir sobre la ruidosa señal del mundo real.



### 13. Recomendaciones y Mejores Prácticas para Proyectos Embebidos

Para garantizar el éxito al integrar MiniML Engine en prototipos físicos, sigue esta guía de diseño recomendada para entornos de producción.

#### 1. Respeta la Navaja de Ockham (Empieza por Legacy)

No uses un cañón para matar un mosquito. Si tu objetivo es encender un ventilador cuando una combinación de temperatura y humedad supera un límite, **no entrenes una red neuronal**. Usa un `DecisionTreeClassifier` o un `MiniLinearModel`. Consumirán bytes en lugar de Kilobytes y se ejecutarán en microsegundos. Reserva **MiniTensor** (Deep Learning) estrictamente para extracción de características complejas (series temporales, señales acústicas, visión).

#### 2. Acondicionamiento de Señal (Filtrado Previo)

**MiniML no es un filtro de hardware.** La función `predict()` espera datos estables.

* Implementa en C++ un filtro paso bajo (Low-Pass Filter) o una media móvil (*Moving Average*) sobre las lecturas del sensor `analogRead()` *antes* de pasarle el arreglo al modelo.
* Utiliza el `MiniScaler` exportado por el framework (`preprocess_data()`) invariablemente; las redes neuronales son extremadamente sensibles a entradas no normalizadas.

#### 3. Cuantifica por Defecto (Always INT8)

A menos que estés trabajando con un microprocesador potente (como un Cortex-M4F o superior con megabytes de almacenamiento), haz que la llamada a `.quantize()` sea obligatoria en tu script de Python para cualquier modelo de la familia MLP o CNN. El ahorro del 75% en memoria Flash justifica con creces la pérdida sub-porcentual de precisión matemática.

#### 4. Profiling antes de Flashear

Antes de conectar la placa física, utiliza el **CLI de MiniML** o el estimador de memoria integrado. Verifica el `Memory Footprint` generado en la consola. Si el modelo proyecta usar más del 70% de la SRAM de tu microcontrolador objetivo, rediseña la arquitectura (reduce el número de neuronas ocultas o aumenta el *stride* en tus convoluciones). Deja siempre un margen de SRAM libre (30%) para las variables globales, el stack del sistema operativo (si usas FreeRTOS) y el manejo de buses I2C/SPI.

#### 5. Gestión del Flujo de Datos (Ventanas de Tiempo)

En Edge AI, rara vez se predice sobre una sola lectura. Se infiere sobre una ventana de tiempo (ej. las últimas 50 lecturas del acelerómetro).

* Evita alojar arreglos dinámicos (`malloc`) para acumular estas lecturas. Usa un *Buffer Circular* (Ring Buffer) estático en tu código de Arduino para empujar los nuevos datos del sensor y descartar los viejos en tiempo constante $O(1)$, pasando este buffer ordenado a la función de inferencia de MiniML.



### 14. Fórmulas Matemáticas de Validación (Cuantificación)

Para los ingenieros e investigadores que necesiten auditar la pérdida matemática generada por la compresión INT8 en sus proyectos, el framework se rige por las siguientes ecuaciones fundamentales.

**Error de Cuantificación Absoluto ($E_q$):**
Mide la diferencia exacta entre el peso original en punto flotante ($W$) y el peso reconstruido a partir del entero de 8 bits ($Q_w$) multiplicado por su escala ($S$).


$$E_q = W - (Q_w \times S)$$

**Relación Señal a Ruido de Cuantificación (SQNR):**
Para redes Deep Learning (MiniTensor), evaluar el error peso por peso no es práctico. La métrica SQNR (Signal-to-Quantization-Noise Ratio) evalúa la degradación general de una capa entera. Un SQNR alto (típicamente $> 40 \text{ dB}$) indica que la red sobrevivió a la cuantificación sin pérdida crítica de información.


$$\text{SQNR (dB)} = 10 \log_{10} \left( \frac{\sum W^2}{\sum E_q^2} \right)$$



### 15. Prácticas Apropiadas (Best Practices) para Edge AI

El despliegue en *bare-metal* no perdona errores de arquitectura. Para garantizar que los modelos de MiniML y MiniTensor operen de forma robusta, estable y predecible en el silicio, se deben adoptar las siguientes prácticas de ingeniería:

* **Recorte de Valores Atípicos (Outlier Clipping) antes de PTQ:**
Antes de invocar `.quantize()`, analiza la distribución de los pesos de tu red. Si una capa tiene miles de pesos entre $-1.0$ y $1.0$, pero un único peso anómalo de $15.0$, el factor de escala $S$ se adaptará a ese $15.0$, aplastando todos los demás pesos útiles a $0$.
* *Solución:* Aplica una función de recorte (*Gradient Clipping* o *Weight Clipping*) durante el entrenamiento para mantener los pesos distribuidos uniformemente.


* **Selección Estricta del Conjunto de Calibración:**
Al usar el método `.calibrate(dataset)` en un MLP, no le pases el mismo dataset perfecto con el que entrenaste. Pásale un subconjunto de datos ruidosos capturados directamente del hardware físico. Esto obliga a los rangos dinámicos a prepararse para las fluctuaciones reales del sensor (ruido térmico del ADC).
* **Protección del Hilo Principal (Non-Blocking AI):**
En microcontroladores de un solo núcleo, la llamada a `predict()` bloquea la ejecución. Si tu red neuronal convolucional tarda 12 ms en inferir, durante esos 12 ms el microcontrolador no podrá actualizar pantallas OLED ni mantener el balanceo de motores.
* *Solución:* Desacopla la adquisición de datos de la inferencia utilizando interrupciones de hardware (Timers/ISR) para llenar el buffer de entrada, y llama a `predict()` únicamente en el bucle `loop()` principal cuando el buffer esté lleno.


* **Alineación de Dimensiones (Geometría de Tensores):**
Cuando diseñes arquitecturas con `ResidualBlock1D`, asegúrate de que el número de canales de entrada coincida exactamente con los de salida ($C_{in} = C_{out}$) o implementa una convolución proyectiva de 1x1. C++ no tiene recolección de basura (*Garbage Collection*); si fuerzas dimensiones asimétricas, el generador estático fallará la compilación para proteger la placa física.



### 16. Casos de Uso Reales

El ecosistema dual de MiniML Engine abarca todo el espectro del procesamiento embebido. Aquí se definen los escenarios donde el framework despliega su máximo potencial.

#### A. Robótica Educativa y de Servicio (Bajo Costo)

* **El Problema:** Plataformas robóticas implementadas en instituciones o liceos que operan con hardware muy limitado (ej. placas basadas en AVR) y necesitan tomar decisiones inteligentes basadas en sensores de proximidad o infrarrojos, sin depender de una Raspberry Pi costosa.
* **Solución MiniML:** Utilizar un modelo Legacy como `DecisionTreeClassifier` o `RandomForest`. Estos modelos se evalúan en microsegundos usando estructuras `while` en memoria constante ($O(1)$), dejando el 99% de la CPU y la RAM libres para la cinemática de los motores y la lógica de evasión de obstáculos.

#### B. Mantenimiento Predictivo Industrial (Vibración y Acústica)

* **El Problema:** Un motor de ensamblaje industrial sufre micro-fallas mecánicas imperceptibles antes de romperse. Enviar gigabytes de audio o lecturas de acelerómetros a la nube para su análisis es lento, costoso y un riesgo de ciberseguridad.
* **Solución MiniTensor:** Una red neuronal basada en `Conv1D` y `MaxPool1D`, cuantizada a INT8 con QAT. El microcontrolador lee una ventana de 256 muestras temporales del acelerómetro y extrae características locales (frecuencias anómalas). La inferencia ocurre localmente en milisegundos, y la placa solo envía una señal de "ALERTA" a la red central cuando detecta el patrón de falla.

#### C. Soft-Sensors Agrícolas y Ambientales

* **El Problema:** Medir la tasa de evapotranspiración del suelo o la concentración de ciertos gases requiere sensores químicos de miles de dólares, inaccesibles para el monitoreo a gran escala.
* **Solución MiniML (MLP Híbrido):** Se despliegan sensores ultrabaratos (temperatura, humedad relativa, presión atmosférica, luminosidad LDR). Se entrena un perceptrón multicapa (`MiniNeuralNetwork`) que correlaciona estas variables simples para predecir la variable compleja deseada. Empaquetado con PTQ *Per-Channel*, el modelo opera con alta precisión consumiendo apenas unos pocos cientos de bytes de memoria Flash, funcionando durante años con una batería de litio pequeña.

#### D. Tiny Vision (Clasificación de Matrices Ópticas)

* **El Problema:** Detectar la presencia humana o gestos direccionales sin violar la privacidad utilizando cámaras de video estándar.
* **Solución MiniTensor:** Utilizar cámaras de muy baja resolución (ej. sensores térmicos de 8x8 o 24x24 píxeles). Mediante una arquitectura de `SeparableConv2D`, se reduce drásticamente el costo de las multiplicaciones matriciales gracias al *Operator Fusion*. El MCU no "ve" a una persona, sino una matriz térmica abstracta, infiriendo estados (ej. "Persona a la izquierda", "Sala vacía") sin procesar ni almacenar rostros o imágenes nítidas.

---

# Capítulo 4. Modelos y Capas de Deep Learning Embebido (MiniTensor)

El motor **MiniTensor** representa el salto arquitectónico de MiniML Engine hacia la Inteligencia Artificial compleja. A diferencia de los modelos *Legacy* (que operan sobre matrices estáticas y reglas condicionales), MiniTensor implementa un motor de diferenciación automática (*Autograd*) y un grafo computacional dinámico capaz de modelar topologías profundas.

El verdadero logro de ingeniería de MiniTensor no es solo entrenar estos modelos en Python, sino su capacidad de exportar estas capas matemáticas complejas a **C++ plano, predecible y optimizado para ejecutarse en SRAM de microcontroladores con menos de 2KB de capacidad**.

A continuación, se detalla el funcionamiento técnico, las matemáticas subyacentes y la optimización *bare-metal* de cada una de las capas soportadas por la API `miniml.nn`.



### 1. Capa `Linear` (Dense / Fully Connected)

* **Descripción Matemática:** Es la capa fundamental del Perceptrón Multicapa. Realiza una transformación lineal sobre los datos de entrada aplicando una matriz de pesos y un vector de sesgo.

$$Y = X \cdot W^T + B$$


* **Funcionamiento Técnico:** Cada neurona en esta capa está conectada a todas las activaciones de la capa anterior. Es excelente para aprender relaciones no espaciales y combinaciones lógicas de características extraídas.
* **Optimización en Edge (C++):** La matriz de pesos $W$ (que suele ser masiva) se extrae y se etiqueta con la directiva `PROGMEM` para vivir exclusivamente en la memoria Flash. El exportador de MiniML genera bucles `for` anidados que calculan el producto escalar leyendo directamente de la ROM. La RAM (SRAM) solo se utiliza para almacenar el pequeño vector de salida $Y$.



### 2. Capas `Conv1D` y `Conv2D` (Convoluciones)

* **Descripción Matemática:** Realizan la extracción de características locales deslizando un núcleo (*Kernel/Filter*) a través de la dimensión espacial (2D) o temporal (1D) de los datos de entrada.
* **Funcionamiento Técnico:**
* **`Conv1D`:** Ideal para procesar secuencias temporales, como lecturas de acelerómetros (vibración), señales de ECG o audio crudo.
* **`Conv2D`:** Diseñada para matrices espaciales, como imágenes térmicas, matrices de sensores de presión o cámaras ópticas de bajísima resolución (*Tiny Vision*).


* **Optimización en Edge (C++):** En frameworks tradicionales (como TensorFlow o PyTorch), la convolución a menudo se calcula usando algoritmos como `im2col` (Image to Column) para aprovechar multiplicaciones de matrices rápidas, lo cual duplica o triplica el consumo de RAM. **MiniTensor no hace esto.** El exportador a C++ de MiniML calcula la geometría de la convolución dinámicamente y genera bucles anidados precisos. Utiliza macros de lectura segura para multiplicar los tensores directamente contra los pesos en `PROGMEM` sin crear copias intermedias de la matriz de entrada, salvando la SRAM.



### 3. Capa `SeparableConv2D` (MobileNet-Style)

* **Descripción Matemática y Técnica:** Las convoluciones estándar son computacionalmente prohibitivas para microcontroladores sin FPU (Floating-Point Unit). `SeparableConv2D` factoriza una convolución estándar en dos operaciones más pequeñas y eficientes:
1. **Depthwise Convolution:** Aplica un solo filtro por cada canal de entrada (filtrado espacial).
2. **Pointwise Convolution:** Aplica una convolución $1 \times 1$ para combinar las salidas de la capa *depthwise* (filtrado por canales).


* **Optimización en Edge (Operator Fusion):** Es una de las joyas de la corona de MiniTensor. El exportador nativo de C++ implementa **Operator Fusion** (Fusión de Operadores). En lugar de calcular el paso *Depthwise*, guardarlo en la RAM, y luego calcular el paso *Pointwise*, el compilador fusiona ambas operaciones matemáticamente en el mismo ciclo de bucle.
* **Impacto:** Reduce la cantidad de multiplicaciones (ciclos de reloj) y la huella de memoria en órdenes de magnitud comparado con `Conv2D`. Es estrictamente obligatoria para procesamiento de imágenes en hardware tipo ESP32 o Cortex-M0.



### 4. Capas `MaxPool1D` y `MaxPool2D` (Submuestreo)

* **Descripción Matemática:** Realiza una reducción de dimensionalidad (downsampling) no paramétrica. Desliza una ventana sobre el tensor de entrada y extrae únicamente el valor máximo dentro de esa ventana.
* **Funcionamiento Técnico:** Sirve para dos propósitos críticos: lograr invariancia espacial a pequeñas traslaciones (si el patrón se mueve ligeramente, sigue siendo detectado) y reducir exponencialmente el número de parámetros que llegarán a las capas lineales finales.
* **Optimización en Edge (C++):** Al no tener pesos entrenables, no consume memoria Flash. Se implementa en C++ como un algoritmo de búsqueda de máximos con control de *Stride* (paso). La gestión de la ventana deslizante se calcula puramente con índices de punteros, resultando en una operación casi "gratuita" a nivel de memoria RAM.



### 5. Capa `Flatten` (Aplanado)

* **Descripción Matemática:** Transforma un tensor multidimensional (ej. `[Batch, Channels, Height, Width]`) en un vector unidimensional consecutivo `[Batch, N]`.
* **Funcionamiento Técnico:** Es el puente arquitectónico estricto entre el mundo de la extracción de características (Convoluciones/Pooling) y el mundo de la clasificación (Capas Lineales).
* **Optimización en Edge (C++):** **Costo Cero (Zero-Cost Operation).** En el C++ generado por MiniML, `Flatten` no ejecuta ninguna instrucción de copiado en memoria, ni reasigna variables, lo cual sería letal para la SRAM. Simplemente re-interpreta la forma matemática (*shape*) del puntero de memoria del tensor anterior para que la capa `Linear` pueda iterar sobre él linealmente.



### 6. Capa `ResidualBlock1D` (ResNet-Style)

* **Descripción Matemática:** Implementa una "Skip Connection" (conexión de salto). Matemáticamente, en lugar de que una capa intente aprender la transformación directa $\mathcal{H}(x)$, intenta aprender la función residual $\mathcal{F}(x)$, y la salida final se define como la suma con la identidad de la entrada:

$$Y = \mathcal{F}(x) + x$$


* **Funcionamiento Técnico:** Soluciona el problema del desvanecimiento del gradiente (*Vanishing Gradient*) en redes neuronales profundas. Permite construir detectores de señales temporales (audio, vibración) mucho más profundos, robustos y estables.
* **Optimización en Edge (C++):** Requiere una indexación geométrica estricta. El exportador C++ de MiniML se asegura en tiempo de compilación de que las dimensiones del tensor de entrada $x$ y el tensor procesado $\mathcal{F}(x)$ coincidan milimétricamente (a través de padding o proyecciones $1 \times 1$). La suma se realiza *in-place* sobre el tensor de salida, protegiendo al microcontrolador de picos de consumo dinámico de RAM que típicamente ocurren al sumar matrices en ramas paralelas.



### 7. Casos de Uso en la Práctica Física (Hardware Real)

Para comprender el verdadero poder de **MiniTensor**, es vital conectar la abstracción matemática de las capas con el hardware físico y las señales del mundo real. A diferencia de los modelos *Legacy* que evalúan lecturas estáticas, las topologías de Deep Learning están diseñadas para encontrar patrones ocultos en **ventanas de tiempo** o **matrices espaciales**.

Aquí se detalla cómo se comportan estas arquitecturas cuando se conectan a sensores físicos:

#### A. Análisis de Vibraciones y Acústica (Mantenimiento Predictivo)

* **El Problema Físico:** Un motor industrial genera un espectro de frecuencias complejo. Un acelerómetro (ej. MPU6050) conectado por I2C envía cientos de lecturas de los ejes X, Y, Z por segundo. Un modelo clásico fallaría al intentar analizar una sola lectura aislada.
* **Topología Ideal:** `Conv1D` $\rightarrow$ `MaxPool1D` $\rightarrow$ `ResidualBlock1D` $\rightarrow$ `Linear`.
* **Cómo funciona en el hardware:** El microcontrolador llena un *Buffer Circular* con, por ejemplo, 128 muestras temporales. La capa `Conv1D` desliza su núcleo (kernel) sobre este buffer para detectar micro-frecuencias anómalas (fricción, desgaste de rodamientos). El `ResidualBlock1D` asegura que la red sea lo suficientemente profunda para entender la diferencia entre el encendido del motor y una falla real, sin colapsar la SRAM.

#### B. Reconocimiento de Gestos Dinámicos (IMU / Wearables)

* **El Problema Físico:** Clasificar movimientos humanos complejos (ej. dibujar una "O" o una "Z" en el aire con un guante inteligente) procesando datos de giroscopios en hardware portátil alimentado por baterías de moneda.
* **Topología Ideal:** `Flatten` $\rightarrow$ `Linear` $\rightarrow$ `Linear` (con activaciones `ReLU`).
* **Cómo funciona en el hardware:** Las lecturas del acelerómetro y giroscopio se acumulan en una matriz 2D. La capa `Flatten` deforma esta matriz a un vector 1D instantáneamente (cero costo de RAM). Las capas `Linear` actúan como un perceptrón profundo, procesando la totalidad del movimiento para emitir una clasificación ("Gesto A", "Gesto B") en milisegundos.

#### C. Tiny Vision (Visión Artificial de Ultra-Baja Resolución)

* **El Problema Físico:** Detectar si hay una persona en una habitación, o monitorear puntos calientes en un tablero eléctrico, usando un microcontrolador que no tiene RAM para almacenar una foto normal en JPEG.
* **Topología Ideal:** `SeparableConv2D` $\rightarrow$ `MaxPool2D` $\rightarrow$ `SeparableConv2D` $\rightarrow$ `Linear`.
* **Cómo funciona en el hardware:** Se utiliza un arreglo de sensores térmicos (como el AMG8833 de 8x8 píxeles) o cámaras SPI minúsculas. La capa `SeparableConv2D` analiza los gradientes de temperatura o bordes espaciales aplicando *Operator Fusion*. Al separar la convolución, el microcontrolador (ej. ESP32) puede inferir la presencia de un patrón visual complejo gastando apenas una fracción de la RAM que exigiría una `Conv2D` tradicional.



### 8. ¿Para qué se utilizarían? (El Límite Arquitectónico)

Como arquitecto de software, la decisión de utilizar el módulo **MiniTensor** en lugar de un modelo **Legacy** debe basarse en la naturaleza de los datos:

* **Usa MiniTensor (Deep Learning) SI:** 1. Tus datos tienen dimensiones espaciales (matrices/imágenes) o secuenciales/temporales (audio, ventanas de vibración).
2. La relación entre las variables de entrada es altamente no lineal y extremadamente compleja.
3. Dispones de un microcontrolador con al menos 16 KB a 32 KB de memoria Flash libre para alojar los tensores paramétricos.
* **NO uses MiniTensor SI:**
1. Estás leyendo un solo sensor en un instante específico (ej. "Si la temperatura > 30°C"). Usa `DecisionTree`.
2. El microcontrolador destino es un chip ultra-limitado (ej. ATtiny85 con 512 bytes de SRAM).





### 9. Sintaxis de Entrenamiento (La API de MiniTensor)

Para diseñar e invocar el entrenamiento de estas redes neuronales complejas, **MiniTensor** provee una API secuencial limpia e intuitiva, fuertemente inspirada en los estándares de la industria, pero operando al 100% en Python puro y fácil de aprender.

Aquí se muestra el flujo técnico para ensamblar las capas, definir el optimizador y ejecutar el bucle de entrenamiento (*Training Loop*).

#### A. Definición de la Topología (Construcción del Grafo)

El contenedor `nn.Sequential` agrupa las capas y gestiona la propagación hacia adelante y hacia atrás automáticamente.

```python
from miniml import Tensor, nn, optim

# Definición de un modelo ResNet-Style para análisis de vibración (1D)
modelo_edge = nn.Sequential([
    # Capa de extracción temporal: 1 canal de entrada (ej. Eje X), 4 filtros
    nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
    nn.ReLU(),
    
    # Reducción de dimensionalidad matemática
    nn.MaxPool1d(kernel_size=2, stride=2),
    
    # Bloque Residual para aprendizaje profundo sin desvanecimiento de gradiente
    nn.ResidualBlock1D(in_channels=4, out_channels=4),
    
    # Transición al mundo de la clasificación (Zero-Cost Operation)
    nn.Flatten(),
    
    # Capas Lineales (Asumiendo que el Flatten resulta en 16 características)
    nn.Linear(in_features=16, out_features=8),
    nn.ReLU(),
    
    # Salida binaria (ej. 0 = Motor Normal, 1 = Falla Mecánica)
    nn.Linear(in_features=8, out_features=1),
    nn.Sigmoid()
])

```

#### B. Invocación del Entrenamiento (El Bucle Autograd)

Dado que MiniTensor maneja el grafo dinámico, tú tienes el control absoluto sobre el ciclo de optimización.

```python
# 1. Definir Función de Pérdida (Loss) y Optimizador
criterio = nn.MSELoss() # O CrossEntropyLoss para clasificación multiclase
optimizador = optim.SGD(modelo_edge.parameters(), lr=0.01)

epocas = 100

print("Iniciando Entrenamiento Zero-Dependency...")

for epoca in range(epocas):
    # Asumiendo 'X_train' (Tensores de entrada) y 'Y_train' (Tensores objetivo)
    
    # Paso 1: Forward Pass (Calcular la predicción)
    predicciones = modelo_edge(X_train)
    
    # Paso 2: Calcular el Error (Loss)
    perdida = criterio(predicciones, Y_train)
    
    # Paso 3: Zero Grad (Limpiar gradientes del ciclo anterior)
    optimizador.zero_grad()
    
    # Paso 4: Backward Pass (Motor Autograd calcula derivadas por regla de la cadena)
    perdida.backward()
    
    # Paso 5: Optimización (Actualizar los pesos en memoria)
    optimizador.step()
    
    if epoca % 10 == 0:
        print(f"Época {epoca}/{epocas} | Error: {perdida.data:.4f}")

```

#### C. Preparación y Exportación a Edge

Una vez que el motor Autograd ha convergido y el error es mínimo, se invoca la infraestructura de Edge AI para su despliegue físico.

```python
from miniml.exporters import cpp_writer
from miniml.exporters.library_packer import LibraryPackager

# 1. Compresión Híbrida INT8 (Drástica reducción de Flash)
modelo_edge.quantize()

# 2. Transpilación del grafo matemático a C++ plano
# Se debe proveer la forma exacta del tensor de entrada (Batch, Channels, Length)
codigo_cpp = cpp_writer.generate_cpp_code(modelo_edge, input_shape=(1, 1, 32))

# 3. Empaquetado Industrial para PlatformIO / Arduino
LibraryPackager.create_arduino_zip(
    model_name="VibrationResNet",
    cpp_code=codigo_cpp,
    version="1.0.0",
    quantized=True
)

```


### 10. Prácticas Apropiadas (Best Practices) para el Manejo de Capas en Edge

Diseñar redes neuronales para hardware embebido no es igual a diseñar para la nube. En la nube, un error de dimensionamiento resulta en unos milisegundos extra de latencia; en un microcontrolador, resulta en un colapso total del sistema (*Hard Fault* o *Stack Overflow*).

Para garantizar que tus topologías creadas en **MiniTensor** sobrevivan la transición al silicio físico, debes adoptar una mentalidad de ingeniería de bajo nivel. Aquí se detallan las reglas de oro para la manipulación de capas:

#### A. El Peligro del Aplanado Prematuro (Flattening)

La capa `Linear` (Densa) es, con diferencia, la que más memoria Flash consume, ya que cada neurona se conecta con todas las entradas posibles. Si aplicas la capa `Flatten` demasiado pronto, destruirás la memoria de la placa.

* **El Error Común:** Aplicar una convolución a una ventana temporal de 256 muestras con 8 filtros, y pasarla directamente a un `Linear` mediante `Flatten`. Esto genera un vector de $256 \times 8 = 2048$ características. Una capa oculta de apenas 32 neuronas requeriría **65,536 pesos** (más de 65 KB, superando la capacidad completa de muchos chips).
* **La Práctica Correcta:** Utiliza `MaxPool1D` o `MaxPool2D` de forma agresiva para reducir la dimensionalidad espacial/temporal *antes* de aplanar. Debes asegurarte de que el tensor resultante que entra al `Flatten` sea lo más pequeño posible (idealmente $< 128$ características).

#### B. Gestión de Picos de SRAM (Stride vs. Pooling)

Cada vez que la red pasa de una capa a otra, el microcontrolador debe reservar SRAM para la matriz resultante de la capa actual antes de borrar la anterior.

* **El Cuello de Botella:** Si usas una `Conv1D` con `stride=1` (paso de 1) seguida de un `MaxPool1D` de `kernel=2`, el microcontrolador primero debe guardar la matriz de alta resolución completa en SRAM, para luego reducirla a la mitad en el siguiente paso.
* **La Práctica Correcta (Hardware-Aware):** Si estás muy al límite de memoria dinámica, omite el `MaxPool` y configura la convolución con `stride=2`. Esto obliga a la capa `Conv1D` o `SeparableConv2D` a calcular la extracción de características y el submuestreo simultáneamente, instanciando directamente un tensor con la mitad del tamaño en la SRAM.

#### C. Apilamiento de Núcleos Pequeños (Receptive Field)

A diferencia de los procesadores de escritorio, los microcontroladores sufren para procesar núcleos de convolución grandes (como 7x7 o 11x11) debido al costo exponencial de las multiplicaciones.

* **La Práctica Correcta:** Para aumentar el "campo receptivo" (qué tanto de la señal "ve" la red al mismo tiempo), es matemáticamente y computacionalmente más eficiente apilar dos capas `Conv1D` con núcleos pequeños de `kernel_size=3`, en lugar de una sola con `kernel_size=5`. Obtienes la misma cobertura espacial, pero reduces drásticamente la cantidad de parámetros en `PROGMEM` y operaciones por ciclo.

#### D. Geometría Estricta en Bloques Residuales

La capa `ResidualBlock1D` suma matemáticamente la entrada original al resultado de las transformaciones internas ($y = \mathcal{F}(x) + x$).

* **La Regla de Oro:** En MiniTensor, el número de canales de entrada (`in_channels`) **debe** ser estrictamente igual al número de canales de salida (`out_channels`) al usar un bloque residual básico. Dado que el compilador C++ no tiene recolección de basura (*Garbage Collection*) ni redimensionamiento dinámico seguro de arreglos, intentar sumar dos tensores de distinto tamaño arrojará un error de compilación. Si necesitas cambiar la profundidad del canal, debes usar convoluciones de transición explícitas (proyecciones de 1x1).

#### E. Aislamiento del Bucle Principal (Non-Blocking Inference)

En la práctica física, la capa matemática no debe paralizar el microcontrolador.

* **La Práctica Correcta:** Nunca llames al método `predict()` (el cual atraviesa todas las capas de MiniTensor) dentro de una interrupción de hardware (ISR). Las funciones de las capas convolucionales toman milisegundos, y bloquear una interrupción detendrá los temporizadores del sistema, el WiFi (en el caso de un ESP32) o el bus I2C. Acumula los datos del sensor pasivamente usando variables `volatile` en la interrupción, y ejecuta la inferencia exclusivamente en el contexto seguro del `loop()` principal.



### 11. Funciones de Activación y Pérdida (El Motor Matemático)

En el corazón de **MiniTensor** residen las funciones de activación y pérdida. Las primeras son responsables de inyectar no linealidad en la red (permitiendo que aprenda patrones complejos en lugar de simples líneas rectas), mientras que las segundas calculan el error para guiar al motor Autograd durante el entrenamiento.

Es crucial entender una distinción arquitectónica clave en el framework: las **Funciones de Activación** se exportan al microcontrolador (C++) para la inferencia, mientras que las **Funciones de Pérdida (Loss)** viven casi exclusivamente en el PC (Python) durante la fase de optimización de pesos, a menos que se implemente aprendizaje en el dispositivo (*On-Device Learning*).

A continuación, se detalla la matemática y la ingeniería de bajo nivel detrás de cada una:


#### A. `ReLU` (Rectified Linear Unit)

* **Descripción Matemática:** Es la función de activación más utilizada en Deep Learning. Filtra los valores negativos, dejándolos en cero, y permite el paso de los valores positivos sin alterarlos.

$$f(x) = \max(0, x)$$



*Derivada (Autograd):* $f'(x) = 1$ si $x > 0$, de lo contrario $0$.
* **Funcionamiento Técnico:** Al evitar la saturación en valores positivos, soluciona el problema del desvanecimiento del gradiente (*Vanishing Gradient*) en redes profundas como las que usan `ResidualBlock1D`.
* **Optimización en Edge (C++):** Es la función "reina" del Edge Computing. Computacionalmente, tiene un costo casi **nulo**. No requiere multiplicaciones ni divisiones, solo una simple instrucción de bifurcación condicional (`if x > 0`).
```cpp
// Implementación C++ generada por MiniML
float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

```


Es 100% segura frente a desbordamientos aritméticos (*Overflows*) en microcontroladores de 8-bits.


#### B. `Sigmoid` (Función Sigmoidea)

* **Descripción Matemática:** Aplasta cualquier número real en un rango estricto entre $0$ y $1$, dándole forma de curva "S".

$$f(x) = \frac{1}{1 + e^{-x}}$$



*Derivada (Autograd):* $f'(x) = f(x) \cdot (1 - f(x))$
* **Funcionamiento Técnico:** Se utiliza predominantemente en la **última capa** de la red para problemas de clasificación binaria (ej. `0 = Normal`, `1 = Falla`). El valor resultante puede interpretarse como una probabilidad (ej. `0.85` = 85% de certeza).
* **Optimización en Edge (C++):** A diferencia de ReLU, Sigmoid es **peligrosa en hardware embebido**. La función exponencial ($e^{-x}$) es matemáticamente muy costosa si el microcontrolador no tiene FPU (Unidad de Coma Flotante). Además, si $x$ es un número negativo muy grande, el cálculo en 8-bits colapsará en `NaN` (Not a Number).
* **Solución MiniML:** El exportador C++ implementa **Clipping (Recorte) Matemático**. Antes de calcular el exponente, el valor de $x$ se recorta a límites seguros (típicamente entre $-15$ y $15$) para evitar el colapso del FPU por software del microcontrolador.


#### C. `MSELoss` (Mean Squared Error - Error Cuadrático Medio)

* **Descripción Matemática:** Mide el promedio de los cuadrados de los errores, es decir, la diferencia matemática entre el valor predicho por la red ($\hat{y}_i$) y el valor real esperado ($y_i$).

$$L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$


* **Funcionamiento Técnico (Python):** Se utiliza estrictamente para tareas de **Regresión** (predicción de variables continuas, como estimar la temperatura exacta o el voltaje restante de una batería).
* **Uso en Autograd:** Penaliza drásticamente las predicciones que están muy alejadas de la realidad debido al cuadrado numérico. Esto obliga al optimizador (SGD) a hacer correcciones agresivas en los pesos durante las primeras épocas del entrenamiento.
* **Uso en Edge:** Raramente se exporta a C++, a menos que se esté diseñando un *Autoencoder* para detectar anomalías donde la placa deba calcular qué tan diferente es la señal reconstruida de la original.


#### D. `CrossEntropyLoss` (Pérdida de Entropía Cruzada)

* **Descripción Matemática:** Mide el rendimiento de un modelo de clasificación evaluando la divergencia entre dos distribuciones de probabilidad (la real y la predicha). Usualmente se asume que las predicciones han pasado por una función Softmax o Sigmoid.

$$L = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$



*(Donde $C$ es el número de clases, $y_i$ es el indicador binario si $i$ es la clase correcta, y $\hat{y}_i$ es la probabilidad predicha).*

* **Funcionamiento Técnico (Python):** Es el estándar absoluto para **Clasificación Multiclase** (ej. reconocer 5 gestos distintos con un guante IMU). La entropía cruzada penaliza severamente al modelo si este predice con alta probabilidad una clase incorrecta (debido al logaritmo).
* **Optimización en Edge (C++):** En el mundo real del Edge Computing, el cálculo de la entropía cruzada se ignora por completo en la fase de inferencia. El microcontrolador solo necesita saber "cuál es la clase ganadora". Por lo tanto, el exportador C++ omite la función logarítmica y simplemente implementa una búsqueda de máximo (`argmax`) sobre las salidas puras de la red, ahorrando cientos de ciclos de reloj.



### 12. Guía de Arquitectura: Sinergia entre Capas y Activaciones

Diseñar una red neuronal para sistemas embebidos no consiste en apilar capas al azar. Para que **MiniTensor** logre la convergencia matemática en el PC y sobreviva a las limitaciones de memoria (SRAM) en el microcontrolador, la topología debe seguir un diseño de "embudo" (Funnel Design).

A continuación, se presenta la guía técnica para ensamblar el grafo computacional combinando las capas y activaciones disponibles.

#### A. El Patrón Estándar de Extracción y Decisión

La arquitectura de una red profunda en MiniTensor siempre debe dividirse en dos bloques conceptuales: el **Bloque de Extracción de Características** (Feature Extractor) y el **Bloque de Clasificación** (Classifier).

1. **Extracción (Las Convoluciones):** Utiliza `Conv1D`, `Conv2D` o `SeparableConv2D` para leer la señal cruda.
* *Regla de Oro:* Siempre sigue una capa convolucional con una activación `ReLU`. Las convoluciones son operaciones lineales; sin `ReLU`, apilar múltiples convoluciones equivaldría matemáticamente a una sola convolución gigante, desperdiciando recursos y limitando el aprendizaje.


2. **Reducción (El Submuestreo):** Inmediatamente después del `ReLU`, utiliza `MaxPool1D` o `MaxPool2D`. Esto reduce la dimensionalidad espacial a la mitad, aliviando la carga de SRAM para la siguiente capa.
3. **El Puente (Aplanado):** Usa la capa `Flatten` únicamente cuando las dimensiones espaciales o temporales sean lo suficientemente pequeñas (ej. una matriz resultante de $4 \times 4$). Esta capa tiene costo cero en memoria y prepara el tensor para el bloque de clasificación.
4. **Decisión (Perceptrón Multicapa):** Cierra la red con una o dos capas `Linear`. La capa final dictamina la salida del modelo. Si es clasificación binaria, la última capa `Linear` debe tener `out_features=1` y estar seguida incondicionalmente por una activación `Sigmoid` para aplastar la salida a un rango de probabilidad $[0, 1]$.

#### B. Sinergia de Topologías: Casos de Uso Reales

* **Caso 1: Detector de Anomalías Acústicas (Ej. Micrófono I2S en ESP32)**
* *Flujo:* `Conv1D` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool1D` $\rightarrow$ `ResidualBlock1D` $\rightarrow$ `Flatten` $\rightarrow$ `Linear` $\rightarrow$ `Sigmoid`.
* *Sinergia:* El `ResidualBlock1D` se coloca *después* del `MaxPool1D`. Al hacerlo, el bloque residual procesa un tensor más pequeño, permitiendo a la red aprender patrones profundos de frecuencias de audio anómalas sin colapsar la RAM dinámica durante el *Forward Pass*.


* **Caso 2: Detección de Gestos con IMU (Ej. Acelerómetro MPU6050)**
* *Flujo:* `Linear` $\rightarrow$ `ReLU` $\rightarrow$ `Linear` $\rightarrow$ `ReLU` $\rightarrow$ `Linear` (Clasificación).
* *Sinergia:* Para ventanas de tiempo muy cortas (ej. 10 muestras), las convoluciones pueden ser excesivas. Un bloque puramente denso, enrutado a través de activaciones `ReLU` para evitar la saturación matemática, puede clasificar trayectorias espaciales complejas con una latencia inferior a 2 milisegundos.


* **Caso 3: Visión Térmica para Detección de Presencia (Ej. Sensor AMG8833 8x8)**
* *Flujo:* `SeparableConv2D` $\rightarrow$ `ReLU` $\rightarrow$ `SeparableConv2D` $\rightarrow$ `ReLU` $\rightarrow$ `Flatten` $\rightarrow$ `Linear`.
* *Sinergia:* Al usar cámaras de ultra-baja resolución, el `MaxPool2D` destruiría los pocos píxeles de información útil. En su lugar, se apilan capas `SeparableConv2D` consecutivas. Gracias al *Operator Fusion*, el microcontrolador evalúa los bordes térmicos minimizando las operaciones matriciales.



### 13. Consideraciones Críticas de Deep Learning Embebido

* **Picos de SRAM Transitorios:** El momento de mayor peligro en el microcontrolador ocurre durante la transición entre capas (ej. de `Conv1D` a `MaxPool1D`). El C++ generado debe instanciar el tensor resultante antes de liberar la memoria del anterior. Mantén el tamaño del lote (Batch Size) estrictamente en `1` para la inferencia, y audita el estimador de memoria del CLI antes de flashear.
* **Alineación Geométrica:** Las capas estáticas de C++ asumen que el buffer de entrada tiene exactamente las dimensiones con las que el modelo fue entrenado en Python. Si entrenaste la red esperando una ventana de 64 lecturas, enviarle un arreglo de 60 o 65 lecturas provocará una desalineación de punteros, leyendo basura de la memoria y resultando en un colapso del sistema o predicciones sin sentido.



### 14. Frontera de Responsabilidad: Matemáticas vs. Realidad Física (Condiciones de Uso)

**MiniML Engine** es un motor matemático determinista. El framework y su creador garantizan la estabilidad algorítmica, la gestión de memoria (*PROGMEM* y SRAM) y la precisión de la inferencia a nivel de código máquina.

Si un modelo cuantizado arroja una predicción de $0.8520$ en el simulador del PC o en un emulador a nivel de instrucciones, **MiniML garantiza que el silicio físico arrojará exactamente $0.8520$ dados los mismos valores de entrada.**

Sin embargo, el mundo físico está sujeto a las leyes de la termodinámica y el electromagnetismo, dominios que escapan al control de cualquier software. **No se hace responsable a MiniML Engine por fallos, predicciones erróneas o accidentes en prototipos físicos derivados de anomalías externas de hardware.** 

#### A. Lo que NO garantiza el framework (Anomalías Físicas)

1. **Ruido del Conversor Analógico-Digital (ADC):** MiniML asume señales limpias. Si tu sensor de temperatura inyecta picos parásitos, ruido blanco eléctrico o sufre interferencia electromagnética (EMI) de motores cercanos, el modelo predecirá sobre "basura" numérica.
2. **Caídas de Tensión (Brownouts):** El encendido de relés o servos provoca caídas momentáneas en el voltaje de la placa (ej. de 5V a 4.2V). Esto altera la lectura física del sensor exactamente en el milisegundo en que la IA ingiere la matriz de datos, corrompiendo la inferencia.
3. **Latencia de Buses (I2C/SPI) e Integridad de Cableado:** Un cable suelto, una resistencia *pull-up* incorrecta, o un retraso en la lectura del protocolo I2C desfasarán la ventana temporal de los datos. La capa `Conv1D` perderá la coherencia secuencial de la señal.

#### B. Condiciones de Uso de la IA Embebida

El integrador (el ingeniero de hardware o desarrollador de firmware) asume la responsabilidad total de entregar datos estables a la función `predict()`. Para que la IA embebida funcione según los estándares industriales, es **obligatorio** cumplir con las siguientes condiciones de acondicionamiento en el código C++ principal:

* Implementar filtros de hardware o software (Filtros RC paso bajo, *Debounce*, Medias Móviles) *antes* de que la matriz llegue a la red neuronal.
* Utilizar las rutinas de normalización generadas por el `MiniScaler` del framework de manera estricta para asegurar que la magnitud física del mundo real encaje en el espacio latente del modelo entrenado.
* Diseñar fuentes de alimentación aisladas y robustas para los sensores analógicos, separando la lógica de control de la carga de potencia.

---

# Capítulo 5. Módulo de Hardware y Simulación (Aún en fase de experimentación)

Aunque la filosofía de **MiniML Engine** es compilar y exportar el modelo matemático a C++ para que el microcontrolador opere de forma completamente autónoma (desconectado del PC), existe una fase crítica en todo proyecto de Machine Learning: **la recolección de datos y la validación del prototipo**.

Para cubrir esta etapa, el framework integra dos herramientas alojadas en el PC anfitrión (Host): `serial_manager.py` y `virtual_sensor.py`. Estos scripts en Python actúan como el puente de comunicación y simulación entre el ecosistema matemático de la PC y el silicio físico.

A continuación, se detalla la arquitectura de estos módulos, cómo utilizarlos en escenarios reales y, lo más importante, dónde termina la responsabilidad del software y comienza la del hardware.



### 1. `serial_manager.py` (Ingesta de Datos Físicos)

* **¿Qué es y cómo funciona?**
Es un gestor de comunicaciones UART (Puerto Serie). Su función principal es "escuchar" el bus serial (USB) al que está conectado el microcontrolador (ej. `COM3` en Windows o `/dev/ttyUSB0` en Linux) y capturar el flujo de datos que el hardware está midiendo en tiempo real.
* **Detalles Técnicos:**
El script está diseñado para decodificar cadenas de bytes (`utf-8`) enviadas por la placa mediante comandos clásicos como `Serial.println()`. Cuenta con un analizador (parser) interno que toma cadenas separadas por comas (formato CSV en crudo, ej. `25.4, 60.1, 1024`) y las transforma automáticamente en arreglos (listas nativas de Python) o matrices bidimensionales listas para ser ingeridas por la función `.fit()` de MiniML.
* **Separación Software/Hardware:**
El `serial_manager.py` **no controla** el sensor. Solo lee un buffer de memoria en la PC. El microcontrolador es el único responsable de configurar el ADC, interrogar al sensor vía I2C/SPI y empaquetar la cadena de texto a la velocidad correcta (Baud Rate, ej. `115200`). Si el microcontrolador envía datos corruptos, el gestor de Python simplemente registrará basura numérica.

### 2. `virtual_sensor.py` (Simulación Determinista)

* **¿Qué es y cómo funciona?**
Es un generador de señales sintéticas. Permite a los arquitectos de software probar, depurar y validar topologías de **MiniTensor** o modelos Legacy *sin* necesidad de tener la placa física conectada o los sensores electrónicos comprados.
* **Detalles Técnicos:**
El módulo inyecta funciones matemáticas (senoidales, ondas cuadradas, rampas) y les aplica perturbaciones estadísticas (ruido Gaussiano o picos aleatorios) para imitar las imperfecciones del mundo real.
Por ejemplo, puedes pedirle al sensor virtual que genere 1000 muestras de una "onda de vibración normal" y 200 muestras de una "vibración anómala de alta frecuencia". Esto genera instantáneamente un dataset en la PC para entrenar tu modelo.
* **Separación Software/Hardware:**
Los datos de `virtual_sensor.py` son matemáticamente perfectos dentro de su aleatoriedad controlada. Sirven para probar si la red neuronal *puede* aprender un patrón. Sin embargo, un sensor real sufrirá de deriva térmica y degradación electromecánica, factores que el sensor virtual no puede modelar con precisión absoluta.



### 3. Casos de Uso Reales en el Ciclo de Desarrollo

La combinación de estos dos módulos permite un flujo de trabajo iterativo y seguro (Hardware-in-the-Loop simulado):

#### A. Recolección de Datasets (Data Harvesting)

* **Escenario:** Quieres crear un `DecisionTreeClassifier` que detecte el riesgo de incendio usando un Arduino con un sensor de temperatura DHT22 y un sensor de gas MQ-2.
* **Uso:** Escribes un código simple en Arduino que imprima por serial: `Temp,Gas,Clase`. Enciendes un encendedor cerca de los sensores para simular peligro. En la PC, ejecutas `serial_manager.py`, el cual graba esta transmisión en vivo y construye el conjunto de datos estructurado de forma automática. Luego, le pasas ese conjunto a MiniML para entrenar el modelo y exportarlo a C++.

#### B. Validación Pre-Despliegue (Sanity Check)

* **Escenario:** Acabas de diseñar una topología compleja en MiniTensor (`SeparableConv2D` $\rightarrow$ `Linear`) pero quieres asegurarte de que la arquitectura converge antes de exportarla a la memoria Flash del ESP32.
* **Uso:** Utilizas `virtual_sensor.py` (O mediante el CLI de MiniML con el comando "sensor") para inyectar una matriz de datos espaciales con ruido artificial. Si el motor Autograd de MiniML no logra reducir la pérdida (`Loss`) con datos sintéticos perfectos, sabrás de inmediato que tu topología está mal diseñada, ahorrándote horas de depuración en la placa física.

#### C. Inferencia en el Host (Monitorización Serial)

* **Escenario:** El microcontrolador es tan limitado (ej. ATtiny85) que no puede correr la inferencia, o simplemente quieres usar el microcontrolador solo como una tarjeta de adquisición de datos (DAQ).
* **Uso:** El microcontrolador envía los datos crudos por USB. El `serial_manager.py` los recibe, se los pasa al modelo entrenado que está **ejecutándose en Python en la PC**, y la PC es la que toma la decisión o dibuja la gráfica.



### 4. Limitaciones Estrictas (Lo que el Módulo NO puede hacer)

Es fundamental entender los límites de la ingeniería del software cuando choca con la física de los sistemas embebidos. **No debes culpar al framework MiniML si ocurren los siguientes fallos:**

1. **Problemas de Baud Rate y Sincronización:**
* El `serial_manager.py` asume que el puerto serie de la PC y el código C++ del microcontrolador están configurados a la **misma velocidad** (ej. `9600` o `115200` baudios). Si hay un desajuste, Python recibirá caracteres extraños (``), y el script fallará al intentar parsear flotantes. Esto es un error de configuración física, no del framework.


2. **Latencia del Bus USB:**
* El módulo serial en Python **no es un sistema en tiempo real (RTOS)**. El sistema operativo (Windows/Linux) agrupa los paquetes USB antes de entregárselos a Python. Si intentas enviar datos a 10,000 muestras por segundo desde el microcontrolador, el `serial_manager.py` no podrá procesarlos uno por uno instantáneamente; se llenará el buffer y habrá latencia o pérdida de paquetes (Drop).


3. **Imposibilidad de Diagnóstico Eléctrico:**
* Si un cable GND (tierra) está suelto en tu protoboard, el sensor enviará datos ruidosos o valores al máximo de la escala (ej. `1023` en un ADC de 10 bits). El `serial_manager.py` leerá ese `1023` obedientemente. El software no tiene forma de saber que el hardware está defectuoso; para el modelo, es simplemente un dato más.


4. **No es un Flasheador/Programador:**
* El módulo de hardware **no sube (flashea) el código C++ a la placa**. Su función es capturar datos. El empaquetador de MiniML te entrega un `.zip`; es tu responsabilidad usar el IDE de Arduino, PlatformIO o `avrdude` para compilar y quemar ese binario en el silicio físico.

---

# Capítulo 6. Exportación y Empaquetado C++

La fase de exportación es el núcleo tecnológico que separa a **MiniML Engine** de los frameworks de Inteligencia Artificial tradicionales. Mientras que librerías como TensorFlow Lite o PyTorch Mobile requieren compilar un "intérprete" pesado dentro del microcontrolador para leer un archivo de modelo (`.tflite` o `.pt`), MiniML elimina por completo la necesidad de un intérprete.

El framework realiza una **transpilación estricta (Ingeniería Inversa)**: toma la topología matemática y los pesos entrenados en Python, y escribe un código fuente nativo en **C++ plano, estático y determinista**.

A continuación, se detalla el funcionamiento técnico interno de este proceso y cómo prepara el terreno para el flasheo en el hardware físico.



### 1. El Proceso de Transpilación a C++ (Paso a Paso)

Cuando invocas la función de exportación (`cpp_writer.py`), el framework ejecuta una secuencia de operaciones críticas para asegurar que el modelo se adapte al "Bare Metal" (hardware sin sistema operativo).

#### Fase A: Extracción y Desacoplamiento (Stripping)

En la PC, tu modelo (ej. `MiniNeuralNetwork` o un modelo Autograd de `MiniTensor`) contiene una inmensa cantidad de metadatos: historiales de gradientes, hiperparámetros del optimizador (SGD) y objetos dinámicos de Python.

* El exportador elimina toda esta información (ya que no se entrena en la placa).
* Extrae únicamente los **parámetros congelados** (matrices de pesos y vectores de sesgo) y el **grafo de topología** (el orden exacto de las capas que debe atravesar la señal).

#### Fase B: Aplanamiento Matemático (Static Arrays)

C++ en sistemas embebidos odia la memoria dinámica (`malloc`, `new`, o `std::vector`). La fragmentación del *Heap* (SRAM) es la causa principal de que los microcontroladores se cuelguen después de unas horas de funcionamiento.

* Para evitar esto, el módulo `ml_exporter.py` aplana todas las matrices multidimensionales (2D, 3D) en **arreglos unidimensionales (1D) estáticos de tamaño fijo**.
* El tamaño exacto se calcula en tiempo de exportación y se quema (hardcoded) en el código C++, permitiendo al compilador conocer exactamente cuánta memoria física consumirá el modelo antes de subirlo a la placa.

#### Fase C: Generación de la Lógica de Inferencia (`predict`)

El exportador no genera un código genérico; escribe una función `predict()` **a la medida exacta de tu modelo**.

* **Para un DecisionTree:** Genera arreglos paralelos estáticos y un simple bucle `while` que navega por índices mediante instrucciones `if/else`.
* **Para MiniTensor (Deep Learning):** Genera múltiples funciones anidadas. Si tu red tiene una capa `Conv1D`, el exportador escribe explícitamente los bucles `for` anidados para esa convolución específica, inyectando las macros de matemáticas necesarias (ej. multiplicación de punto fijo o de-cuantización al vuelo) y dimensionando los tensores intermedios con la directiva estática de C++ para proteger la pila (Stack).



### 2. El Puente hacia el Flasheo (IDE y Compilador)

Es vital comprender una limitación arquitectónica de diseño: **MiniML Engine genera código fuente (.cpp / .h), no binarios ejecutables (.hex / .bin).**

El framework no asume qué placa específica estás utilizando (podría ser un ATmega328P de Arduino, un ESP32 de Espressif, o un STM32 de STMicroelectronics). Por lo tanto, no se encarga de "subir" o flashear el código al microcontrolador.

El flujo de trabajo exacto desde la PC hasta el hardware es el siguiente:

1. **Python (PC):** MiniML finaliza su ejecución y guarda los archivos `.cpp` y `.h` (o el archivo `.zip` empaquetado) en tu directorio local.
2. **Importación al IDE:** El desarrollador toma esta librería generada y la incluye en su entorno de desarrollo preferido (**Arduino IDE** para principiantes y *makers*, o **PlatformIO** para ingeniería industrial).
3. **Compilación (Cross-Compilation):** El compilador de C++ del IDE (usualmente GCC-AVR o GCC-ARM) toma el código crudo de MiniML, lo somete a optimizaciones extremas (`-O2` o `-O3`), y lo enlaza (linker) con las librerías específicas del hardware para manejar los sensores.
4. **Flasheo:** El IDE se comunica con el programador de la placa vía USB (UART) y graba el binario final en el silicio.

Al generar código fuente estándar de C++, MiniML asegura una portabilidad absoluta. Si compila en un Arduino Uno de 16MHz, compilará y se ejecutará de forma exponencialmente más rápida en un ESP32 de 240MHz, sin tener que cambiar ni una sola línea de la configuración del modelo de Inteligencia Artificial.



### 3. La Fusión de la Cuantificación con la Exportación C++

El puente entre el entrenamiento en alta precisión (PC) y la ejecución restringida (Edge) se materializa a través del módulo `quantizer.py` de **MiniTensor**. Durante la exportación a C++, la cuantificación no es simplemente un redondeo de números; es una reestructuración profunda de cómo el microcontrolador gestionará sus registros de memoria en tiempo de ejecución.

A continuación, se detalla la mecánica interna de esta transformación, paso a paso, desde la intercepción de los tensores en Python hasta la generación del binario en C++.

#### A. El Rol de `quantizer.py` como Middleware

Cuando se invoca el proceso de exportación de un modelo cuantizado, el archivo `quantizer.py` actúa como un analizador estático sobre el grafo computacional de MiniTensor:

1. **Extracción de Parámetros:** El cuantizador recorre la topología (`nn.Sequential`) aislando las capas paramétricas (`Linear`, `Conv1D`, `SeparableConv2D`). Las capas sin estado (como `ReLU` o `MaxPool`) se ignoran en esta fase.
2. **Cálculo de Escalas Estáticas:** Extrae la matriz flotante de pesos ($W$) y determina el factor de escala máximo ($S$) necesario para mapear ese dominio dentro del rango estricto de un entero con signo de 8 bits ($-127$ a $127$). Si la cuantificación es *Per-Channel* (por defecto), calcula un vector unidimensional de escalas ($S_c$), una por cada filtro o neurona.
3. **Inyección de Metadatos:** Una vez comprimidos los pesos, el módulo etiqueta los tensores de MiniTensor con un flag interno (`is_quantized = True`) y adjunta los factores de escala al objeto de la capa, preparándolos para el transpilador de C++.

#### B. Generación Estática (Traducción a C++)

El módulo `cpp_writer.py` lee los metadatos dejados por `quantizer.py` y altera drásticamente su plantilla de generación de código.

En lugar de exportar enormes arreglos de tipo `float` (que consumen 4 bytes por parámetro), el exportador escribe las matrices utilizando el tipo de dato `int8_t` (1 byte por parámetro) y las blinda con la directiva `PROGMEM`.

**Ejemplo de Transpilación de una Capa `Linear` (C++ Generado):**

```cpp
// 1. Matriz de Pesos Cuantizada (Ocupa un 75% menos de ROM)
const int8_t capa1_pesos[32] PROGMEM = {
    112, -45, 8, 126, -110, 0, 34, ...
};

// 2. Factores de Escala (Mantenidos en Float32 para precisión)
// Al ser Per-Channel, hay una escala por cada neurona de salida
const float capa1_escalas[4] PROGMEM = {
    0.00342f, 0.00198f, 0.00511f, 0.00289f
};

// 3. Sesgos (Biases) (Mantenidos en Float32, su impacto en RAM/ROM es mínimo)
const float capa1_sesgos[4] PROGMEM = {
    -0.12f, 0.55f, 0.03f, -1.04f
};

```

Esta arquitectura garantiza que el *payload* (el peso muerto del modelo) resida exclusivamente en el almacenamiento físico, sin tocar la RAM estática del microcontrolador.

#### C. Lógica de Inferencia: De-cuantificación "Al Vuelo"

El verdadero desafío técnico resuelto por el exportador de MiniML es **cómo realizar los cálculos matemáticos sin descomprimir toda la matriz en la SRAM**.

Si el C++ tomara la matriz de `int8_t` y la copiara a un arreglo temporal de `float`, el microcontrolador sufriría un *Stack Overflow* instantáneo. Para evitarlo, el C++ generado implementa un patrón de **De-cuantificación Just-in-Time (Al Vuelo)** a nivel de registro.

**El Bucle C++ Exportado:**

```cpp
void predict_capa1(const float* input, float* output) {
    int peso_idx = 0;
    
    // Iterar sobre cada neurona de salida (Canal)
    for (int out_n = 0; out_n < 4; out_n++) {
        float suma = 0.0f;
        
        // Leer la escala específica para este canal desde la Flash
        float escala_actual = pgm_read_float_near(&capa1_escalas[out_n]);
        
        // Producto escalar (Dot Product)
        for (int in_n = 0; in_n < 8; in_n++) {
            // 1. Leer UN SOLO BYTE de peso desde la Flash (O(1) en RAM)
            int8_t peso_q = pgm_read_byte_near(&capa1_pesos[peso_idx]);
            
            // 2. De-cuantizar reconstruyendo el flotante localmente
            float peso_float = (float)peso_q * escala_actual;
            
            // 3. Acumular la multiplicación
            suma += input[in_n] * peso_float;
            peso_idx++;
        }
        
        // Sumar el sesgo y aplicar activación (ej. ReLU)
        suma += pgm_read_float_near(&capa1_sesgos[out_n]);
        output[out_n] = (suma > 0) ? suma : 0.0f;
    }
}

```

#### D. Limitaciones del Empaquetado Cuantizado

Al documentar o implementar este módulo, el arquitecto debe tener presente las siguientes restricciones impuestas por la física del hardware embebido:

1. **Cuello de Botella de la FPU por Software:** Aunque los pesos ocupan 1/4 del espacio físico, la inferencia en C++ (como se ve en el código superior) reconstruye los valores a `float` para garantizar estabilidad ante desbordamientos (*overflows*). En placas modernas (ESP32, Cortex-M4), esto no penaliza la latencia gracias a sus Unidades de Coma Flotante de hardware. Sin embargo, en un AVR clásico (ATmega328P de 8-bits), el cast `(float)peso_q * escala_actual` obliga al chip a emular matemáticas flotantes por software. **Resultado:** El modelo cabrá perfectamente en la memoria, pero su latencia de ejecución será ligeramente mayor que si se usara aritmética de punto fijo puro (CMSIS-NN).
2. **Soporte de Topologías Complejas:** El exportador en C++ soporta la cuantificación híbrida para capas estandarizadas (`Linear`, `Conv1D`, `Conv2D`). Sin embargo, en capas donde se aplican optimizaciones espaciales como `Operator Fusion` (`SeparableConv2D`), el manejo de punteros anidados y la de-cuantificación paralela incrementan la complejidad de la generación de código. El orquestador (`ml_manager`) forzará la cuantificación *Per-Tensor* si detecta que la estructura de fusión corre el riesgo de desalinear las memorias de caché del microprocesador destino. Atento a ese detalle.



### 4. Gestión Estricta y Seguridad de Memoria (`PROGMEM` & SRAM Estática)

El mayor enemigo de la Inteligencia Artificial en el "Bare Metal" no es la velocidad del procesador, sino la memoria. Los microcontroladores (especialmente los de 8-bits como la familia AVR) carecen de una Unidad de Gestión de Memoria (MMU) y de un Recolector de Basura (*Garbage Collector*). Un solo error en la asignación de arreglos resulta en un desbordamiento de pila (*Stack Overflow*) o en la fragmentación del montón (*Heap Fragmentation*), causando que el sistema se congele de forma silenciosa.

Para garantizar la estabilidad industrial en sistemas de misión crítica, el exportador a C++ de **MiniML Engine** implementa una arquitectura de **"Zero-Dynamic Allocation"** (Cero Asignación Dinámica), apoyándose en directivas de hardware como `PROGMEM` y macros de protección estática para el motor MiniTensor.

A continuación, se detalla cómo el framework domina la física de la memoria y cómo el ingeniero de firmware debe interactuar con estas barreras.


#### A. La Fortaleza de la Memoria No Volátil: `PROGMEM`

`PROGMEM` (Program Memory) es una directiva del compilador GCC (especialmente en `avr/pgmspace.h`) que le ordena al microcontrolador almacenar una variable exclusivamente en la memoria Flash (ROM), prohibiendo que se cargue en la RAM dinámica (SRAM) al arrancar el dispositivo.

* **El Problema Clásico:** En C++ estándar, si defines `const float pesos[1000] = {...};`, el compilador guardará esos datos en la Flash, pero al iniciar el programa, **los copiará íntegramente a la SRAM** para un acceso más rápido. Si tienes 2KB de SRAM, tu placa colapsará antes de ejecutar la primera línea del `setup()`.
* **La Solución MiniML:** El exportador etiqueta cada matriz paramétrica del modelo (Pesos, Sesgos, Factores de Escala INT8, Umbrales de Árboles) con `PROGMEM`.

**Mecánica de Lectura Segura:**
Como los datos en `PROGMEM` no están en el espacio de direcciones de la RAM, no puedes leerlos usando punteros estándar como `pesos[i]`. El código generado por MiniML implementa macros de lectura segura para extraer byte por byte o flotante por flotante en exactamente un ciclo de reloj por instrucción.

```cpp
// 1. Declaración protegida en el Header (.h) exportado
extern const int8_t layer1_weights[4096] PROGMEM;

// 2. Lectura Segura en el Bucle de Inferencia (.cpp)
// En lugar de hacer: int8_t w = layer1_weights[i]; (Lo cual leería basura)
int8_t w = pgm_read_byte_near(&layer1_weights[i]);
float b  = pgm_read_float_near(&layer1_biases[n]);

```

*Nota de Arquitectura:* En procesadores ARM modernos de 32-bits (como ESP32 o STM32), la memoria Flash está mapeada directamente en el espacio de direcciones de memoria de datos. En estos chips, la macro `PROGMEM` se define vacía por compatibilidad cruzada, y el compilador GCC-ARM maneja el enlazado estático de forma transparente.


#### B. El Modelo de SRAM Estática (Protección MiniTensor)

Mientras que los pesos viven en la ROM, las *activaciones* (los resultados matemáticos que fluyen de una capa a otra, como la salida de una convolución) deben vivir forzosamente en la RAM (SRAM) porque cambian con cada nueva lectura del sensor.

El framework tiene estrictamente prohibido el uso de `new`, `malloc()`, `free()`, o la clase `std::vector` de C++.

* **La Filosofía *Arena Allocation*:** En lugar de crear y destruir arreglos dinámicos al vuelo, el transpilador calcula matemáticamente el **tamaño máximo** del tensor intermedio que existirá durante el *Forward Pass* en tiempo de exportación.
* **Implementación C++:** MiniML exporta arreglos globales estáticos o locales definidos en tiempo de compilación. Las funciones `predict()` se pasan punteros a estos buffers pre-asignados (*In-Place Computing*).

```cpp
// Buffer de trabajo estático pre-asignado.
// Nunca crecerá ni se encogerá, evitando el Heap Fragmentation.
float tensor_buffer_A[128]; 
float tensor_buffer_B[64];

void predict_minitensor(const float* input, float* output) {
    // La capa 1 lee del input físico y escribe en el buffer A
    forward_conv1d(input, tensor_buffer_A);
    
    // La capa 2 lee del buffer A y escribe en el buffer B
    forward_maxpool1d(tensor_buffer_A, tensor_buffer_B);
    
    // La capa 3 (Linear) lee del buffer B y escribe directamente en el output
    forward_linear(tensor_buffer_B, output);
}

```


#### C. Guía y Mejores Prácticas de Seguridad en Memoria

Para aprovechar al máximo esta arquitectura sin romper el ecosistema del hardware, es vital comprender que la seguridad de memoria se divide en dos fases: el **Diseño de la Arquitectura** (en el Host con Python) y la **Implementación del Firmware** (en el Edge con C++).

El framework opera bajo un principio: *"Entrena sin límites en la PC, pero diseña para sobrevivir en el silicio"*.

##### Fase 1: El Entorno Python (Diseño Consciente del Hardware)

Aunque Python maneja la memoria dinámicamente y el Recolector de Basura (*Garbage Collector*) evita colapsos durante el entrenamiento, **el programador que escribe el script en Python es el arquitecto del hardware final**. Si se diseña irresponsablemente aquí, el código C++ generado será matemáticamente perfecto, pero físicamente imposible de flashear.

1. **Responsabilidad Topológica:** En Python no te preocupas por "cómo" se asigna la memoria, sino por "cuánta" se va a exportar. Definir un `nn.Linear(in_features=1024, out_features=512)` en Python correrá en segundos en la PC, pero generará una matriz de más de $524,000$ parámetros. Al exportar, ese modelo exigirá al menos 524 KB de ROM, bloqueando la compilación en la mayoría de microcontroladores de gama baja.
* *Práctica Recomendada:* Mantén núcleos de convolución pequeños (`kernel_size=3`), prioriza `SeparableConv2D`, y aplica capas de submuestreo (`MaxPool`) de forma agresiva antes de aplanar (`Flatten`) la red hacia las capas lineales.


2. **Invocación Obligatoria del Cuantizador:** Llamar al método `modelo.quantize()` en tu script de Python no ahorra memoria en tu PC (de hecho, requiere cálculos adicionales para las escalas), pero es el seguro de vida del hardware. Es la orden explícita para que el transpilador aplique la compresión `int8_t` y la protección `PROGMEM` en el código destino.
3. **Restricción de Lotes (Batch Size) Visualizados:** Puedes usar `batch_size = 64` durante el entrenamiento en PC para que el Descenso de Gradiente converja rápido. Sin embargo, al probar tu modelo y diseñar la lógica de aplicación, asume siempre `batch_size = 1`. El microcontrolador no procesará lotes masivos en tiempo real, evaluará ventana por ventana.

##### Fase 2: El Entorno C++ (Implementación del Firmware)

Una vez que MiniML exporta la librería estática, el integrador de C++ debe adherirse a estas directrices para no vulnerar la arquitectura de "Cero Asignación Dinámica" (Zero-Dynamic Allocation) generada:

1. **Nunca Pases Grandes Buffers por Valor:** El código principal (`loop()`) de Arduino o FreeRTOS es responsabilidad del integrador. Cuando llames a la función `predict()`, **siempre debes pasar los arreglos del sensor por referencia (punteros)**.
* ❌ **Fatal:** `modelo.predict(lecturas_sensor);` (El compilador intentará hacer una copia profunda del arreglo en la memoria Pila/Stack, causando un desbordamiento inmediato).
* ✅ **Correcto:** `modelo.predict(&lecturas_sensor[0], &salida_prediccion[0]);`


2. **Control del Alcance (Scope) de las Entradas:** No declares arreglos de sensores masivos dentro del bucle `loop()` o interrupciones. Las variables locales viven en el Stack de la memoria SRAM. Si la red ingiere una matriz térmica, declararla localmente consumirá kilobytes críticos.
* *Solución:* Declara el arreglo de entrada del sensor como `static float img_buffer[576];` o ponlo a nivel global antes de la función `setup()` para proteger el montón (Heap).


3. **Profiling en Tiempo de Compilación (Auditoría `sizeof`):** Aprovecha que MiniML genera buffers estáticos globales (`tensor_buffer_A`, etc.). Puedes insertar advertencias estáticas en C++ para que el compilador aborte si el consumo supera tu hardware:
```cpp
#if (sizeof(tensor_buffer_A) + sizeof(tensor_buffer_B) > 1024)
    #error "ALERTA: Los tensores intermedios superan 1KB de SRAM. Riesgo inminente de inestabilidad física."
#endif

```

4. **Precauciones con el Recorte de Tipos (Type-Casting):** El C++ generado mezcla matrices cuantizadas (`int8_t` en ROM) con buffers temporales flotantes en SRAM. Si editas manualmente la librería generada, nunca alteres la firma de los tipos de las macros `pgm_read_*`. Leer un `float` de Flash usando `pgm_read_byte` truncará los punteros y arrojará predicciones corrompidas.



### 5. Estructura del Código C++ Generado (Arquitectura de la Librería)

A diferencia de los scripts para principiantes que exportan un único archivo monolítico (`.ino`), **MiniML Engine** está diseñado para integrarse en cadenas de suministro de software (CI/CD) y proyectos de ingeniería complejos.

El módulo `LibraryPackager` no genera simples fragmentos de código; compila un archivo `.zip` que contiene una **librería C++ estándar y modular**, lista para ser importada nativamente en PlatformIO o Arduino IDE.

#### A. Árbol de Directorios del Paquete Generado

Cuando exportas un modelo (ej. llamado `DetectorFallas`), el `.zip` extraído presenta la siguiente arquitectura estricta:

```text
DetectorFallas/
├── include/
│   └── DetectorFallas.h       # Declaraciones, Firmas y directivas externas
├── src/
│   └── DetectorFallas.cpp     # Lógica matemática y matrices PROGMEM reales
├── library.json               # Manifiesto para MLOps (PlatformIO)
├── library.properties         # Manifiesto Legacy (Arduino IDE)
└── keywords.txt               # Resaltado de sintaxis para el IDE

```

#### B. Separación de Declaración e Implementación (Ejemplos)

Esta separación `include/` vs `src/` evita el temido error de compilación por *Múltiples Definiciones* (Multiple Definition Error) que ocurre cuando un modelo "Header-Only" se incluye en varios archivos `.cpp` de un mismo proyecto físico.

**1. El Archivo de Cabecera (`include/DetectorFallas.h`):**
Contiene las guardas de inclusión, define las estructuras de datos pre-asignadas para la SRAM y declara las funciones de inferencia públicas.

```cpp
#ifndef DETECTORFALLAS_H
#define DETECTORFALLAS_H

#include <stdint.h>
#include <avr/pgmspace.h> // O su equivalente en ARM

// --- Buffers Estáticos de SRAM ---
// Estos buffers deben ser usados por el programa principal
extern float tensor_buffer_in[128];
extern float tensor_buffer_out[1];

// --- Declaración externa de Pesos (PROGMEM) ---
extern const int8_t layer1_weights[512] PROGMEM;
extern const float layer1_scales[4] PROGMEM;

// --- API Pública del Modelo ---
void DetectorFallas_predict(const float* input, float* output);

#endif // DETECTORFALLAS_H

```

**2. El Archivo de Implementación (`src/DetectorFallas.cpp`):**
Aquí es donde reside el peso físico del modelo y la lógica matemática transpilada. Las matrices de pesos se definen aquí de forma estática, encapsulando la memoria Flash.

```cpp
#include "DetectorFallas.h"
#include <math.h>

// Definición física de los buffers en RAM
float tensor_buffer_in[128];
float tensor_buffer_out[1];

// Inyección de los pesos cuantizados directamente en la ROM
const int8_t layer1_weights[512] PROGMEM = {
    12, -45, 88, 126, -101, 0, 3, /* ... 505 bytes más ... */
};
const float layer1_scales[4] PROGMEM = {
    0.012f, 0.005f, 0.033f, 0.019f
};

// Implementación de la Inferencia (Ejemplo de capa Linear Cuantizada)
void DetectorFallas_predict(const float* input, float* output) {
    int w_idx = 0;
    for (int out_c = 0; out_c < 4; out_c++) {
        float suma = 0.0f;
        float scale = pgm_read_float_near(&layer1_scales[out_c]);
        
        for (int in_c = 0; in_c < 128; in_c++) {
            int8_t weight = pgm_read_byte_near(&layer1_weights[w_idx]);
            suma += input[in_c] * ((float)weight * scale);
            w_idx++;
        }
        // Activación ReLU inline
        output[out_c] = (suma > 0.0f) ? suma : 0.0f;
    }
}

```


### 6. Limitaciones Técnicas del Empaquetado y Exportación

Por más robusto que sea el transpilador, el paso de un entorno de alto nivel a código máquina estático presenta fricciones. Todo arquitecto que utilice **MiniML Engine** debe auditar sus proyectos considerando las siguientes limitantes de la versión actual.

#### ⚠️ Estado del Exportador a Rust (Experimental / Inacabado)

El framework posee un módulo de exportación dirigido a **Rust** (`miniml_rust`) destinado a sistemas embebidos seguros y WebAssembly (`no_std`). Sin embargo, **este exportador se encuentra actualmente en estado inacabado y experimental.**

* **Falta de Soporte Deep Learning:** El generador de código Rust **no soporta** las topologías avanzadas de MiniTensor, específicamente `SeparableConv2D` y `ResidualBlock1D`.
* **Módulos Desactualizados:** La implementación de la capa `MaxPool2D` en Rust está depreciada y no se alinea con la geometría dinámica del motor Autograd actual.
* **Directiva de Producción:** Para entornos profesionales, despliegues industriales o proyectos académicos críticos, **es estrictamente obligatorio utilizar el exportador de C++**. El C++ es el estándar de oro actual del framework, contando con soporte total para cuantización *Per-Channel*, Operator Fusion y gestión agresiva de la Flash (`PROGMEM`).

#### Limitación de Punteros de 16-bits (AVR)

En microcontroladores de 8-bits clásicos (como el ATmega2560), los punteros estándar en C++ son de 16 bits, lo que significa que solo pueden direccionar hasta $65,535$ bytes (64 KB) de memoria continua.

* Si exportas un modelo cuantizado cuyas matrices en Flash (juntas en un solo arreglo) superan los 32 KB, el compilador GCC-AVR podría generar un desbordamiento silencioso (*Pointer Truncation*).
* *Mitigación:* Para modelos que se acerquen a estos límites, el hardware objetivo debe ser forzosamente de 32-bits (ej. ESP32, STM32, RP2040), los cuales utilizan punteros de 32 bits y direccionan Megabytes sin esfuerzo.

#### Ausencia de Vectorización Explícita (SIMD / DSP)

El código C++ generado por MiniML es altamente portable porque utiliza bucles `for` matemáticos estándar. Sin embargo, no inyecta intrínsecos de ensamblador específicos de hardware (como las instrucciones DSP o funciones intrínsecas de ARM Cortex-M4/M7).

* Esto significa que el rendimiento real depende críticamente de la capacidad del compilador (`-O3`) para auto-vectorizar los bucles (*Loop Unrolling* y *SIMD*). Si se requiere aprovechar al máximo las instrucciones DSP, el modelo exportado puede quedar ligeramente por detrás de una implementación manual escrita puramente sobre CMSIS-NN, aunque el ahorro en tiempo de desarrollo compense esta mínima diferencia de latencia. 

#### Consideraciones para el desarrollo futuro

En futuras actualizaciones, se considerará implementar la Vectorización Explícita para el módulo de empaquetado y exportación. Además de darle soporte al rust_writter para que sea compatible con todos los módulos disponibles de Deep Learning actualmente.

---

# Capítulo 7. CLI de MiniML (Interfaz de Línea de Comandos)

El ecosistema de **MiniML Engine** no está confinado a los scripts de entrenamiento en Python. Para facilitar el ciclo de vida del desarrollo industrial (MLOps), el framework expone una propia Interfaz de Línea de Comandos (CLI) a través de su punto de entrada principal (`main.py`).

Esta herramienta es el puente interactivo para que los ingenieros puedan auditar arquitecturas, perfilar el consumo de memoria en el silicio, recolectar datos físicos y simular inferencias en tiempo real mediante la misma terminal de su IDE sin escribir código adicional.

A continuación, se desglosa el funcionamiento técnico, los parámetros y los casos de uso de cada comando disponible en el CLI de MiniML.



## Estructura General y Ejecución

El CLI se invoca desde la terminal apuntando al archivo principal del framework. La estructura general del comando es:


```bash
python main.py <comando> [argumentos]

```

O bien, si tiene el paquete de PyPI instalado en su IDE, basta con hacer un


```bash
miniml --help

```

Para corroborar que el CLI funciona y responde perfectamente desde la terminal de su entorno de desarrollo. Si tiene el paquete instalado, también basta con solo hacer: 


```bash
miniml <comando> --arg 

```

para usar las características que se explican más abajo


Los cuatro comandos principales soportados son: `inspect`, `estimate`, `sensor` y `simulate`.



## 1. Comando: `inspect` (Auditoría de Arquitectura)

* **¿Para qué sirve?**
Lee un modelo matemático guardado en disco y renderiza un resumen estructural en la terminal. Es fundamental para verificar que la topología del grafo (capas, entradas, salidas) se ha guardado correctamente antes de intentar exportarlo a C++.
* **Argumentos Obligatorios:**
* `--model`: Ruta absoluta o relativa al archivo JSON del modelo generado por MiniML.


* **Caso de Uso:**
Acabas de recibir un modelo entrenado por otro ingeniero del equipo y necesitas saber qué arquitectura interna tiene antes de desplegarlo.
* **Ejemplo de Uso:**
```bash
python main.py inspect --model modelos/detector_fallas.json

```



## 2. Comando: `estimate` (Perfilado de Memoria Edge AI)

* **¿Para qué sirve?**
Es la herramienta de diagnóstico más crítica del framework. Realiza un análisis estático de los pesos y tensores intermedios del modelo para calcular **exactamente** cuántos bytes de memoria RAM dinámica (SRAM) y almacenamiento Flash (ROM) consumirá al ser compilado en el microcontrolador objetivo.
* **Argumentos Disponibles:**
* `--model` *(Obligatorio)*: Ruta al archivo JSON del modelo.
* `--flash` *(Opcional)*: Límite físico de la memoria Flash del chip en bytes. Por defecto es **32256** (Arduino Uno / ATmega328P).
* `--sram` *(Opcional)*: Límite físico de la memoria SRAM en bytes. Por defecto es **2048** (Arduino Uno).
* `--lang` *(Opcional)*: Lenguaje de transpilación. Opciones: `C`, `C++`, `Rust`. Por defecto: `C++`.
* `--quantized` *(Flag Opcional)*: Si se incluye, el estimador calculará la huella de memoria asumiendo compresión de pesos a INT8.
* `--input_shape` *(Opcional)*: Para redes convolucionales, define la forma del tensor de entrada separada por comas (ej. `1,28,28`).


* **Funcionamiento Técnico:**
El estimador arroja un reporte porcentual detallado. Si el consumo de SRAM o Flash supera el 90% de la capacidad del hardware definido, el CLI detonará una advertencia `[⚠️ ADVERTENCIA]` alertando al arquitecto que el despliegue físico corre riesgo inminente de inestabilidad o *Stack Overflow*.
* **Ejemplo de Uso:**
Perfilando un modelo cuantizado para un ESP32 (asumiendo 4MB de Flash y 320KB de RAM):
```bash
python main.py estimate --model modelos/vision_edge.json --flash 4194304 --sram 327680 --quantized --input_shape 1,24,24

```



## 3. Comando: `sensor` (Recolección de Datos / Data Harvesting)

* **¿Para qué sirve?**
Abre un puente de comunicación serial entre el PC y el microcontrolador físico (o un entorno simulado). Se utiliza para la ingesta en vivo de lecturas de sensores y la construcción de Datasets crudos directamente en formato CSV.
* **Argumentos Disponibles:**
* `--port` *(Opcional)*: El puerto físico del hardware (ej. `COM3` en Windows o `/dev/ttyUSB0` en Linux). Por defecto, arranca en modo `"SIMULADOR"`.
* `--baudrate` *(Opcional)*: Velocidad de transmisión en baudios. Debe coincidir con el `Serial.begin()` del hardware. Por defecto es **9600**.
* `--label` *(Opcional)*: La etiqueta (*target/class*) que se asignará automáticamente a todos los datos capturados en esta sesión. Por defecto: `clase_0`.
* `--log` *(Opcional)*: Ruta del archivo CSV donde se escribirán y guardarán los datos de forma persistente.
* `--verbose` *(Flag Opcional)*: Si se activa, imprime el flujo de datos crudos en la terminal en tiempo real.


* **Caso de Uso:**
Crear un dataset para detectar cuando un motor está vibrando anómalamente. Conectas el ESP32 con un acelerómetro al PC, lo ajustas al motor defectuoso, y ejecutas el comando asignando la etiqueta `motor_roto`.
* **Ejemplo de Uso:**
```bash
python main.py sensor --port COM4 --baudrate 115200 --label motor_roto --log datasets/vibracion.csv --verbose

```



## 4. Comando: `simulate` (REPL de Inferencia en Vivo)

* **¿Para qué sirve?**
Lanza un entorno interactivo (*Read-Eval-Print Loop*) alojado en la PC que emula el comportamiento matemático del microcontrolador. Permite inyectar datos manualmente al modelo entrenado o evaluar lotes masivos desde un archivo CSV para observar latencias simuladas y respuestas de la red neuronal, soportando tanto modelos clásicos (Legacy) como Deep Learning (MiniTensor).
* **Argumentos Obligatorios:**
* `--model`: Ruta al archivo JSON del modelo a simular.


* **Modos de Operación Interna:**
Al iniciar el simulador (`miniml-sim>`), el usuario tiene tres opciones:
1. **Modo Manual:** Escribir una matriz plana de flotantes separada por comas (ej. `25.3, 1024, 0.5`). El CLI empaquetará la entrada, calculará el tiempo en milisegundos que le toma atravesar el grafo y devolverá la salida de activación de la capa final.
2. **Modo Lotes (Dataset CSV):** Escribir la ruta a un archivo `.csv` local. El simulador iterará fila por fila, filtrará automáticamente cualquier cabecera no numérica, y realizará predicciones en ráfaga inyectando una pequeña pausa (`0.05s`) para emular el flujo de un monitor serial físico real.
3. **Salida:** Comandos de escape `salir`, `exit` o `quit`.


* **Ejemplo de Uso (Ejecución):**
```bash
python main.py simulate --model modelos/clasificador_gestos.json

```


*Interacción en la terminal:*
```text
miniml-sim> 1.2, 3.4, -0.5
  [Hardware Sim] Procesado en 1.45 ms
  [Red Neuronal] Reacción/Salida -> [0.8912]

miniml-sim> test_dataset.csv
  [SIMULADOR] Procesando archivo: test_dataset.csv
  Fila 1 -> Input: [2.1, 4.0, -1.2]... | Output: [0.952]

```

---

*"Cuando comencé a escribir las primeras líneas de código de este framework hace 10 meses, la meta parecía casi irracional: construir un ecosistema de Machine Learning desde cero, sin depender de los gigantes de la industria, y forzar que esas matemáticas complejas encajaran en microcontroladores con menos memoria RAM que un simple archivo de texto.*

*Desarrollar **MiniML Engine** localmente, depurando tensores en la madrugada y peleando contra la fragmentación de la memoria SRAM y los límites estrictos de `PROGMEM`, ha sido un ciclo de ingeniería exhaustivo y solitario por momentos. Pero cada error de segmentación y cada kernel panic en las placas físicas valió la pena.*

*Este framework no nació en un laboratorio corporativo con servidores ilimitados. Nació de la necesidad absoluta de democratizar la Inteligencia Artificial en Lationamérica, con recursos de bajo costo. Nació para demostrar que no necesitas estar conectado a la nube o tener un presupuesto alto para hacer hardware inteligente. Nació para que herramientas educativas, prototipos comunitarios y sistemas de bajo costo puedan tomar decisiones matemáticas complejas en el borde, con total privacidad y eficiencia energética.*

*La Inteligencia Artificial embebida ya no es un lujo reservado para procesadores costosos. Si tienes un microcontrolador de dos dólares y la voluntad de optimizar tu código, tienes en tus manos el poder de clasificar la realidad física.*

*El código fuente ahora está allá afuera. Rompan la librería, busquen sus límites, forkeen el repositorio y construyan hardware que importe, en los escritorios de los entusiastas y la comunidad, en las aulas de clases y en los prototipos ensamblados placa por placa.*

*Gracias por confiar en esta arquitectura. Nos vemos en el código."*

* Michego Takoro