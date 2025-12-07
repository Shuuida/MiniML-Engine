# **Quantization in MiniML: Complete Guide**

## **Table of Contents**

1. [Introduction](https://www.google.com/search?q=%23introduction)  
2. [Quantization Architecture](https://www.google.com/search?q=%23quantization-architecture)  
3. [Quantization Methods](https://www.google.com/search?q=%23quantization-methods)  
4. [Quantization Process](https://www.google.com/search?q=%23quantization-process)  
5. [Exporting to C Firmware](https://www.google.com/search?q=%23exporting-to-c-firmware)  
6. [Comparative Table](https://www.google.com/search?q=%23comparative-table)  
7. [Current Limitations](https://www.google.com/search?q=%23current-limitations)  
8. [Recommendations for Embedded Projects](https://www.google.com/search?q=%23recommendations-for-embedded-projects)  
9. [Usage Examples](https://www.google.com/search?q=%23usage-examples)

## **Introduction**

MiniML implements **Post-Training Quantization (PTQ)** for neural networks, enabling model size reduction and accelerated inference on low-cost microcontrollers by utilizing 8-bit integer arithmetic (int8) instead of 32-bit floating point.

### **Key Benefits**

* **Memory Reduction**: \~75% less space usage (int8 vs float32).  
* **Acceleration**: Faster integer operations on MCUs lacking an FPU.  
* **CMSIS-NN Compatibility**: Integration with optimized ARM kernels.  
* **Preserved Precision**: Typical accuracy loss is \< 2%.

## **Quantization Architecture**

### **Core Components**

#### **1\. MiniNeuralNetwork (ml\_runtime.py)**

The base class that implements:

* Activation calibration (calibrate()).  
* Weight and bias quantization (quantize()).  
* Native export to C (to\_arduino\_code()).

#### **2\. CMSISAdapter (adapters/cmsis\_nn/adapter.py)**

An advanced adapter that generates code compatible with:

* **CMSIS-NN**: Optimized ARM kernels for Cortex-M.  
* **Portable Fallback**: Standard C implementation with no dependencies.

### **Quantization Flow**

graph TD;  
    A\[Training (Float32)\] \--\> B\[Calibration (calibrate())\];  
    B \--\> C\[Quantization (quantize())\];  
    C \--\> D\[Export (export\_to\_c())\];  
    D \--\> E\[C Firmware (int8 inference)\];

1. **Calibration**: Calculates act\_scales for input, hidden, and output layers.  
2. **Quantization**: Converts weights to int8 and biases to int32.  
3. **Export**: Generates optimized C code.

## **Quantization Methods**

### **1\. Per-Layer Symmetric Quantization**

**Implementation**: Default method in MiniNeuralNetwork.quantize().

#### **Characteristics:**

* **Weights**: Quantized to int8 with range \[-127, 127\].  
* **Zero-point**: Implicitly 0 (symmetric quantization).  
* **Scale**: One scale per weight matrix (per-layer).  
* **Biases**: Quantized to int32 to preserve precision.

#### **Formulas:**

**For Weights (W):**

abs\_max \= max(abs(min(W)), abs(max(W)))  
scale\_w \= abs\_max / 127.0  
q\_w \= round(w / scale\_w)  \# Clipped to \[-127, 127\]

**For Biases (B):**

effective\_scale \= input\_scale \* scale\_w  
b\_int32 \= round(b / effective\_scale)  \# Clipped to int32 range

**Requantization Multiplier:**

requant\_mult \= effective\_scale / output\_scale

#### **Pros:**

* ✅ Simple and fast implementation.  
* ✅ Low computational overhead.  
* ✅ CMSIS-NN compatible.  
* ✅ Good accuracy for small-to-medium networks.

#### **Cons:**

* ❌ Lower accuracy than per-channel quantization for large networks.  
* ❌ Sensitive to outliers in weight distribution.

### **2\. CMSIS-NN Quantization (Fixed-Point)**

**Implementation**: CMSISAdapter.generate\_c().

#### **Characteristics:**

* **Weights**: int8\_t stored in aligned arrays.  
* **Biases**: Pre-quantized int32\_t.  
* **Multipliers**: Converted to Q31 format (significand \+ shift).  
* **Activations**: int8\_t throughout the entire inference.

#### **Q31 Format for Multipliers:**

def \_quantize\_multiplier(real\_multiplier: float) \-\> Tuple\[int, int\]:  
    significand, shift \= math.frexp(real\_multiplier)  
    q\_mult \= int(round(significand \* (1 \<\< 31)))  
    \# Adjustment to avoid overflow  
    if q\_mult \== (1 \<\< 31):  
        q\_mult //= 2  
        shift \+= 1  
    return q\_mult, shift

#### **Pros:**

* ✅ Maximum speed on Cortex-M (SIMD kernels).  
* ✅ Full int8 inference (no floats).  
* ✅ Low power consumption.  
* ✅ Standard ARM CMSIS-NN compatibility.

#### **Cons:**

* ❌ Requires CMSIS-NN library (not portable).  
* ❌ Limited to ARM Cortex-M architectures.  
* ❌ Higher implementation complexity.

### **3\. AVR Hybrid Mode (Arduino 8-bit)**

**Implementation**: MiniNeuralNetwork.to\_arduino\_code().

#### **Characteristics:**

* **Weights**: int8\_t stored in PROGMEM (Flash).  
* **Scales**: float stored in PROGMEM.  
* **Computation**: Hybrid (int8 storage, float compute).  
* **Biases**: Original float stored in PROGMEM.

#### **Strategy:**

// Read quantized weight from Flash  
int8\_t w \= pgm\_read\_byte(\&W1\[i\]\[j\]);  
// On-the-fly dequantization  
float dequantized \= (float)w \* scale \* input;

#### **Pros:**

* ✅ Saves SRAM (weights in Flash, not RAM).  
* ✅ Ideal for 8-bit AVR (Arduino Uno/Nano).  
* ✅ No FPU required (uses float only for scales/accumulators).  
* ✅ Portable (no external dependencies).

#### **Cons:**

* ❌ Slower than pure int8 (due to float conversions).  
* ❌ Requires FPU or float emulation.  
* ❌ Higher Flash usage than pure int8 (due to float biases).

## **Quantization Process**

### **Step 1: Calibration (calibrate())**

**Purpose**: Determine activation ranges for each layer.

def calibrate(self, dataset: List\[List\[float\]\]):  
    """  
    Calculates activation ranges (min/max) for Input, Hidden, and Output.  
    Essential for int8 Post-Training Quantization.  
    """  
    \# Iterates over the dataset and finds:  
    \# \- max\_in: absolute max of inputs  
    \# \- max\_hidden: absolute max of hidden activations  
    \# \- max\_out: absolute max of outputs  
      
    self.act\_scales \= {  
        'input': max\_in / 127.0,  
        'hidden': max\_hidden / 127.0,  
        'output': max\_out / 127.0  
    }

**Important Notes**:

* Automatically executed after fit().  
* Requires a calibration dataset (typically the training set).  
* Scales are saved in act\_scales for later use.

### **Step 2: Quantization (quantize())**

**Purpose**: Convert weights and biases from float32 to int8/int32.

def quantize(self, per\_channel: bool \= True):  
    """  
    Quantizes weights to int8 and biases to int32.  
    Requires previously calculated act\_scales.  
    """  
    \# For each layer:  
    \# 1\. Calculate scale per weight row  
    \# 2\. Quantize weights to int8  
    \# 3\. Quantize biases to int32  
    \# 4\. Calculate requantization multipliers  
      
    self.q\_W1, self.i32\_B1, self.requant\_mult1, self.s\_W1\_list \= ...  
    self.q\_W2, self.i32\_B2, self.requant\_mult2, self.s\_W2\_list \= ...  
    self.quantized \= True

**Generated Attributes**:

* q\_W1, q\_W2: Quantized weight matrices (int8).  
* i32\_B1, i32\_B2: Quantized bias vectors (int32).  
* requant\_mult1, requant\_mult2: Requantization multipliers (float).  
* s\_W1\_list, s\_W2\_list: Scales per weight row (float).

### **Step 3: Export (export\_to\_c())**

**Purpose**: Generate optimized C code for firmware.

**Automatic Detection**:

* If model has q\_W1 → Uses CMSISAdapter (Preferred).  
* If that fails → Uses to\_arduino\_code() (Native fallback).  
* If scaler exists → Includes preprocessing code.

## **Exporting to C Firmware**

### **Option 1: CMSIS-NN (Recommended for ARM Cortex-M)**

**Generated by**: CMSISAdapter.generate\_c().

**Features**:

* Code optimized for arm\_fully\_connected\_s8().  
* Full int8 inference.  
* Portable fallback if CMSIS-NN is not enabled.

**Code Structure**:

// Data arrays (aligned for SIMD)  
const int8\_t W1\[N\] ALIGNED(4) \= {...};  
const int32\_t B1\[M\] ALIGNED(4) \= {...};  
const int32\_t MULT1\[M\] ALIGNED(4) \= {...};  
const int32\_t SHIFT1\[M\] ALIGNED(4) \= {...};

\#ifdef CMSISNN\_ENABLED  
    // Uses optimized ARM kernels  
    arm\_fully\_connected\_s8(...);  
\#else  
    // Standard C portable fallback  
    // Manual loop implementation  
\#endif

**Pros**:

* Maximum speed on Cortex-M4/M7.  
* Low energy consumption.  
* Pure int8 inference (no floats).

### **Option 2: AVR Hybrid Mode (Arduino 8-bit)**

**Generated by**: MiniNeuralNetwork.to\_arduino\_code().

**Features**:

* Weights in PROGMEM (Flash).  
* Hybrid calculation (int8 storage, float compute).  
* Ideal for AVR without FPU.

**Code Structure**:

\#include \<avr/pgmspace.h\>

// Quantized weights in Flash  
const int8\_t W1\[N\]\[M\] PROGMEM \= {...};  
// Scales in Flash  
const float sW1\[N\] PROGMEM \= {...};  
// Original biases in Flash  
const float B1\[N\] PROGMEM \= {...};

void predict(float \*row, float \*out) {  
    // On-the-fly dequantization  
    float w \= (float)pgm\_read\_byte(\&W1\[i\]\[j\]) \* pgm\_read\_float(\&sW1\[i\]);  
    // Computation in float  
    sum \+= w \* input\[j\];  
}

**Pros**:

* Saves SRAM (data in Flash).  
* Portable (no dependencies).  
* Compatible with Arduino Uno/Nano.

## **Comparative Table**

| Feature | Per-Layer Symmetric | CMSIS-NN Fixed-Point | AVR Hybrid Mode |
| :---- | :---- | :---- | :---- |
| **Weight Precision** | int8 | int8 | int8 |
| **Bias Precision** | int32 | int32 | float32 |
| **Activation Precision** | float32 (ref) | int8 | float32 |
| **Scale** | Per layer | Per layer | Per layer |
| **Zero-Point** | 0 (symmetric) | 0 (symmetric) | 0 (symmetric) |
| **Weight Storage** | RAM/Flash | RAM (aligned) | PROGMEM (Flash) |
| **Inference Speed** | Medium | Very High | Low-Medium |
| **Power Consumption** | Medium | Low | Medium |
| **Memory Required** | \~75% less | \~75% less | \~75% less (weights) |
| **Compatibility** | Universal | ARM Cortex-M | AVR 8-bit |
| **Dependencies** | None | CMSIS-NN | None |
| **FPU Required** | Optional | No | Yes (emulation OK) |
| **SIMD Optimized** | No | Yes | No |
| **Recommended for** | General | Cortex-M4/M7 | Arduino Uno/Nano |
| **Code Overhead** | Low | Medium | Low |
| **Complexity** | Low | Medium | Low |

### **Typical Performance Metrics**

| Metric | Original Float32 | Per-Layer (int8) | CMSIS-NN | AVR Hybrid |
| :---- | :---- | :---- | :---- | :---- |
| **Model Size** | 100% | \~25% | \~25% | \~25% (weights) |
| **Inference Speed** | 1x | 2-3x | 5-10x | 1.5-2x |
| **Accuracy Loss** | 0% | 0.5-2% | 0.5-2% | 0.5-2% |
| **RAM Usage** | High | Medium | Medium | Low |
| **Flash Usage** | Low | Medium | Medium | High |

## **Current Limitations**

### **Technical Limitations**

1. **Neural Networks Only**  
   * ✅ MiniNeuralNetwork supports full quantization.  
   * ❌ Other models (DecisionTree, RandomForest, etc.) do not support quantization yet.  
2. **Post-Training Quantization**  
   * ❌ No Quantization-Aware Training (QAT).  
   * ❌ Quantization happens *after* training.  
   * ⚠️ May experience accuracy loss in sensitive models.  
3. **Per-Layer (Not Per-Channel)**  
   * ❌ One scale per layer, not per channel.  
   * ⚠️ Lower accuracy than per-channel for large networks.  
   * ✅ Sufficient for small-to-medium networks.  
4. **Float Activations (Native Mode)**  
   * ⚠️ In to\_arduino\_code(), activations are calculated in float.  
   * ✅ Only CMSIS-NN uses pure int8 activations.  
   * ⚠️ Requires FPU or emulation for hybrid mode.  
5. **Limited to MLP (2 Layers)**  
   * ❌ Only supports architectures with up to 2 hidden layers.  
   * ❌ Does not support deep networks (3+ layers).  
   * ✅ Sufficient for most embedded use cases.  
6. **Calibration Requires Dataset**  
   * ⚠️ calibrate() needs a complete dataset.  
   * ⚠️ No calibration with reduced datasets.  
   * ✅ Runs automatically after fit().

### **Hardware Limitations**

1. **CMSIS-NN**: Only ARM Cortex-M.  
2. **Hybrid Mode**: Requires FPU or float emulation.  
3. **PROGMEM**: Only available on AVR.

## **Recommendations for Embedded Projects**

### **Which Method to Use?**

#### **🏆 Top Recommendation: CMSIS-NN (if available)**

**For**: ARM Cortex-M4, M7, M33, M55

* ✅ Maximum speed and efficiency.  
* ✅ Full int8 inference.  
* ✅ Low power consumption.  
* ✅ Optimized kernels with SIMD.

**Usage Example**:

\# Train and export  
model \= MiniNeuralNetwork(n\_inputs=2, n\_hidden=4, n\_outputs=1)  
model.fit(dataset)  
\# Automatic quantization in export\_to\_c()  
code \= ml\_manager.export\_to\_c("my\_model")

#### **🥈 Second Option: AVR Hybrid Mode**

**For**: Arduino Uno, Nano, and other 8-bit AVRs

* ✅ Saves SRAM (data in Flash).  
* ✅ Portable (no dependencies).  
* ✅ Compatible with limited hardware.

**Usage Example**:

\# Similar to above, but export\_to\_c() will fallback to to\_arduino\_code()  
\# if CMSISAdapter is not available  
code \= ml\_manager.export\_to\_c("my\_model")

#### **🥉 Third Option: Native Per-Layer**

**For**: Projects requiring maximum portability

* ✅ No external dependencies.  
* ✅ Works on any platform.  
* ⚠️ Slower than CMSIS-NN.

### **Hardware Selection Guide**

| Hardware | MCU | RAM | Flash | FPU | Recommendation |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **STM32F4** | Cortex-M4 | 192KB | 1MB | Yes | CMSIS-NN |
| **STM32F7** | Cortex-M7 | 512KB | 2MB | Yes | CMSIS-NN |
| **Arduino Uno** | AVR | 2KB | 32KB | No | Hybrid Mode |
| **ESP32** | Xtensa | 520KB | 4MB | Yes | CMSIS-NN (if ported) |
| **Raspberry Pi Pico** | Cortex-M0+ | 264KB | 2MB | No | Hybrid Mode |

### **Best Practices**

1. **Always Calibrate with Representative Dataset**  
   \# Use the same training dataset  
   model.fit(training\_dataset)  \# Calibrates automatically

2. **Validate Post-Quantization Accuracy**  
   \# Compare accuracy before and after  
   accuracy\_before \= evaluate(model, test\_set)  
   model.quantize()  
   accuracy\_after \= evaluate\_quantized(model, test\_set)  
   assert accuracy\_after \>= accuracy\_before \- 0.02  \# 2% tolerance

3. **Use Input Scaling**  
   \# Scaler helps maintain consistent ranges  
   ml\_manager.train\_pipeline(  
       model\_name="model",  
       dataset=data,  
       model\_type="neural\_network",  
       scaling="minmax"  \# Recommended  
   )

4. **Optimize Architecture for Quantization**  
   * Use ReLU activations (friendlier to quantization).  
   * Avoid very deep layers.  
   * Limit weight range during training.

## **Usage Examples**

### **Example 1: Full Training and Export**

from miniml import ml\_manager

\# Example Dataset (XOR)  
dataset \= \[  
    \[0.0, 0.0, 0\],  
    \[0.0, 1.0, 1\],  
    \[1.0, 0.0, 1\],  
    \[1.0, 1.0, 0\]  
\]

\# Train with scaling  
result \= ml\_manager.train\_pipeline(  
    model\_name="xor\_nn",  
    dataset=dataset,  
    model\_type="neural\_network",  
    params={  
        "n\_inputs": 2,  
        "n\_hidden": 4,  
        "n\_outputs": 1,  
        "epochs": 2000,  
        "learning\_rate": 0.1  
    },  
    scaling="minmax"  
)

\# Export to C (automatic quantization)  
c\_code \= ml\_manager.export\_to\_c("xor\_nn")

\# Save code  
with open("xor\_model.h", "w") as f:  
    f.write(c\_code)

### **Example 2: Manual Quantization**

from miniml.ml\_runtime import MiniNeuralNetwork

\# Create and train model  
model \= MiniNeuralNetwork(n\_inputs=2, n\_hidden=4, n\_outputs=1)  
model.fit(dataset)

\# Calibration (automatic after fit, but can be manual)  
model.calibrate(dataset)

\# Explicit Quantization  
model.quantize(per\_channel=True)

\# Verify quantization  
print(f"Quantized: {model.quantized}")  
print(f"Act scales: {model.act\_scales}")  
print(f"W1 shape: {len(model.q\_W1)}x{len(model.q\_W1\[0\])}")

### **Example 3: Using CMSISAdapter Directly**

from miniml.ml\_runtime import MiniNeuralNetwork  
from adapters.cmsis\_nn.adapter import CMSISAdapter

\# Train model  
model \= MiniNeuralNetwork(n\_inputs=2, n\_hidden=4, n\_outputs=1)  
model.fit(dataset)  
model.quantize()

\# Generate CMSIS-NN code  
adapter \= CMSISAdapter(model)  
adapter.generate\_c("model\_cmsis.h")

### **Example 4: Saving and Loading Quantized Model**

\# Save model (includes act\_scales and quantized weights)  
ml\_manager.save\_model("xor\_nn", "xor\_nn.json")

\# Load model (restores act\_scales)  
ml\_manager.load\_model("xor\_nn", "xor\_nn.json")

\# Re-quantize if necessary  
model \= ml\_manager.get\_model("xor\_nn")  
if not model.quantized:  
    model.quantize()

\# Export  
c\_code \= ml\_manager.export\_to\_c("xor\_nn")

## **Conclusion**

MiniML offers a robust and flexible quantization system for neural networks, with three main modes adapted to different platforms:

1. **CMSIS-NN**: Maximum speed for ARM Cortex-M.  
2. **AVR Hybrid Mode**: Ideal for 8-bit Arduino.  
3. **Native Per-Layer**: Portable and universal.

Quantization reduces model size by \~75% and accelerates inference 2-10x, with typical accuracy loss \< 2%, making it ideal for embedded AI applications.

Last Update: 2024  
MiniML Version: 1.0.1