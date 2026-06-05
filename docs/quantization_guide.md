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

* **Memory Reduction**: ~75% less space usage (int8 vs float32).  
* **Acceleration**: Faster integer operations on MCUs lacking an FPU.  
* **CMSIS-NN Compatibility**: Integration with optimized ARM kernels.  
* **Preserved Precision**: Typical accuracy loss is < 2%.

## **Quantization Architecture**

### **Core Components**

#### **1. MiniNeuralNetwork (ml_runtime.py)**

The base class that implements:

* Activation calibration (calibrate()).  
* Weight and bias quantization (quantize()).  
* Native export to C (to_arduino_code()).

#### **2. CMSISAdapter (adapters/cmsis_nn/adapter.py)**

An advanced adapter that generates code compatible with:

* **CMSIS-NN**: Optimized ARM kernels for Cortex-M.  
* **Portable Fallback**: Standard C implementation with no dependencies.

### **Quantization Flow**

graph TD;  
    A[Training (Float32)] --> B[Calibration (calibrate())];  
    B --> C[Quantization (quantize())];  
    C --> D[Export (export_to_c())];  
    D --> E[C Firmware (int8 inference)];

1. **Calibration**: Calculates act_scales for input, hidden, and output layers.  
2. **Quantization**: Converts weights to int8 and biases to int32.  
3. **Export**: Generates optimized C code.

## **Quantization Methods**

### **1. Per-Layer Symmetric Quantization**

**Implementation**: Default method in MiniNeuralNetwork.quantize().

#### **Characteristics:**

* **Weights**: Quantized to int8 with range [-127, 127].  
* **Zero-point**: Implicitly 0 (symmetric quantization).  
* **Scale**: One scale per weight matrix (per-layer).  
* **Biases**: Quantized to int32 to preserve precision.

#### **Formulas:**

**For Weights (W):**

abs_max = max(abs(min(W)), abs(max(W)))  
scale_w = abs_max / 127.0  
q_w = round(w / scale_w)  # Clipped to [-127, 127]

**For Biases (B):**

effective_scale = input_scale * scale_w  
b_int32 = round(b / effective_scale)  # Clipped to int32 range

**Requantization Multiplier:**

requant_mult = effective_scale / output_scale

#### **Pros:**

* ✅ Simple and fast implementation.  
* ✅ Low computational overhead.  
* ✅ CMSIS-NN compatible.  
* ✅ Good accuracy for small-to-medium networks.

#### **Cons:**

* ❌ Lower accuracy than per-channel quantization for large networks.  
* ❌ Sensitive to outliers in weight distribution.

### **2. CMSIS-NN Quantization (Fixed-Point)**

**Implementation**: CMSISAdapter.generate_c().

#### **Characteristics:**

* **Weights**: int8_t stored in aligned arrays.  
* **Biases**: Pre-quantized int32_t.  
* **Multipliers**: Converted to Q31 format (significand + shift).  
* **Activations**: int8_t throughout the entire inference.

#### **Q31 Format for Multipliers:**

```python
def _quantize_multiplier(real_multiplier: float) -> Tuple[int, int]:  
    significand, shift = math.frexp(real_multiplier)  
    q_mult = int(round(significand * (1 << 31)))  
    # Adjustment to avoid overflow  
    if q_mult == (1 << 31):  
        q_mult //= 2  
        shift += 1  
    return q_mult, shift

```

#### **Pros:**

* ✅ Maximum speed on Cortex-M (SIMD kernels).  
* ✅ Full int8 inference (no floats).  
* ✅ Low power consumption.  
* ✅ Standard ARM CMSIS-NN compatibility.

#### **Cons:**

* ❌ Requires CMSIS-NN library (not portable).  
* ❌ Limited to ARM Cortex-M architectures.  
* ❌ Higher implementation complexity.

### **3. AVR Hybrid Mode (Arduino 8-bit)**

**Implementation**: MiniNeuralNetwork.to_arduino_code().

#### **Characteristics:**

* **Weights**: int8_t stored in PROGMEM (Flash).  
* **Scales**: float stored in PROGMEM.  
* **Computation**: Hybrid (int8 storage, float compute).  
* **Biases**: Original float stored in PROGMEM.

#### **Strategy:**

```c
// Read quantized weight from Flash  
int8_t w = pgm_read_byte(&W1[i][j]);  
// On-the-fly dequantization  
float dequantized = (float)w * scale * input;

```

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

```python
def calibrate(self, dataset: List[List[float]]):  
    """  
    Calculates activation ranges (min/max) for Input, Hidden, and Output.  
    Essential for int8 Post-Training Quantization.  
    """  
    # Iterates over the dataset and finds:  
    # - max_in: absolute max of inputs  
    # - max_hidden: absolute max of hidden activations  
    # - max_out: absolute max of outputs  
      
    self.act_scales = {  
        'input': max_in / 127.0,  
        'hidden': max_hidden / 127.0,  
        'output': max_out / 127.0  
    }

```

**Important Notes**:

* Automatically executed after fit().  
* Requires a calibration dataset (typically the training set).  
* Scales are saved in act_scales for later use.

### **Step 2: Quantization (quantize())**

**Purpose**: Convert weights and biases from float32 to int8/int32.

```python
def quantize(self, per_channel: bool = True):  
    """  
    Quantizes weights to int8 and biases to int32.  
    Requires previously calculated act_scales.  
    """  
    # For each layer:  
    # 1. Calculate scale per weight row  
    # 2. Quantize weights to int8  
    # 3. Quantize biases to int32  
    # 4. Calculate requantization multipliers  
      
    self.q_W1, self.i32_B1, self.requant_mult1, self.s_W1_list = ...  
    self.q_W2, self.i32_B2, self.requant_mult2, self.s_W2_list = ...  
    self.quantized = True

```

**Generated Attributes**:

* q_W1, q_W2: Quantized weight matrices (int8).  
* i32_B1, i32_B2: Quantized bias vectors (int32).  
* requant_mult1, requant_mult2: Requantization multipliers (float).  
* s_W1_list, s_W2_list: Scales per weight row (float).

### **Step 3: Export (export_to_c())**

**Purpose**: Generate optimized C code for firmware.

**Automatic Detection**:

* If model has q_W1 → Uses CMSISAdapter (Preferred).  
* If that fails → Uses to_arduino_code() (Native fallback).  
* If scaler exists → Includes preprocessing code.

## **Exporting to C Firmware**

### **Option 1: CMSIS-NN (Recommended for ARM Cortex-M)**

**Generated by**: CMSISAdapter.generate_c().

**Features**:

* Code optimized for arm_fully_connected_s8().  
* Full int8 inference.  
* Portable fallback if CMSIS-NN is not enabled.

**Code Structure**:

```c
// Data arrays (aligned for SIMD)  
const int8_t W1[N] ALIGNED(4) = {...};  
const int32_t B1[M] ALIGNED(4) = {...};  
const int32_t MULT1[M] ALIGNED(4) = {...};  
const int32_t SHIFT1[M] ALIGNED(4) = {...};

#ifdef CMSISNN_ENABLED  
    // Uses optimized ARM kernels  
    arm_fully_connected_s8(...);  
#else  
    // Standard C portable fallback  
    // Manual loop implementation  
#endif

```

**Pros**:

* Maximum speed on Cortex-M4/M7.  
* Low energy consumption.  
* Pure int8 inference (no floats).

### **Option 2: AVR Hybrid Mode (Arduino 8-bit)**

**Generated by**: MiniNeuralNetwork.to_arduino_code().

**Features**:

* Weights in PROGMEM (Flash).  
* Hybrid calculation (int8 storage, float compute).  
* Ideal for AVR without FPU.

**Code Structure**:

```c
#include <avr/pgmspace.h>

// Quantized weights in Flash  
const int8_t W1[N][M] PROGMEM = {...};  
// Scales in Flash  
const float sW1[N] PROGMEM = {...};  
// Original biases in Flash  
const float B1[N] PROGMEM = {...};

void predict(float *row, float *out) {  
    // On-the-fly dequantization  
    float w = (float)pgm_read_byte(&W1[i][j]) * pgm_read_float(&sW1[i]);  
    // Computation in float  
    sum += w * input[j];  
}

```

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
| **Memory Required** | ~75% less | ~75% less | ~75% less (weights) |
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
| **Model Size** | 100% | ~25% | ~25% | ~25% (weights) |
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
   * ⚠️ In to_arduino_code(), activations are calculated in float.  
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

```python
# Train and export  
model = MiniNeuralNetwork(n_inputs=2, n_hidden=4, n_outputs=1)  
model.fit(dataset)  
# Automatic quantization in export_to_c()  
code = ml_manager.export_to_c("my_model")

```

#### **🥈 Second Option: AVR Hybrid Mode**

**For**: Arduino Uno, Nano, and other 8-bit AVRs

* ✅ Saves SRAM (data in Flash).  
* ✅ Portable (no dependencies).  
* ✅ Compatible with limited hardware.

**Usage Example**:

# Similar to above, but export_to_c() will fallback to to_arduino_code()  
# if CMSISAdapter is not available  
code = ml_manager.export_to_c("my_model")

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
   # Use the same training dataset  
   model.fit(training_dataset)  # Calibrates automatically

2. **Validate Post-Quantization Accuracy**  
   # Compare accuracy before and after  
   accuracy_before = evaluate(model, test_set)  
   model.quantize()  
   accuracy_after = evaluate_quantized(model, test_set)  
   assert accuracy_after >= accuracy_before - 0.02  # 2% tolerance

3. **Use Input Scaling**  
   # Scaler helps maintain consistent ranges  
   ml_manager.train_pipeline(  
       model_name="model",  
       dataset=data,  
       model_type="neural_network",  
       scaling="minmax"  # Recommended  
   )

4. **Optimize Architecture for Quantization**  
   * Use ReLU activations (friendlier to quantization).  
   * Avoid very deep layers.  
   * Limit weight range during training.

## **Usage Examples**

### **Example 1: Full Training and Export**


```python
from miniml import ml_manager

# Example Dataset (XOR)  
dataset = [  
    [0.0, 0.0, 0],  
    [0.0, 1.0, 1],  
    [1.0, 0.0, 1],  
    [1.0, 1.0, 0]  
]

# Train with scaling  
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

# Export to C (automatic quantization)  
c_code = ml_manager.export_to_c("xor_nn")

# Save code  
with open("xor_model.h", "w") as f:  
    f.write(c_code)

```

### **Example 2: Manual Quantization**

```python
from miniml.ml_runtime import MiniNeuralNetwork

# Create and train model  
model = MiniNeuralNetwork(n_inputs=2, n_hidden=4, n_outputs=1)  
model.fit(dataset)

# Calibration (automatic after fit, but can be manual)  
model.calibrate(dataset)

# Explicit Quantization  
model.quantize(per_channel=True)

# Verify quantization  
print(f"Quantized: {model.quantized}")  
print(f"Act scales: {model.act_scales}")  
print(f"W1 shape: {len(model.q_W1)}x{len(model.q_W1[0])}")

```

### **Example 3: Using CMSISAdapter Directly**

```python
from miniml.ml_runtime import MiniNeuralNetwork  
from adapters.cmsis_nn.adapter import CMSISAdapter

# Train model  
model = MiniNeuralNetwork(n_inputs=2, n_hidden=4, n_outputs=1)  
model.fit(dataset)  
model.quantize()

# Generate CMSIS-NN code  
adapter = CMSISAdapter(model)  
adapter.generate_c("model_cmsis.h")

```

```python
### **Example 4: Saving and Loading Quantized Model**

# Save model (includes act_scales and quantized weights)  
ml_manager.save_model("xor_nn", "xor_nn.json")

# Load model (restores act_scales)  
ml_manager.load_model("xor_nn", "xor_nn.json")

# Re-quantize if necessary  
model = ml_manager.get_model("xor_nn")  
if not model.quantized:  
    model.quantize()

# Export  
c_code = ml_manager.export_to_c("xor_nn")

```

## **Conclusion**

MiniML offers a robust and flexible quantization system for neural networks, with three main modes adapted to different platforms:

1. **CMSIS-NN**: Maximum speed for ARM Cortex-M.  
2. **AVR Hybrid Mode**: Ideal for 8-bit Arduino.  
3. **Native Per-Layer**: Portable and universal.

Quantization reduces model size by ~75% and accelerates inference 2-10x, with typical accuracy loss < 2%, making it ideal for embedded AI applications.

Last Update: 2024  
MiniML Version: 1.0.1