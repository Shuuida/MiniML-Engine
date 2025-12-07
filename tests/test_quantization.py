from miniml import ml_runtime
from adapters.cmsis_nn.adapter import CMSISAdapter

def test_quantization_workflow():
    print("==============================================")
    print("🚀 TEST: Quantization & CMSIS Adapter Workflow")
    print("==============================================")

    # 1. SETUP: Usar ReLU para mayor estabilidad en int8
    print("\n[1] Entrenando modelo (XOR) con ReLU...")
    X = [[0,0], [0,1], [1,0], [1,1]]
    y = [[0], [1], [1], [0]]
    
    # IMPORTANTE: hidden_activation='relu'
    nn = ml_runtime.MiniNeuralNetwork(n_inputs=2, n_hidden=8, n_outputs=1, seed=42)
    nn.hidden_activation = 'relu' 
    nn.output_activation = 'relu' # Usamos ReLU en salida para test simple (0 a 1+)
    
    dataset = [x + yi for x, yi in zip(X, y)]
    nn.fit(dataset)
    
    # 2. CALIBRACIÓN
    # Nota: calibrate() se llama automáticamente después de fit() (línea 988 en ml_runtime.py)
    # Verificamos que act_scales se haya calculado correctamente
    print("\n[2] Verificando calibración...")
    if not hasattr(nn, 'act_scales') or not nn.act_scales:
        raise RuntimeError("act_scales no se calcularon automáticamente después de fit()")
    print(f"   Escalas: {nn.act_scales}")

    # 3. CUANTIFICACIÓN
    print("\n[3] Cuantificando...")
    nn.quantize(per_channel=True)

    # 4. SIMULACIÓN C (Validación Numérica)
    print("\n[4] Validando Lógica Aritmética...")
    
    def simulate_c_inference(input_vec, model):
        # Quantize Input
        s_in = model.act_scales['input']
        input_s8 = [max(-128, min(127, int(round(v / s_in)))) for v in input_vec]
        
        # HIDDEN LAYER
        hidden_s8 = []
        for i in range(model.n_hidden):
            acc = model.i32_B1[i]
            for j in range(model.n_inputs):
                acc += int(model.q_W1[i][j]) * int(input_s8[j])
            
            # Requantize
            val = int(round(acc * model.requant_mult1[i]))
            
            # ACTIVATION (Simular lo que hará el C)
            if getattr(model, 'hidden_activation', 'sigmoid') == 'relu':
                val = max(0, val)
                
            hidden_s8.append(max(-128, min(127, val)))

        # OUTPUT LAYER
        output_s8 = []
        for k in range(model.n_outputs):
            acc = model.i32_B2[k]
            for i in range(model.n_hidden):
                acc += int(model.q_W2[k][i]) * int(hidden_s8[i])
            
            val = int(round(acc * model.requant_mult2[k]))
            
            # ACTIVATION
            if getattr(model, 'output_activation', 'sigmoid') == 'relu':
                val = max(0, val)
                
            output_s8.append(max(-128, min(127, val)))
            
        # Dequantize Output para comparar
        return [v * model.act_scales['output'] for v in output_s8]

    # Prueba
    test_in = [0, 1]
    pred_float = nn.predict([test_in])[0][0]
    pred_sim = simulate_c_inference(test_in, nn)[0]
    
    print(f"   Float: {pred_float:.4f} | Simulado C: {pred_sim:.4f}")
    err = abs(pred_float - pred_sim)
    
    if err < 0.15:
        print("   ✅ Precisión Aceptable.")
    else:
        print(f"   ❌ Error alto ({err:.4f}).")

    # 5. GENERACIÓN C
    print("\n[5] Generando Adapter...")
    adapter = CMSISAdapter(nn)
    adapter.generate_c("test_cmsis.h")
    
    with open("test_cmsis.h", "r") as f:
        content = f.read()
        # Verificar definiciones correctas del nuevo adapter
        checks = ["const int32_t MULT1", "const float MULT1_F", "arm_fully_connected_s8"]
        if all(c in content for c in checks):
             print("   ✅ Código C generado correctamente con todas las definiciones.")
        else:
             print("   ❌ Faltan definiciones en el código C.")

if __name__ == "__main__":
    test_quantization_workflow()