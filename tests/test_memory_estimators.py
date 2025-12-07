"""
Test Suite para memory_estimator.py
-----------------------------------
Valida:
1. Estimación precisa de Flash/SRAM para distintos modelos.
2. Detección correcta de reducción de tamaño por Cuantización (int8).
3. Inmutabilidad de modelos que no soportan cuantización (Linear/SVM).
4. Generación de advertencias críticas (Stack Overflow, Flash Exceeded).
5. Detección de consumo masivo de memoria en KNN.
"""

import unittest
from miniml import ml_runtime
from estimators import memory_estimator

class TestMemoryEstimator(unittest.TestCase):

    def setUp(self):
        # Dataset simple para pruebas rápidas
        self.data_xor = [[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]]
        self.data_reg = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]

    def test_decision_tree_estimation(self):
        print("\n[TEST] Árbol de Decisión (Flash & SRAM)...")
        model = ml_runtime.DecisionTreeClassifier(max_depth=3)
        model.fit(self.data_xor)
        
        est = memory_estimator.estimate_memory(model)
        
        # Un árbol pequeño debe ocupar algo de flash (>0) y poca SRAM
        print(f"   > Flash: {est['flash_bytes']} B, SRAM: {est['sram_bytes_peak']} B")
        self.assertGreater(est['flash_bytes'], 0)
        self.assertGreater(est['sram_bytes_peak'], 0)
        self.assertFalse(est['too_big'], "Un árbol simple no debería exceder límites por defecto")

    def test_quantization_reduction_nn(self):
        print("\n[TEST] Reducción por Cuantización en Neural Network...")
        # Crear una NN mediana
        nn = ml_runtime.MiniNeuralNetwork(n_inputs=2, n_hidden=10, n_outputs=1)
        nn.fit(self.data_xor) # Inicializa matrices
        
        # 1. Estimación en modo Float (Normal)
        est_float = memory_estimator.estimate_memory(nn, quantized=False)
        flash_float = est_float['flash_bytes']
        
        # 2. Estimación en modo Int8 (Simulado)
        # Nota: Pasamos quantized=True al estimador para ver la predicción
        est_int8 = memory_estimator.estimate_memory(nn, quantized=True)
        flash_int8 = est_int8['flash_bytes']
        
        print(f"   > Float Flash: {flash_float} B")
        print(f"   > Int8 Flash:  {flash_int8} B")
        
        # Validación: La versión int8 debe ser significativamente menor (aprox 1/4 en pesos)
        # No es exactamente 1/4 porque los biases siguen siendo float, pero debe bajar notablemente.
        self.assertLess(flash_int8, flash_float * 0.6, "La cuantización no redujo la memoria significativamente")
        self.assertTrue(est_int8['quantized_est'], "El estimador no reportó modo cuantizado efectivo")

    def test_no_quantization_linear(self):
        print("\n[TEST] Inmutabilidad de Linear Model ante Cuantización...")
        lin = ml_runtime.MiniLinearModel()
        lin.fit(self.data_reg)
        
        # Linear Model NO soporta int8 en este framework, siempre es float.
        # El estimador debe ser inteligente y dar el mismo tamaño aunque pidamos quantized=True.
        est_float = memory_estimator.estimate_memory(lin, quantized=False)
        est_force = memory_estimator.estimate_memory(lin, quantized=True)
        
        print(f"   > Linear Float: {est_float['flash_bytes']} B")
        print(f"   > Linear 'Int8': {est_force['flash_bytes']} B")
        
        self.assertEqual(est_float['flash_bytes'], est_force['flash_bytes'], 
                         "El modelo lineal cambió de tamaño erróneamente (no soporta int8)")

    def test_knn_memory_warning(self):
        print("\n[TEST] Advertencia de Memoria Masiva en KNN...")
        # Crear un dataset "grande" para un microcontrolador (ej. 100 muestras)
        large_data = [[float(i), float(i)] + [0] for i in range(100)]
        
        knn = ml_runtime.KNearestNeighbors(k=3)
        knn.fit(large_data)
        
        est = memory_estimator.estimate_memory(knn)
        print(f"   > KNN Flash: {est['flash_bytes']} B")
        print(f"   > Advertencias: {est['warnings']}")
        
        # Debe haber una advertencia específica sobre el dataset embebido
        has_warning = any("KNN embeds full dataset" in w for w in est['warnings'])
        self.assertTrue(has_warning, "No se generó advertencia sobre el dataset de KNN")

    def test_limits_exceeded(self):
        print("\n[TEST] Detección de Límites Excedidos (Too Big)...")
        # Usamos una NN y ponemos límites ridículamente bajos para forzar el error
        nn = ml_runtime.MiniNeuralNetwork(n_inputs=10, n_hidden=20, n_outputs=5)
        # Inicializamos pesos aleatorios (simulamos fit)
        
        # Límite: 10 bytes de Flash (imposible para una NN)
        est = memory_estimator.estimate_memory(nn, target_flash=10, target_sram=1000)
        
        print(f"   > Flash estimada: {est['flash_bytes']} B vs Target: 10 B")
        print(f"   > Too Big Flag: {est['too_big']}")
        print(f"   > Warnings: {est['warnings']}")
        
        self.assertTrue(est['too_big'], "No se marcó 'too_big' al exceder flash")
        self.assertTrue(any("Flash estimate" in w for w in est['warnings']), "Falta advertencia de Flash")

    def test_pipeline_scaler_overhead(self):
        print("\n[TEST] Overhead del Scaler en Memoria...")
        # Modelo sin scaler
        lin = ml_runtime.MiniLinearModel()
        lin.fit(self.data_reg)
        base_flash = memory_estimator.estimate_memory(lin)['flash_bytes']
        
        # Agregar scaler manualmente
        scaler = ml_runtime.MiniScaler()
        scaler.fit(self.data_reg)
        lin.scaler = scaler # Simulamos lo que hace ml_manager
        
        est_with_scaler = memory_estimator.estimate_memory(lin)
        scaler_flash = est_with_scaler['flash_bytes']
        
        print(f"   > Base Flash: {base_flash} B")
        print(f"   > Con Scaler: {scaler_flash} B")
        
        self.assertGreater(scaler_flash, base_flash, "El Scaler no sumó bytes a la estimación")

if __name__ == '__main__':
    print("==============================================")
    print("   VALIDACIÓN DE ESTIMADOR DE MEMORIA (C)     ")
    print("==============================================")
    unittest.main(argv=['first-arg-is-ignored'], exit=False)