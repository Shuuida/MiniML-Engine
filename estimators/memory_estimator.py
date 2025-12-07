"""
memory_estimator.py
Estimaciones de uso de Flash y SRAM para modelos MiniML exportados a C.
Ajustado para la arquitectura de runtime actual (Soporte Int8 selectivo).
"""
from typing import Any, Dict

def estimate_memory(model: Any, quantized: bool = False, target_flash: int = None, target_sram: int = None) -> Dict[str, Any]:
    """
    Estima el uso de Flash (PROGMEM) y SRAM (Stack/Heap) para modelos MiniML.
    
    Args:
        model: Instancia del modelo entrenado (MiniML).
        quantized (bool): Si True, simula el uso de memoria en modo int8 (solo si el modelo lo soporta).
        target_flash (int): Límite de Flash deseado (bytes) para generar advertencias.
        target_sram (int): Límite de SRAM deseado (bytes).

    Returns:
        Dict con 'flash_bytes', 'sram_bytes_peak', 'warnings', 'too_big'.
    """
    flash = 0
    sram_peak = 0
    warnings = []

    # Detectar si el modelo ya está cuantizado internamente
    is_model_quantized = getattr(model, 'quantized', False)
    # La cuantización efectiva aplica si el modelo ya lo está, o si el usuario lo solicita y el modelo lo soporta
    # Nota: Linear, SVM y KNN actualmente NO soportan exportación int8 en ml_runtime.
    supports_quantization = hasattr(model, 'quantize') or isinstance(model, (dict,)) # Dict asumo genérico
    
    effective_quantized = is_model_quantized or (quantized and supports_quantization)
    
    # Peso base: 1 byte si es int8, 4 bytes si es float
    # PERO cuidado: SVM/Linear/KNN siempre son float en la versión actual del runtime
    base_weight_size = 1 if effective_quantized else 4

    # ----------------------------------------------------
    # ESTIMACIÓN DEL MODELO

    # ÁRBOLES Y BOSQUES (DecisionTree / RandomForest)
    if hasattr(model, 'root') or hasattr(model, 'trees'):
        # Árboles usan int/float mixtos, la cuantización de pesos no aplica directamente a la estructura
        # pero los valores de hoja podrían ser int en clasificación. Asumimos estándar C export.
        try:
            # Intentar importar la utilidad de aplanado de forma segura
            try:
                from miniml.ml_compat import _flatten_tree_to_arrays
            except ImportError:
                print (f"No se ha podido utilizar el aplanado de forma segura: Error en la exportación")

            def get_tree_size(root_node):
                if not root_node: return 0
                struct = _flatten_tree_to_arrays(root_node)
                n_nodes = len(struct.get('feature_index', []))
                # Estructura C por nodo:
                # int feature_index (2) + float threshold (4) + int left (2) + int right (2) + value (4 o 2)
                # Total aprox: 14 bytes por nodo en arrays paralelos
                return n_nodes * 14

            if hasattr(model, 'trees'): # RandomForest
                total_tree_flash = 0
                max_tree_depth = 0
                for t in model.trees:
                    total_tree_flash += get_tree_size(getattr(t, 'root', None))
                    max_tree_depth = max(max_tree_depth, getattr(t, 'max_depth', 10))
                flash += total_tree_flash
                # SRAM: Stack recursivo O(1) en exportación iterativa, pero necesita variables locales
                # + buffer de votos para el ensemble: int votes[n_trees]
                sram_peak += (len(model.trees) * 2) + 64 
            else: # DecisionTree
                flash += get_tree_size(model.root)
                sram_peak += 32 # Variables locales de la función predict

        except Exception as e:
            warnings.append(f"Tree estimation error: {str(e)}")

    # REDES NEURONALES (MiniNeuralNetwork)
    elif hasattr(model, 'W1') and hasattr(model, 'W2'):
        try:
            # Aquí sí aplica la cuantización int8 vs float
            nn_weight_size = 1 if effective_quantized else 4
            
            # Flash: Pesos + Biases
            # W1: n_hidden * n_inputs
            rows1 = len(model.W1)
            cols1 = len(model.W1[0]) if rows1 > 0 else 0
            # W2: n_outputs * n_hidden
            rows2 = len(model.W2)
            cols2 = len(model.W2[0]) if rows2 > 0 else 0
            
            weights_count = (rows1 * cols1) + (rows2 * cols2)
            biases_count = rows1 + rows2 # B1 + B2
            
            # En exportación C:
            # Pesos = nn_weight_size
            # Biases = siempre float (4 bytes) en la implementación actual
            # Escalas (si cuantizado) = float (4 bytes) * 2
            
            flash += (weights_count * nn_weight_size) + (biases_count * 4)
            if effective_quantized:
                flash += 8 # Escalas W1_scale, W2_scale

            # SRAM: Activaciones
            # La implementación C declara float a1[...] y float a2[...] en el mismo scope.
            # Se suman, no se hace max.
            # float a1[hidden] + float a2[output]
            sram_activations = (rows1 + rows2) * 4 # Siempre float
            sram_peak += sram_activations + 32 # Overhead función

        except Exception:
            warnings.append('Could not introspect NN matrices.')

    # LINEAR / SVM (Pesos planos)
    elif hasattr(model, 'weights') and isinstance(model.weights, list):
        # Actualmente runtime exporta siempre como float array
        n_weights = len(model.weights)
        flash += n_weights * 4 
        sram_peak += 16 # Variable acumuladora 's'

    # KNN (Dataset embebido)
    elif hasattr(model, 'X_train') and model.X_train:
        # KNN exporta todo el dataset como const float[]
        try:
            n_samples = len(model.X_train)
            n_features = len(model.X_train[0]) if n_samples > 0 else 0
            
            # Dataset X (float) + Labels y (int o float)
            bytes_x = n_samples * n_features * 4
            bytes_y = n_samples * (4 if getattr(model, 'task', '') == 'regression' else 2) # int en AVR son 2
            
            dataset_bytes = bytes_x + bytes_y
            flash += dataset_bytes
            warnings.append(f'KNN embeds full dataset (~{dataset_bytes/1024:.1f} KB). High Flash usage.')
            
            # SRAM: Distancias e Indices arrays
            # float distances[n_samples] + int indices[n_samples]
            sram_peak += (n_samples * 4) + (n_samples * 2)
            
        except Exception:
            warnings.append('KNN dataset size unknown.')

    # -------------------------------------------
    # ESTIMACIÓN DEL PREPROCESAMIENTO (SCALER)
    if hasattr(model, 'scaler') and model.scaler:
        # El scaler exporta arrays float para min/max/mean/std
        # Generalmente 2 arrays de tamaño n_features
        try:
            n_feats = getattr(model.scaler, 'n_features_trained', 0)
            if n_feats:
                # 2 arrays de floats (ej. min y denom)
                scaler_flash = n_feats * 4 * 2
                flash += scaler_flash
                # El scaler opera in-place sobre el array de entrada, bajo impacto SRAM
        except Exception:
            pass

    # ---------------------------------
    # CHEQUEOS FINALES
    too_big = False
    
    # Umbrales típicos Arduino Uno (ATMega328P): Flash 32KB, SRAM 2KB
    if target_flash and flash > target_flash:
        warnings.append(f'CRITICAL: Flash estimate ({flash} B) > Target ({target_flash} B).')
        too_big = True
    
    if target_sram and sram_peak > target_sram:
        warnings.append(f'CRITICAL: SRAM peak ({sram_peak} B) > Target ({target_sram} B). Stack overflow risk.')
        too_big = True

    return {
        'flash_bytes': int(flash),
        'sram_bytes_peak': int(sram_peak),
        'warnings': warnings,
        'too_big': too_big,
        'quantized_est': effective_quantized
    }