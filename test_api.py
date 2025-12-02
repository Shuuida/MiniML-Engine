import json
import time
import traceback
from core import ml_runtime, ml_manager

def trace_prediction(node, row, depth=0):
    """Traza el recorrido de una predicción a través del árbol."""
    indent = "  " * depth
    if node is None:
        return f"{indent}❌ Nodo nulo — la predicción no pudo continuar.\n"

    # Si el nodo es una hoja, devolvemos su valor
    if not isinstance(node, dict) or ("index" not in node and "value" in node):
        return f"{indent}🌿 Hoja alcanzada → predicción final: {node}\n"

    if "index" not in node or "value" not in node:
        return f"{indent}⚠️ Nodo malformado: {node}\n"

    i = node["index"]
    v = node["value"]

    # Validación de seguridad
    if i >= len(row):
        return f"{indent}❌ Índice fuera de rango (index={i}, row_len={len(row)})\n"

    cond = row[i] < v
    branch = "izquierda" if cond else "derecha"
    log = f"{indent}🔎 Nodo (index={i}, value={v}) → row[{i}]={row[i]} → rama {branch}\n"

    # Descenso recursivo
    next_node = node["left"] if cond else node["right"]
    return log + trace_prediction(next_node, row, depth + 1)


def inspect_tree(node, depth=0):
    """Recorre e imprime la estructura del árbol."""
    indent = "  " * depth
    if node is None:
        return f"{indent}❌ Nodo vacío\n"

    if isinstance(node, dict):
        lines = [f"{indent}🌳 Nodo nivel {depth}: índice={node.get('index')} valor={node.get('value')}"]
        if "left" in node or "right" in node:
            lines.append(f"{indent}├── Izquierda:")
            lines.append(inspect_tree(node.get("left"), depth + 1))
            lines.append(f"{indent}├── Derecha:")
            lines.append(inspect_tree(node.get("right"), depth + 1))
        return "\n".join(lines)
    else:
        return f"{indent}🌿 Hoja: {node}\n"


def run_pipeline(mode="classification"):
    print(f"\n--- 🧩 Ejecutando diagnóstico para modo: {mode.upper()} ---\n")

    # Dataset base
    if mode == "classification":
        data = [
            [2.7, 2.5, 0],
            [1.3, 3.5, 0],
            [3.5, 1.4, 1],
            [3.9, 4.0, 1]
        ]
        X_test = [[2.5, 2.3], [3.7, 3.9]]
    else:
        data = [
            [1.0, 2.0, 2.1],
            [2.0, 3.0, 3.9],
            [3.0, 4.0, 6.1],
            [4.0, 5.0, 8.2]
        ]
        X_test = [[2.5, 3.5], [3.5, 4.5]]

    start = time.time()
    print("🧠 Entrenando modelo...")

    try:
        result = ml_manager.train_decision_tree(
            model_name=f"tree_{mode}",
            dataset=data,
            max_depth=3,
            min_size=1,
            backend="mini"  # <-- FORZAR EL BACKEND 'MINI' PARA DEPURACIÓN
        )
        print("✅ Entrenamiento completado:", result)
    except Exception as e:
        print("❌ Error durante el entrenamiento:", e)
        traceback.print_exc()
        return None

    model_info = ml_manager._MODEL_REGISTRY.get(f"tree_{mode}")
    if not model_info:
        print("❌ Modelo no encontrado en el registro global.")
        return None

    model = model_info["model"]
    print("\n📊 Estructura del árbol raíz:\n")
    print(inspect_tree(model.root))

    preds = []
    print("\n🔍 Trazando predicciones una a una:\n")

    for i, row in enumerate(X_test):
        print(f"📘 Ejemplo #{i + 1} → {row}")
        try:
            trace_log = trace_prediction(model.root, row)
            print(trace_log)
            pred = None
            if hasattr(model, "predict_tree"):
                pred = model.predict_tree(model.root, row)
            elif hasattr(model, "predict_tree_regression"):
                pred = model.predict_tree_regression(model.root, row)
            else:
                # Fallback al método estándar predict()
                pred = model.predict([row])[0]
            print(f"✅ Predicción final: {pred}\n")
            preds.append(pred)
        except Exception as e:
            print(f"❌ Error al predecir fila {i}: {e}")
            traceback.print_exc()
            preds.append(None)

    duration = time.time() - start
    return {
        "mode": mode,
        "predictions": preds,
        "duration": duration,
        "tree_type": model_info["type"]
    }


def main():
    print("=" * 60)
    print("🧪 DIAGNÓSTICO AVANZADO MINI ML - ML Runtime")
    print("=" * 60)

    results = []
    for m in ["classification", "regression"]:
        res = run_pipeline(m)
        if res:
            results.append(res)

    print("\n\n--- RESULTADOS FINALES ---\n")
    print(json.dumps(results, indent=2))

    with open("ml_debug_log.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(results, indent=2))
    print("\n🗂️ Log detallado guardado como 'ml_debug_log.txt'")


if __name__ == "__main__":
    main()