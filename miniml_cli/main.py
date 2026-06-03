"""
MiniML CLI - Interfaz de Línea de Comandos
==========================================
Punto de entrada principal para operaciones MLOps.
Soporta inspección de modelos, estimación de memoria embebida,
recolección de sensores y un entorno interactivo de simulación (REPL).
"""

import argparse
import sys
import os
import json
import time

# Importaciones del motor MiniML
from miniml.ml_exporter import deserialize_model, print_cli_summary
from estimators.memory_estimator import estimate_memory
from miniml.hardware.serial_manager import serial_manager

try:
    from miniml.autograd.tensor import Tensor
    _HAS_TENSOR = True
except ImportError:
    _HAS_TENSOR = False


def load_model_from_json(filepath):
    """Utilidad para cargar y reconstruir un modelo desde disco."""
    if not os.path.exists(filepath):
        print(f"[ERROR] El archivo '{filepath}' no existe.")
        sys.exit(1)
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        model = deserialize_model(data)
        if model is None:
            print("[ERROR] Fallo al reconstruir el modelo. Formato inválido.")
            sys.exit(1)
        return model
    except Exception as e:
        print(f"[ERROR] Excepción al leer el modelo: {e}")
        sys.exit(1)


def cmd_inspect(args):
    """Comando: Muestra el resumen de la arquitectura del modelo."""
    model = load_model_from_json(args.model)
    print_cli_summary(model)


def cmd_estimate(args):
    """Comando: Calcula la memoria requerida para el microcontrolador."""
    model = load_model_from_json(args.model)
    
    # Inferir el shape. En CLI permitimos pasarlo como '1,28,28'
    shape_tuple = None
    if args.input_shape:
        try:
            shape_tuple = tuple(map(int, args.input_shape.split(',')))
        except:
            print("[WARN] Formato de input_shape inválido. Usa comas ej: 1,28,28")

    print("\nCalculando huella de memoria para hardware embebido...")
    report = estimate_memory(
        model=model,
        quantized=args.quantized,
        target_flash=args.flash,
        target_sram=args.sram,
        language=args.lang,
        input_shape=shape_tuple
    )

    print("\n" + "="*50)
    print(" 📊 Reporte de Memoria Edge AI (MiniML Estimator)")
    print("="*50)
    if "error" in report:
        print(f"[ERROR] {report['error']}")
    else:
        print(f" Memoria Flash (Programa) : {report['flash_bytes']} bytes / {report['flash_total']} bytes ({report['flash_percent']}%)")
        print(f" Memoria SRAM (Variables) : {report['sram_bytes']} bytes / {report['sram_total']} bytes ({report['sram_percent']}%)")
        print(f" Lenguaje de Exportación  : {report['language']}")
        print(f" Cuantización INT8 Activa : {'Sí' if report['quantized'] else 'No'}")
        
        if report['sram_percent'] > 90.0 or report['flash_percent'] > 90.0:
            print("\n [⚠️ ADVERTENCIA] El modelo está muy cerca del límite de memoria del chip.")
    print("="*50 + "\n")


def cmd_sensor(args):
    """Comando: Inicia el stream de sensores o simulador."""
    port = args.port
    label = args.label
    
    print(f"\nIniciando recolección de datos en: {port}")
    success = serial_manager.connect(port, baudrate=args.baudrate, verbose=args.verbose)
    
    if not success:
        sys.exit(1)

    if args.log:
        serial_manager.start_logging(filename=args.log, label=label)

    try:
        print("\nPresiona CTRL+C para detener la recolección de datos...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nDeteniendo sensor...")
    finally:
        serial_manager.disconnect()


def cmd_simulate(args):
    """
    Comando: Entorno interactivo REPL para probar modelos en tiempo real.
    Acepta entradas manuales o carga de datasets para evaluación en vivo.
    """
    model = load_model_from_json(args.model)
    is_deep_learning = hasattr(model, "layers")
    
    print("\n" + "="*60)
    print(" 🧠 MiniML REPL - Simulador de Inferencia en Tiempo Real")
    print("="*60)
    print(" Instrucciones:")
    print(" - Escribe valores separados por comas para predecir (ej: 1.2, 3.4, -0.5)")
    print(" - Escribe el path a un archivo .csv para evaluar un dataset por lotes.")
    print(" - Escribe 'salir' o presiona CTRL+C para terminar.")
    print("="*60 + "\n")

    while True:
        try:
            user_input = input("miniml-sim> ").strip()
            
            if user_input.lower() in ['salir', 'exit', 'quit']:
                break
            if not user_input:
                continue

            # Modo 1: Evaluación por Lotes (Dataset CSV)
            if user_input.endswith('.csv') and os.path.exists(user_input):
                print(f"[SIMULADOR] Procesando archivo: {user_input}")
                import csv
                with open(user_input, 'r') as f:
                    reader = csv.reader(f)
                    for row_idx, row in enumerate(reader):
                        try:
                            # Ignorar cabeceras si no son números
                            features = [float(x) for x in row if x.replace('.','',1).lstrip('-').isdigit()]
                            if not features: continue
                            
                            # Realizar predicción
                            if is_deep_learning and _HAS_TENSOR:
                                out = model(Tensor([features]))
                                result = out.data[0]
                            else:
                                result = model.predict([features])
                                
                            print(f"  Fila {row_idx} -> Input: {features[:3]}... | Output: {result}")
                            time.sleep(0.05) # Pequeña pausa visual simulando latencia de hardware
                        except Exception as e:
                            print(f"  [Error en fila {row_idx}]: {e}")
                print("[SIMULADOR] Fin del dataset.\n")
                continue

            # Modo 2: Evaluación Manual Dinámica
            try:
                # Parsear la entrada como lista de floats
                features = [float(x.strip()) for x in user_input.split(',')]
                
                # Tiempo de ejecución para medir latencia lógica
                start_t = time.perf_counter()
                
                if is_deep_learning and _HAS_TENSOR:
                    # Formato MiniTensor Autograd
                    out_tensor = model(Tensor([features]))
                    prediction = out_tensor.data[0]
                else:
                    # Formato MiniML Clásico
                    prediction = model.predict([features])
                    
                end_t = time.perf_counter()
                latency_ms = (end_t - start_t) * 1000

                # Formatear la salida simulando el Monitor Serial de Arduino
                print(f"  [Hardware Sim] Procesado en {latency_ms:.2f} ms")
                print(f"  [Red Neuronal] Reacción/Salida -> {prediction}\n")

            except ValueError:
                print("  [ERROR] Entrada inválida. Usa números separados por comas o un archivo .csv\n")
            except Exception as e:
                print(f"  [ERROR del Modelo] {e}\n")

        except KeyboardInterrupt:
            break

    print("\nSaliendo del simulador de inferencia.")


def main():
    parser = argparse.ArgumentParser(description="MiniML Engine CLI - Herramientas MLOps y Edge AI")
    subparsers = parser.add_subparsers(dest="command", help="Comandos disponibles", required=True)

    # Comando: inspect
    parser_inspect = subparsers.add_parser("inspect", help="Imprime el resumen de la arquitectura del modelo")
    parser_inspect.add_argument("--model", type=str, required=True, help="Ruta al archivo JSON del modelo")

    # Comando: estimate
    parser_estimate = subparsers.add_parser("estimate", help="Estima la RAM/Flash para el microcontrolador")
    parser_estimate.add_argument("--model", type=str, required=True, help="Ruta al archivo JSON del modelo")
    parser_estimate.add_argument("--flash", type=int, default=32256, help="Límite de memoria Flash (default: Arduino Uno 32KB)")
    parser_estimate.add_argument("--sram", type=int, default=2048, help="Límite de memoria SRAM (default: Arduino Uno 2KB)")
    parser_estimate.add_argument("--lang", type=str, default="C++", choices=["C", "C++", "Rust"], help="Lenguaje destino")
    parser_estimate.add_argument("--quantized", action="store_true", help="Calcula asumiendo pesos INT8")
    parser_estimate.add_argument("--input_shape", type=str, help="Forma de entrada separada por comas (ej: 1,28,28)")

    # Comando: sensor
    parser_sensor = subparsers.add_parser("sensor", help="Inicia recolección de datos desde Serial o Simulador")
    parser_sensor.add_argument("--port", type=str, default="SIMULADOR", help="Puerto COM/ttyUSB o 'SIMULADOR'")
    parser_sensor.add_argument("--baudrate", type=int, default=9600, help="Velocidad de baudios (default: 9600)")
    parser_sensor.add_argument("--label", type=str, default="clase_0", help="Etiqueta objetivo para los datos recolectados")
    parser_sensor.add_argument("--log", type=str, help="Nombre del archivo CSV para guardar el dataset")
    parser_sensor.add_argument("--verbose", action="store_true", help="Imprime el stream en la terminal en tiempo real")

    # Comando: simulate (El REPL)
    parser_simulate = subparsers.add_parser("simulate", help="Inicia un entorno REPL para probar el modelo en tiempo real")
    parser_simulate.add_argument("--model", type=str, required=True, help="Ruta al archivo JSON del modelo")

    args = parser.parse_args()

    if args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "estimate":
        cmd_estimate(args)
    elif args.command == "sensor":
        cmd_sensor(args)
    elif args.command == "simulate":
        cmd_simulate(args)


if __name__ == "__main__":
    main()