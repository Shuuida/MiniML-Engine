"""
Serial Manager para MiniML Engine (CLI & Headless Edition)
==========================================================
Gestor asíncrono para recolección de datos de sensores y MLOps.
- Soporta Buffer Circular con Timestamps para inferencia programática.
- Streaming opcional directo a Terminal stdout.
"""

import threading
import queue
import time
import csv
import os
import sys
from miniml.hardware.virtual_sensor import VirtualSerial

# Manejo de dependencia opcional (pyserial)
try:
    import serial
    import serial.tools.list_ports
    _HAS_SERIAL = True
except ImportError:
    _HAS_SERIAL = False
    print("[WARN] 'pyserial' no instalado. El módulo de Hardware real estará desactivado.")

class SerialManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SerialManager, cls).__new__(cls)
            cls._instance._init_manager()
        return cls._instance

    def _init_manager(self):
        self.serial_port = None
        self.is_connected = False
        self.is_reading = False
        self.read_thread = None
        
        # Buffer circular seguro para hilos (Vital para scripts de Inferencia)
        self.data_queue = queue.Queue(maxsize=1000) 
        self.last_error = None

        # DATA LOGGER VARS
        self.is_logging = False
        self.log_file = None
        self.csv_writer = None
        self.log_path = ""
        self.verbose_stream = False

    def list_ports(self):
        """Lista los puertos COM disponibles (Útil para auto-descubrimiento en scripts)."""
        result = []
        if _HAS_SERIAL:
            ports = serial.tools.list_ports.comports()
            for p in ports:
                result.append({
                    "device": p.device,
                    "description": p.description,
                    "hwid": p.hwid
                })

        result.append({
            "device": "SIMULADOR",
            "description": "Sensor Virtual (CLI Automation Mode)",
            "hwid": "VIRTUAL_001"
        })
        return result

    def connect(self, port_name, baudrate=9600, verbose=False):
        """Abre la conexión y arranca el hilo de lectura."""
        if self.is_connected:
            self.disconnect()

        self.verbose_stream = verbose
        try:
            if port_name == "SIMULADOR":
                self.serial_port = VirtualSerial()
            else:
                if not _HAS_SERIAL:
                    raise RuntimeError("Librería pyserial no instalada.")
                self.serial_port = serial.Serial(port=port_name, baudrate=int(baudrate), timeout=1)

            self.is_connected = True
            self.is_reading = True
            self.last_error = None
            
            self.serial_port.reset_input_buffer()
            # Limpiar cola residual de conexiones anteriores
            while not self.data_queue.empty():
                try: self.data_queue.get_nowait()
                except queue.Empty: break

            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.read_thread.start()
            
            print(f"[SerialManager] Conectado exitosamente a {port_name}.")
            return True
        except Exception as e:
            self.last_error = str(e)
            self.is_connected = False
            print(f"[SerialManager] Error al conectar: {e}")
            return False

    def disconnect(self):
        """Detiene todo: lectura, grabación y puerto."""
        if self.is_logging:
            self.stop_logging()

        self.is_reading = False
        if self.read_thread:
            self.read_thread.join(timeout=2.0)
        
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        
        self.is_connected = False
        self.serial_port = None
        print("[SerialManager] Puerto desconectado.")

    def start_logging(self, filename="capture", label="Clase_0"):
        """Inicia la grabación de datos a un CSV listo para ML (Modo Acumulativo CLI)."""
        if not self.is_connected:
            return {"success": False, "error": "No hay conexión serial activa"}
        
        if self.is_logging:
            return {"success": False, "error": "Ya se está grabando"}

        try:
            datasets_dir = os.path.join(os.getcwd(), "datasets")
            os.makedirs(datasets_dir, exist_ok=True)
            
            safe_name = f"{filename}.csv"
            self.log_path = os.path.join(datasets_dir, safe_name)
            
            file_exists = os.path.exists(self.log_path)
            
            self.log_file = open(self.log_path, mode='a', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.log_file)
            
            self.current_log_label = label
            self.is_logging = True

            if hasattr(self.serial_port, 'set_label'):
                self.serial_port.set_label(label)
            
            self._headers_written = file_exists
            
            print(f"[SerialManager] Grabando datos en {self.log_path} con etiqueta '{label}'")
            return {"success": True, "path": self.log_path}
            
        except Exception as e:
            self.is_logging = False
            if self.log_file: self.log_file.close()
            return {"success": False, "error": str(e)}

    def stop_logging(self):
        """Detiene la grabación y cierra el archivo."""
        if not self.is_logging:
            return {"success": False, "error": "No hay grabación activa"}
        
        self.is_logging = False
        try:
            if self.log_file:
                self.log_file.flush()
                self.log_file.close()
                self.log_file = None
                self.csv_writer = None
            
            print("[SerialManager] Grabación finalizada.")
            return {"success": True, "path": self.log_path}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _read_loop(self):
        """Bucle infinito: Lee Serial -> Buffer Circular -> CLI -> CSV."""
        while self.is_reading and self.serial_port and self.serial_port.is_open:
            try:
                if self.serial_port.in_waiting > 0:
                    line = self.serial_port.readline()
                    try:
                        decoded = line.decode('utf-8').strip()
                        if decoded:
                            parts = [float(x) for x in decoded.split(',')]
                            timestamp = time.time()
                            
                            # Alimentar Buffer Circular Programático
                            if self.data_queue.full():
                                try: self.data_queue.get_nowait() # Elimina el más viejo
                                except queue.Empty: pass
                            # Formato de diccionario con timestamp
                            self.data_queue.put({"ts": timestamp, "data": parts})

                            # Imprimir en consola si el modo verboso está activo
                            if self.verbose_stream:
                                sys.stdout.write(f"\r[STREAM] {timestamp:.2f} -> {parts}    ")
                                sys.stdout.flush()

                            # Guardar en disco (Formato ML-Ready)
                            if self.is_logging and self.csv_writer:
                                if not hasattr(self, '_headers_written') or not self._headers_written:
                                    headers = [f"Sensor_{i}" for i in range(len(parts))] + ["Target"]
                                    self.csv_writer.writerow(headers)
                                    self._headers_written = True
                                
                                row = parts + [self.current_log_label]
                                self.csv_writer.writerow(row)
                            
                    except ValueError:
                        pass # Ignorar líneas basura (garbage lines)
            except Exception as e:
                print(f"\n[SerialManager] Error en bucle de lectura: {e}")
                self.last_error = str(e)
                time.sleep(0.1)

    def get_buffer(self, max_items=50):
        """Consume datos del buffer de forma programática para scripts de inferencia."""
        results = []
        count = 0
        while not self.data_queue.empty() and count < max_items:
            results.append(self.data_queue.get())
            count += 1
        return results

serial_manager = SerialManager()