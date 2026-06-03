"""
Virtual Sensor Mock para EduBot
==================================
Simula un hardware físico (Arduino/ESP32) usando Duck Typing
para engañar al SerialManager y hacerse pasar por pySerial.
Ahora con soporte consciente del contexto (ondas dinámicas).
"""

import time
import math
import random

class VirtualSerial:
    def __init__(self):
        self.is_open = True
        self.port = "SIMULADOR"
        self.start_time = time.time()
        self.current_label = "Arriba" # Etiqueta por defecto
        
    def set_label(self, label: str):
        """Permite al manager inyectar la etiqueta actual de React."""
        self.current_label = str(label).strip().lower()

    @property
    def in_waiting(self):
        # Engañamos al manager diciéndole que siempre hay 1 byte esperando
        return 1 

    def readline(self):
        # Simulamos el retardo físico de un microcontrolador (aprox 10 FPS)
        time.sleep(0.1) 
        
        t = time.time() - self.start_time
        
        # Bifurcación de ondas según la etiqueta actual
        if self.current_label == "abajo":
            # ONDA CUADRADA: Saltos bruscos entre 10 y -10
            val1 = 10.0 if math.sin(t * 3.0) > 0 else -10.0
            val2 = 10.0 if math.cos(t * 3.0) > 0 else -10.0
        else:
            # ONDA SENOIDAL: Curva suave clásica (usada para "Arriba" u otros)
            val1 = math.sin(t * 3.0) * 10.0
            val2 = math.cos(t * 3.0) * 10.0
            
        # Añadimos un poco de ruido analógico para simular la vida real
        val1 += random.uniform(-0.5, 0.5)
        val2 += random.uniform(-0.5, 0.5)
        
        simulated_line = f"{val1:.4f},{val2:.4f}\n"
        return simulated_line.encode('utf-8')
        
    def reset_input_buffer(self):
        pass
        
    def close(self):
        self.is_open = False