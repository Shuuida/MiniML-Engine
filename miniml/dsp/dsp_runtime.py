"""
DSP Runtime for EduBot
===========================
Motor de Procesamiento Digital de Señales (DSP) Nativo.
Implementación "Zero-Dependency" optimizada para EduBot.

Algoritmos:
- FFT: Cooley-Tukey Recursivo (O(N log N)).
- Filtros: IIR Biquad (Butterworth 2do Orden).
- Ventanas: Hanning / Hamming nativas.
"""

import math
import cmath

class DSPRuntime:

    #Transformada Rápida de Fourier (FFT) - Cooley-Tukey

    @staticmethod
    def _fft_recursive(x):
        """
        Núcleo recursivo de la FFT.
        Entrada: Lista de números complejos.
        Salida: Lista de números complejos transformados.
        """
        N = len(x)
        if N <= 1:
            return x
        
        # División (Divide & Conquer)
        even = DSPRuntime._fft_recursive(x[0::2])
        odd = DSPRuntime._fft_recursive(x[1::2])
        
        # Combinación
        combined = [0] * N
        for k in range(N // 2):
            t = cmath.exp(-2j * cmath.pi * k / N) * odd[k]
            combined[k] = even[k] + t
            combined[k + N // 2] = even[k] - t
            
        return combined

    @staticmethod
    def _next_power_of_2(x):
        """Padding: Ajusta el tamaño al siguiente exponente de 2."""
        return 1 if x == 0 else 2**(x - 1).bit_length()

    @staticmethod
    def compute_fft(data_window, sample_rate=100.0):
        """
        Calcula la FFT y devuelve el espectro de frecuencias.
        Optimizado para Python puro.
        """
        if not data_window:
            return [], []

        N = len(data_window)
        # Padding a potencia de 2 para velocidad máxima
        N_pad = DSPRuntime._next_power_of_2(N)
        
        # Aplicar Ventana Hanning (Reduce spectral leakage)
        # w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))
        windowed_data = []
        if N > 1:
            for i, val in enumerate(data_window):
                win_val = 0.5 * (1 - math.cos(2 * math.pi * i / (N - 1)))
                windowed_data.append(val * win_val)
        else:
            windowed_data = data_window

        # Rellenar con ceros si es necesario (Zero-padding)
        windowed_data.extend([0.0] * (N_pad - N))
        
        # Ejecutar FFT
        complex_spectrum = DSPRuntime._fft_recursive(windowed_data)
        
        # Calcular Magnitudes (Solo mitad positiva del espectro)
        # Frecuencia de Nyquist = SampleRate / 2
        half_n = N_pad // 2
        mags = []
        freqs = []
        
        # Factor de escala para normalizar amplitud
        scale = 2.0 / N if N > 0 else 1.0

        for i in range(half_n):
            # Magnitud = sqrt(re^2 + im^2)
            mag = abs(complex_spectrum[i]) * scale
            mags.append(mag)
            
            # Eje de Frecuencia: f = i * Fs / N
            freqs.append(i * sample_rate / N_pad)
            
        return freqs, mags

    # Filtros Digitales (IIR Biquad - Direct Form II)

    @staticmethod
    def _calculate_coefficients(filter_type, cutoff, fs):
        """
        Calcula coeficientes a0, a1, a2, b0, b1, b2 para un filtro
        Butterworth de 2do orden usando Transformada Bilineal.
        """
        # Frecuencia normalizada
        w0 = 2 * math.pi * cutoff / fs
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        
        # Q factor para Butterworth es 1/sqrt(2) approx 0.707
        alpha = sin_w0 / (2 * 0.7071)

        b0 = b1 = b2 = a0 = a1 = a2 = 0.0

        if filter_type == 'lowpass':
            b0 = (1 - cos_w0) / 2
            b1 = 1 - cos_w0
            b2 = (1 - cos_w0) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
        elif filter_type == 'highpass':
            b0 = (1 + cos_w0) / 2
            b1 = -(1 + cos_w0)
            b2 = (1 + cos_w0) / 2
            a0 = 1 + alpha
            a1 = -2 * cos_w0
            a2 = 1 - alpha
        
        # Normalizar por a0 para la ecuación de diferencias
        return (b0/a0, b1/a0, b2/a0, a1/a0, a2/a0)

    @staticmethod
    def apply_filter(data, filter_type='lowpass', cutoff=5.0, sample_rate=100.0):
        """
        Aplica un filtro IIR Biquad en tiempo real.
        Sin dependencias.
        """
        if not data or len(data) < 2:
            return data
        
        # Protección contra frecuencias imposibles (Nyquist)
        if cutoff >= sample_rate / 2:
            return data # Pass-through

        # Calcular coeficientes
        b0, b1, b2, a1, a2 = DSPRuntime._calculate_coefficients(filter_type, cutoff, sample_rate)

        # Aplicar Ecuación de Diferencias:
        # y[n] = b0*x[n] + b1*x[n-1] + b2*x[n-2] - a1*y[n-1] - a2*y[n-2]
        
        y = [0.0] * len(data)
        x = data
        
        # Inicialización (asumimos estado cero)
        y[0] = b0 * x[0]
        y[1] = b0 * x[1] + b1 * x[0] - a1 * y[0]

        for n in range(2, len(data)):
            y[n] = (b0 * x[n]) + (b1 * x[n-1]) + (b2 * x[n-2]) - (a1 * y[n-1]) - (a2 * y[n-2])

        return y

    @staticmethod
    def sliding_window(buffer, window_size, overlap):
        """Generador de ventanas para procesamiento por lotes."""
        step = max(1, window_size - overlap)
        for i in range(0, len(buffer) - window_size + 1, step):
            yield buffer[i:i + window_size]

    @staticmethod
    def compute_mel_spectrogram(signal: list, sample_rate: float, n_mels: int = 40, n_fft: int = 256):
        """
        Genera una 'imagen' de sonido (Espectrograma Mel) lista para CNNs 2D.
        Simula la cóclea humana (Escala Mel).
        
        Args:
            signal: Lista de floats (audio raw).
            sample_rate: Frecuencia de muestreo (ej: 16000 Hz).
            n_mels: Cantidad de bancos de filtros (altura de la imagen resultante).
            n_fft: Tamaño de la ventana de FFT.
        
        Returns:
            mel_spec: Lista de listas (Matriz 2D) [Tiempo x Mels].
        """
        # Pre-énfasis (Filtro paso alto simple para resaltar agudos)
        emphasized_signal = []
        if len(signal) > 0:
            emphasized_signal.append(signal[0])
            for i in range(1, len(signal)):
                emphasized_signal.append(signal[i] - 0.97 * signal[i-1])
        else:
            return []

        # Framing (Cortar en ventanas solapadas)
        frame_size = n_fft
        hop_size = frame_size // 2
        
        frames = []
        for i in range(0, len(emphasized_signal) - frame_size, hop_size):
            frames.append(emphasized_signal[i : i + frame_size])

        # Procesar cada frame (FFT -> Power -> Mel)
        mel_spectrogram = []
        
        # Pre-calcular bancos Mel (aproximación lineal simple para TinyML)
        # Nota: En producción usaríamos una matriz de filtros triangular real.
        # Aquí mapeamos bins de FFT a bins Mel por promedio simple para velocidad.
        fft_bins = n_fft // 2
        mel_bin_width = fft_bins // n_mels
        
        for frame in frames:
            # Reutilizamos la FFT existente
            freqs, mags = DSPRuntime.compute_fft(frame, sample_rate)
            
            # Solo nos interesa la magnitud (Power Spectrum)
            # Mapeo a Mel (Downsampling de frecuencia)
            mel_row = []
            for m in range(n_mels):
                start_idx = m * mel_bin_width
                end_idx = start_idx + mel_bin_width
                
                # Promedio de energía en esta banda
                if end_idx <= len(mags):
                    avg_energy = sum(mags[start_idx:end_idx]) / max(1, (end_idx - start_idx))
                else:
                    avg_energy = 0.0
                
                # Log-Energy (Estabilidad numérica + percepción humana)
                # log(x + epsilon)
                mel_val = math.log(avg_energy + 1e-9)
                mel_row.append(mel_val)
            
            mel_spectrogram.append(mel_row)
            
        return mel_spectrogram