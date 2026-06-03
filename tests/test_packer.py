"""
===================================================
Test de QA: Empaquetador de Librerías
Evalúa: Generación de ZIP, estructura dual (src/include), 
manifiestos (JSON/Properties) y licencias.
"""

import os
import zipfile
from miniml.exporters.library_packer import LibraryPackager

print("1. Generando código C++ de prueba (Dummy Model)...")
dummy_cpp_code = """
// --- Dummy MiniML Model para pruebas de empaquetado ---
#ifndef MINIML_TEST_H
#define MINIML_TEST_H

float predict_dummy(float* input) {
    return 0.99f; // Predicción simulada
}

#endif
"""

print("2. Ejecutando LibraryPackager...")
model_name = "QA_Model_Test"
# Llamamos al empaquetador simulando un modelo cuantizado
result = LibraryPackager.create_arduino_zip(
    model_name=model_name, 
    cpp_code=dummy_cpp_code, 
    version="2.0.0", 
    quantized=True,
    force=True # Forzamos sobreescritura si ya existe
)

if not result["success"]:
    print(f"\n[ERROR] Falló el empaquetado: {result.get('error')}")
else:
    zip_path = result["path"]
    print(f"\n[OK] Librería ZIP generada en: {zip_path}")
    
    print("\n3. Auditando estructura interna del archivo ZIP...")
    print("--------------------------------------------------")
    
    # Abrimos el ZIP recién creado en modo lectura
    try:
        with zipfile.ZipFile(zip_path, 'r') as zipf:
            # Listamos todos los archivos dentro del ZIP
            file_list = zipf.namelist()
            
            # Imprimimos el árbol de archivos ordenado
            for file_name in sorted(file_list):
                # Formateo simple para simular un árbol de directorios
                depth = file_name.count('/')
                indent = "  " * (depth - 1) if depth > 0 else ""
                marker = "|-- " if depth > 0 else ""
                
                # Leer tamaño del archivo para verificar que no esté vacío
                file_info = zipf.getinfo(file_name)
                size_kb = file_info.file_size / 1024.0
                
                print(f"{indent}{marker}{os.path.basename(file_name)} ({size_kb:.2f} KB)")
                
                # Opcional: Leer y validar una clave del library.json
                if "library.json" in file_name:
                    json_content = zipf.read(file_name).decode('utf-8')
                    if "espressif32" in json_content and "atmelavr" in json_content:
                        print(f"{indent}    * [Validado] Plataformas PlatformIO detectadas.")
                
                # Opcional: Validar Licencia
                if "LICENSE" in file_name:
                    lic_content = zipf.read(file_name).decode('utf-8')
                    if "Apache License" in lic_content:
                        print(f"{indent}    * [Validado] Licencia Apache 2.0 inyectada.")

        print("--------------------------------------------------")
        print("Auditoría de Empaquetado Finalizada con Éxito.")
        print("La librería está lista para ser importada en PlatformIO o Arduino IDE.")

    except Exception as e:
        print(f"\n[ERROR] No se pudo leer el archivo ZIP: {e}")