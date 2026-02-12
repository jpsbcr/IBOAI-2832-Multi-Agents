# get_ip_int_brief_multi.py
import json
import os
from pyats.topology import loader

# --- Configuration ---
TESTBED_FILE = "testbed.yaml"
COMMAND_TO_RUN = "show ip interface brief"

# --- Output File Configuration ---
OUTPUT_DIR = "llm_data_ip_int_brief"
OUTPUT_FILENAME_PREFIX = "ip_int_brief"

# --- Script Logic ---
def collect_ip_int_brief_multi():
    try:
        testbed = loader.load(TESTBED_FILE)
        print(f"Testbed '{TESTBED_FILE}' cargado exitosamente.")

        processed_devices_count = 0
        for device_name, device in testbed.devices.items():
            if device.os == 'iosxr':
                print(f"\n--- Procesando dispositivo: {device.name} ({device.connections.cli.ip}) ---")
                processed_devices_count += 1
                try:
                    print(f"Conectando a {device.name}...")
                    device.connect(init_exec_commands=['terminal length 0'])
                    print(f"Conectado a {device.name}.")

                    print(f"Ejecutando '{COMMAND_TO_RUN}' y parseando la salida con Genie...")
                    parsed_output = device.parse(COMMAND_TO_RUN) # <-- Usa device.parse()
                    
                    # ¡CAMBIO CRÍTICO AQUÍ!
                    # Envolver la salida de Genie con la clave del nombre del dispositivo
                    final_output_for_json = {device_name: parsed_output} # <-- AÑADIDO ESTO
                    json_output = json.dumps(final_output_for_json, indent=2) # <-- Usa final_output_for_json

                    print("\n--- Vista Previa de Salida JSON (primeros 500 caracteres) ---")
                    print(json_output[:500] + "..." if len(json_output) > 500 else json_output)

                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    filename = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME_PREFIX}_{device.name}.json")

                    with open(filename, 'w') as f:
                        f.write(json_output)
                    print(f"\n--- Salida JSON guardada en: {filename} ---")

                except Exception as e:
                    print(f"Error procesando {device.name}: {e}")
                finally:
                    if device.is_connected():
                        print(f"Desconectando de {device.name}...")
                        device.disconnect()
                        print("Desconectado.")
            else:
                print(f"\nSaltando dispositivo '{device.name}' (OS: {device.os}) ya que no es un dispositivo IOS XR.")

        if processed_devices_count == 0:
            print("\nNo se encontraron dispositivos IOS XR en el testbed para procesar.")
        else:
            print(f"\n--- Proceso completado. {processed_devices_count} dispositivos IOS XR procesados. ---")

    except Exception as e:
        print(f"Ocurrió un error al cargar el testbed o en el proceso general: {e}")

if __name__ == "__main__":
    collect_ip_int_brief_multi()