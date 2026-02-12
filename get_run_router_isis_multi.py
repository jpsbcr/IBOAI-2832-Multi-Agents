import json
import os
from pyats.topology import loader
from parser_isis import parse_isis_config

TESTBED_FILE = "testbed.yaml"
COMMANDS_TO_TRY = [
    "show run router isis",
]

OUTPUT_DIR = "llm_data_run_router_isis"
OUTPUT_FILENAME_PREFIX = "run_router_isis"

def collect_run_router_isis_multi():
    try:
        testbed = loader.load(TESTBED_FILE)
        print(f"Testbed '{TESTBED_FILE}' cargado exitosamente.")

        processed_devices_count = 0
        for device_name, device in testbed.devices.items():
            if device.os == "iosxr":
                print(f"\n--- Procesando dispositivo: {device.name} ---")
                processed_devices_count += 1
                try:
                    device.connect(init_exec_commands=["terminal length 0"])
                    raw_output = None
                    used_command = None

                    for cmd in COMMANDS_TO_TRY:
                        try:
                            raw_output = device.execute(cmd)
                            used_command = cmd
                            if raw_output.strip():
                                break
                        except:
                            continue

                    structured_config = parse_isis_config(raw_output)
                    payload = {device_name: {"command": used_command, "parsed": structured_config}}

                    json_output = json.dumps(payload, indent=2, ensure_ascii=False)
                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    filename = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME_PREFIX}_{device.name}.json")
                    with open(filename, "w", encoding="utf-8") as f:
                        f.write(json_output)

                    print(f"Guardado: {filename}")

                except Exception as e:
                    print(f"Error procesando {device.name}: {e}")
                finally:
                    if device.is_connected():
                        device.disconnect()
            else:
                print(f"Saltando {device.name} (no es IOS XR)")

        print(f"\n--- Completado: {processed_devices_count} dispositivos procesados. ---")

    except Exception as e:
        print(f"Ocurrió un error general: {e}")

if __name__ == "__main__":
    collect_run_router_isis_multi()
