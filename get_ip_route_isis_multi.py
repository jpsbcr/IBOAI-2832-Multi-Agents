# get_ip_route_isis_multi.py
import json
import os
import re 
from pyats.topology import loader

# --- Configuration ---
TESTBED_FILE = "testbed.yaml"
COMMAND_TO_RUN = "show ip route isis"

# --- Output File Configuration ---
OUTPUT_DIR = "llm_data_ip_route_isis"
OUTPUT_FILENAME_PREFIX = "ip_route_isis"

# --- Script Logic ---
def collect_ip_route_isis_multi():
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

                    print(f"Ejecutando '{COMMAND_TO_RUN}' y construyendo JSON jerárquico...")
                    
                    raw_output = device.execute(COMMAND_TO_RUN) 
                    
                    # --- Lógica de parsing para construir el JSON jerárquico ---
                    parsed_routes = {
                        "vrfs": {
                            "default": { # Suponemos VRF default si no se especifica
                                "address_family": {
                                    "ipv4": {
                                        "routes": {}
                                    }
                                }
                            }
                        }
                    }

                    # Regex para la línea principal de la ruta (prefijo, AD/Métrica)
                    # Ejemplo: i L2 2.2.2.2/32 [115/30]
                    main_route_pattern = re.compile(
                        r"^(?P<protocol_code>\S+)\s+(?P<protocol_type>\S+)\s+" # e.g., "i L2"
                        r"(?P<prefix>\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}/\d{1,2})\s+" 
                        r"\[(?P<ad>\d+)/(?P<metric>\d+)\]" 
                    )
                    
                    # Regex para las líneas 'via' (sangradas, pueden ser múltiples)
                    # Ejemplo:                 [115/30] via 10.1.2.2, 01:48:00, GigabitEthernet0/0/0/1
                    via_pattern = re.compile(
                        r"^\s+(?:\[(?P<ad_via>\d+)/(?P<metric_via>\d+)\]\s+via\s+)?" # Optional AD/Metric if it's not on the first line
                        r"(?P<next_hop>\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})?,\s*" # Next-hop IP
                        r"(?P<uptime_via>\d{2}:\d{2}:\d{2}|\S+),\s*" # Uptime (can be "never" or an actual time)
                        r"(?P<interface>\S+)$" # Outgoing interface
                    )

                    current_prefix = None
                    for line in raw_output.splitlines():
                        line = line.strip()
                        if not line: continue # Skip empty lines

                        # Intentar coincidir con la línea principal de la ruta
                        main_match = main_route_pattern.match(line)
                        if main_match:
                            route_info = main_match.groupdict()
                            current_prefix = route_info['prefix']
                            
                            # Inicializar la entrada para este prefijo si es nueva
                            if current_prefix not in parsed_routes["vrfs"]["default"]["address_family"]["ipv4"]["routes"]:
                                parsed_routes["vrfs"]["default"]["address_family"]["ipv4"]["routes"][current_prefix] = {"route": []}
                            
                            # No añadimos la ruta aquí todavía, esperamos las 'via' líneas
                            # Guardamos la info principal para combinarla con las vías
                            current_route_entry_template = {
                                "protocol": "isis", # Basado en tu pregunta, puedes refinar esto
                                "protocol_code": route_info.get("protocol_code"), # e.g. "i"
                                "protocol_type": route_info.get("protocol_type"), # e.g. "L2"
                                "administrative_distance": int(route_info['ad']) if route_info['ad'].isdigit() else route_info['ad'],
                                "metric": int(route_info['metric']) if route_info['metric'].isdigit() else route_info['metric'],
                                "next_hops": [] # Lista para múltiples vías
                            }
                            # Añadir esta plantilla de ruta a una lista temporal para el prefijo actual
                            parsed_routes["vrfs"]["default"]["address_family"]["ipv4"]["routes"][current_prefix]["route"].append(current_route_entry_template)
                            
                        else:
                            # Si no es una línea principal, intentar coincidir con una línea 'via' (sangrada)
                            via_match = via_pattern.match(line)
                            if via_match and current_prefix: # Solo si estamos procesando un prefijo
                                via_info = via_match.groupdict()
                                
                                # Obtener la última entrada de ruta para el prefijo actual
                                if parsed_routes["vrfs"]["default"]["address_family"]["ipv4"]["routes"][current_prefix]["route"]:
                                    latest_route_entry = parsed_routes["vrfs"]["default"]["address_family"]["ipv4"]["routes"][current_prefix]["route"][-1]
                                    
                                    # Crear el diccionario para el next-hop
                                    next_hop_entry = {
                                        "next_hop_address": via_info.get("next_hop"),
                                        "outgoing_interface": via_info.get("interface"),
                                        "uptime": via_info.get("uptime")
                                    }
                                    
                                    # Si la línea 'via' también tiene AD/Metric, usar esos valores para esta vía específica
                                    if via_info.get("ad_via") and via_info.get("metric_via"):
                                        next_hop_entry["administrative_distance"] = int(via_info['ad_via']) if via_info['ad_via'].isdigit() else via_info['ad_via']
                                        next_hop_entry["metric"] = int(via_info['metric_via']) if via_info['metric_via'].isdigit() else via_info['metric_via']

                                    latest_route_entry["next_hops"].append(next_hop_entry)
                                else:
                                    print(f"Warning: 'via' line found without a preceding main route line for {current_prefix} on {device_name}: {line}")
                            else:
                                # Líneas que no son rutas ni líneas 'via' (ej., encabezados, saltos de línea)
                                pass # Ignorar estas líneas o manejarlas si es necesario
                    # --- Fin de la lógica de parsing ---

                    # El output final para el JSON incluirá el nombre del dispositivo como clave superior
                    final_output_for_json = {device_name: parsed_routes}

                    json_output = json.dumps(final_output_for_json, indent=2)

                    print("\n--- Vista Previa de Salida JSON (primeros 500 caracteres) ---")
                    print(json_output[:500] + "..." if len(json_output) > 500 else json_output)

                    os.makedirs(OUTPUT_DIR, exist_ok=True)
                    filename = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILENAME_PREFIX}_{device.name}.json")

                    with open(filename, 'w') as f:
                        f.write(json_output)
                    print(f"\n--- Salida JSON guardada en: {filename} ---")

                except Exception as e:
                    print(f"Error procesando {device.name}: {e}")
                    # DEBUG: Imprimir el raw_output completo si hay un error para facilitar la depuración
                    # print(f"\n--- Full RAW Output for {device.name} (Error Debug) ---")
                    # print(raw_output)
                    # print("--- End Full RAW Output ---")
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
    collect_ip_route_isis_multi()