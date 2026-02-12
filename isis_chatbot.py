import os
import json
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# --- Configuration ---
LLM_DATA_ISIS_DB_DETAIL_DIR = "llm_data_isis_db_detail"
LLM_DATA_ISIS_NEIGHBORS_DIR = "llm_data_isis_neighbors"
LLM_DATA_IP_ROUTE_ISIS_DIR = "llm_data_ip_route_isis"
LLM_DATA_IP_INT_BRIEF_DIR = "llm_data_ip_int_brief"
# NUEVO: Directorio para los comandos 'show running-config router isis'
LLM_DATA_RUN_ROUTER_ISIS_DIR = "llm_data_run_router_isis" 

# Directorio para la base de datos vectorial combinada
CHROMA_DB_DIR = "chroma_db_final_rag_context_fixed_metadata" # ¡NUEVO NOMBRE para forzar reconstrucción si ya existía!

os.environ["OPENAI_API_KEY"] = "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" 

if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please set it before running the script (e.g., export OPENAI_API_KEY='sk-...').")
    exit(1)

# --- Parsing Functions (adaptadas para el formato JSON que recibirán) ---

def parse_isis_db_detail(data, device_name): 
    documents = []
    device_data = data.get(device_name) 
    if device_data and "instance" in device_data:
        instance_1_data = device_data["instance"].get("1")
        if instance_1_data and "level" in instance_1_data:
            level_2_data = instance_1_data["level"].get("2")
            if level_2_data and "lspid" in level_2_data:
                for lsp_id, lsp_data_full in level_2_data["lspid"].items():
                    lsp_header_info = lsp_data_full.get('lsp', {})
                    combined_lsp_data = {**lsp_data_full, **lsp_header_info}

                    ip_reachability = combined_lsp_data.get("extended_ipv4_reachability", {})
                    is_neighbor_reachability = combined_lsp_data.get("extended_is_neighbor", {})
                    
                    # --- CAMBIO CRÍTICO AQUÍ: Convertir listas/dicts a strings para metadatos ---
                    ip_reachability_prefixes_str = ", ".join(list(ip_reachability.keys()))
                    is_neighbor_reachability_ids_str = ", ".join(list(is_neighbor_reachability.keys()))

                    doc_content = json.dumps(combined_lsp_data, indent=2)
                    documents.append(Document(
                        page_content=f"ISIS LSP from device {device_name}, LSP ID {lsp_id}:\n{doc_content}",
                        metadata={
                            "source": f"{device_name}_isis_db_detail",
                            "type": "isis_lsp_detail", 
                            "device_name": device_name,
                            "lsp_id": lsp_id,
                            "level": lsp_header_info.get("level"), 
                            "area_address": combined_lsp_data.get("area_address"),
                            "is_attached": lsp_header_info.get("attach_bit") == 1,
                            "originating_system_id": lsp_id.split('.')[0], 
                            "router_id": combined_lsp_data.get("router_id"),
                            "hostname": combined_lsp_data.get("hostname"),
                            "ip_reachability_prefixes": ip_reachability_prefixes_str, # <-- Convertido a string
                            "is_neighbor_reachability_ids": is_neighbor_reachability_ids_str # <-- Convertido a string
                        }
                    ))
    return documents

def parse_isis_neighbors(data, device_name): 
    documents = []
    device_data = data.get(device_name) 
    if device_data and "isis" in device_data: # Acceso adaptado
        isis_instance_data = device_data["isis"].get("1") 
        if isis_instance_data and "vrf" in isis_instance_data:
            vrf_default_data = isis_instance_data["vrf"].get("default") 
            if vrf_default_data and "interfaces" in vrf_default_data:
                for if_name, if_data in vrf_default_data["interfaces"].items():
                    if "neighbors" in if_data:
                        for neighbor_system_id, neighbor_data in if_data["neighbors"].items():
                            doc_content = json.dumps(neighbor_data, indent=2)
                            documents.append(Document(
                                page_content=f"ISIS Neighbor from device {device_name}, interface {if_name}, System ID {neighbor_system_id}:\n{doc_content}",
                                metadata={
                                    "source": f"{device_name}_isis_neighbors",
                                    "type": "isis_neighbor_info", 
                                    "device_name": device_name,
                                    "interface": if_name,
                                    "neighbor_system_id": neighbor_system_id, 
                                    "state": neighbor_data.get("state"),
                                    "holdtime": neighbor_data.get("holdtime"), 
                                    "circuit_type": neighbor_data.get("type"), # Tu JSON usa 'type' aquí
                                    "ip_address": neighbor_data.get("ip_address") # Asegúrate si esta clave existe
                                }
                            ))
    return documents

def parse_ip_route_isis(data, device_name):
    documents = []
    device_data = data.get(device_name) 
    if device_data and "vrfs" in device_data:
        for vrf_name, vrf_data in device_data["vrfs"].items():
            if "address_family" in vrf_data and "ipv4" in vrf_data["address_family"]:
                if "routes" in vrf_data["address_family"]["ipv4"]:
                    for prefix, route_entries in vrf_data["address_family"]["ipv4"]["routes"].items():
                        for route_data in route_entries["route"]:
                            # --- CAMBIO CRÍTICO AQUÍ: Convertir listas/dicts a strings para metadatos ---
                            next_hops_list = route_data.get("next_hops", [])
                            next_hops_str = "; ".join([
                                f"via {nh.get('next_hop_address', 'N/A')} on {nh.get('outgoing_interface', 'N/A')}" 
                                for nh in next_hops_list
                            ])

                            doc_content = json.dumps(route_data, indent=2)
                            documents.append(Document(
                                page_content=f"ISIS IPv4 Route from device {device_name}, VRF {vrf_name}, Prefix {prefix}:\n{doc_content}",
                                metadata={
                                    "source": f"{device_name}_ip_route_isis", 
                                    "type": "isis_ipv4_route", 
                                    "device_name": device_name,
                                    "vrf": vrf_name,
                                    "prefix": prefix,
                                    "metric": route_data.get("metric"),
                                    "administrative_distance": route_data.get("administrative_distance"),
                                    "next_hops_summary": next_hops_str, # <-- Convertido a string
                                    "outgoing_interface": route_data.get("outgoing_interface") # Esto podría ser None si solo hay next_hops_list
                                }
                            ))
    return documents

def parse_ip_int_brief(data, device_name):
    documents = []
    device_data = data.get(device_name) 
    if device_data and "interface" in device_data:
        for interface_name, interface_data in device_data["interface"].items():
            doc_content = json.dumps(interface_data, indent=2)
            documents.append(Document(
                page_content=f"Interface details for device {device_name}, interface {interface_name}:\n{doc_content}",
                metadata={
                    "source": f"{device_name}_ip_int_brief",
                    "type": "device_interface_info", 
                    "device_name": device_name,
                    "interface_name": interface_name,
                    "ip_address": interface_data.get("ip_address"),
                    "status": interface_data.get("status"), 
                    "protocol_status": interface_data.get("protocol_status"), 
                    "method": interface_data.get("method")
                }
            ))
    return documents

# NUEVO: Función de parsing para 'show running-config router isis'
def parse_run_router_isis(data, device_name):
    documents = []
    device_data = data.get(device_name)
    if device_data:
        # Asumimos que 'device_data' contiene la configuración JSON completa para 'router isis'
        doc_content = json.dumps(device_data, indent=2)
        documents.append(Document(
            page_content=f"ISIS Running Configuration for device {device_name}:\n{doc_content}",
            metadata={
                "source": f"{device_name}_run_router_isis",
                "type": "isis_running_config",
                "device_name": device_name,
                # Puedes añadir más metadatos si la estructura del JSON lo permite
                # Por ejemplo, si hay un 'net_address' o 'system_id' a nivel superior.
            }
        ))
    else:
        print(f"DEBUG: No data found for device '{device_name}' in run_router_isis file. Data keys: {list(data.keys())}")
    return documents

def load_and_process_all_data(*directories):
    all_granular_documents = []
    
    parsing_functions = {
        LLM_DATA_ISIS_DB_DETAIL_DIR: parse_isis_db_detail, 
        LLM_DATA_ISIS_NEIGHBORS_DIR: parse_isis_neighbors,   
        LLM_DATA_IP_ROUTE_ISIS_DIR: parse_ip_route_isis, 
        LLM_DATA_IP_INT_BRIEF_DIR: parse_ip_int_brief,
        # NUEVO: Agrega la nueva función de parsing
        LLM_DATA_RUN_ROUTER_ISIS_DIR: parse_run_router_isis 
    }

    filename_prefix_map = {
        "isis_db_detail": LLM_DATA_ISIS_DB_DETAIL_DIR,
        "isis_neighbors": LLM_DATA_ISIS_NEIGHBORS_DIR,
        "ip_route_isis": LLM_DATA_IP_ROUTE_ISIS_DIR,
        "ip_int_brief": LLM_DATA_IP_INT_BRIEF_DIR,
        # NUEVO: Mapeo para los archivos de 'run_router_isis'
        "run_router_isis": LLM_DATA_RUN_ROUTER_ISIS_DIR 
    }

    for directory in directories:
        if not os.path.exists(directory):
            print(f"Warning: Directory '{directory}' not found. Skipping.")
            continue
            
        parser_func = parsing_functions.get(directory)
        if not parser_func:
            print(f"Warning: No parsing function defined for directory '{directory}'. Skipping.")
            continue

        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f) 
                    
                    detected_prefix = None
                    for prefix_key, dir_path in filename_prefix_map.items():
                        if dir_path == directory:
                            # Verifica si el nombre del archivo realmente comienza con el prefijo detectado
                            if filename.startswith(f"{prefix_key}_"):
                                detected_prefix = prefix_key
                                break

                    final_device_name = None
                    if detected_prefix:
                        # Extrae el nombre del dispositivo de forma más robusta
                        name_without_prefix_and_extension = filename[len(f"{detected_prefix}_"):-len(".json")]
                        final_device_name = name_without_prefix_and_extension
                    
                    if not final_device_name:
                         print(f"Warning: Could not extract device name from filename '{filename}' in directory '{directory}'. Skipping.")
                         continue

                    new_docs = parser_func(data, final_device_name)
                    all_granular_documents.extend(new_docs)
                    
                    if not new_docs: 
                        print(f"DEBUG: {parser_func.__name__} returned 0 documents for '{final_device_name}' from file: {filename}")


                    print(f"Processed {parser_func.__name__} for device '{final_device_name}' from file: {filename}")

                except json.JSONDecodeError:
                    print(f"Error: Invalid JSON format in file {filename}. Skipping.")
                except Exception as e:
                    print(f"Error loading/parsing {filename} from directory '{directory}': {e}")
    
    return all_granular_documents

def setup_rag_system():
    print(f"Loading and processing all JSON data...")
    
    parent_documents = load_and_process_all_data(
        LLM_DATA_ISIS_DB_DETAIL_DIR,
        LLM_DATA_ISIS_NEIGHBORS_DIR,
        LLM_DATA_IP_ROUTE_ISIS_DIR, 
        LLM_DATA_IP_INT_BRIEF_DIR,
        # NUEVO: Incluye el nuevo directorio en la carga de datos
        LLM_DATA_RUN_ROUTER_ISIS_DIR 
    )
    
    if not parent_documents:
        print("No valid granular documents found after processing from any directory. Exiting.")
        print("Please ensure your data generation scripts have been run and generated files in the correct directories.")
        return None

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)
    
    embeddings = OpenAIEmbeddings()
    store = InMemoryStore() 

    vectorstore = Chroma(
        collection_name="parent_document_retriever_final_rag_context_fixed_metadata_final", # ¡NUEVO NOMBRE!
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter 
    )

    current_chroma_count = vectorstore._collection.count()
    expected_min_chunks = len(parent_documents) * 2 # Estimación mínima de chunks
    
    # Comprobar si la DB existe y tiene datos suficientes, si no, reconstruir.
    # El nombre de la carpeta y el nombre de la colección se cambian para forzar una reconstrucción si ya existía.
    if not os.path.exists(CHROMA_DB_DIR) or current_chroma_count == 0 or current_chroma_count < expected_min_chunks:
        print(f"Creating/Re-indexing Chroma DB in '{CHROMA_DB_DIR}' with {len(parent_documents)} parent documents...")
        retriever.add_documents(parent_documents)
        print("Chroma DB and Parent Document Store created/updated.")
    else:
        print(f"Loading existing Chroma DB from '{CHROMA_DB_DIR}' with {current_chroma_count} existing chunks.")
        print("If data has changed or new data has been added, delete the 'chroma_db_final_rag_context_fixed_metadata' folder and rerun.") 

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0) 

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Eres un asistente experto en redes que responde preguntas basándose exclusivamente en el contexto de la base de datos IS-IS, la información de vecinos IS-IS, la tabla de rutas IS-IS, la información completa de interfaces de Cisco IOS XR y **la configuración de ejecución de IS-IS** proporcionada. Analiza y sintetiza la información de todos los documentos relevantes de todos los dispositivos para dar una respuesta completa, coherente y precisa. Si la información no está en el contexto, di que no puedes responder con la información disponible. El contexto incluye salidas estructuradas JSON de los comandos 'show isis database detail', 'show isis neighbors', 'show ip interface brief', 'show ip route isis' y 'show running-config router isis' de routers IOS XR."),
        ("human", "Contexto: {context}\n\nPregunta: {input}")
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    print("\n--- RAG system setup complete. Ready to chat! ---")
    return rag_chain

# --- Main Chatbot Loop ---
if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        print("Please set it before running the script (e.g., export OPENAI_API_KEY='sk-...').")
        exit(1)

    # Es buena práctica crear el directorio si no existe para evitar errores
    os.makedirs(LLM_DATA_RUN_ROUTER_ISIS_DIR, exist_ok=True) 

    rag_chain = setup_rag_system()

    if rag_chain:
        print("\nType your questions about ISIS routes, neighbors, interface IPs, or ISIS configuration. Type 'exit' to quit.")
        while True:
            user_query = input("\nYour Question: ")
            if user_query.lower() == 'exit':
                print("Exiting chatbot. Goodbye!")
                break
            
            if not user_query.strip():
                print("Please enter a question.")
                continue

            try:
                response = rag_chain.invoke({"input": user_query})
                print(f"\nAI Response: {response['answer']}")
                
                # Optional: Uncomment to print the sources that were used to generate the answer
                # print("\n--- Sources Used ---")
                # for doc in response['context']:
                #     print(f"- {doc.metadata.get('source', 'Unknown Source')} (Type: {doc.metadata.get('type', 'Unknown Type')}, Device: {doc.metadata.get('device_name', 'N/A')})")

            except Exception as e:
                print(f"An error occurred during query processing: {e}")
                print("Please try again or check your OpenAI API key and network connection.")
    else:
        print("RAG system could not be initialized. Please check the error messages above.")