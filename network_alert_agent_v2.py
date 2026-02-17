# network_alert_agent_v2.py — Agent Alert (ISIS parsing determinista, Pydantic v2-safe)

import os
import json
import re
import socket
from datetime import datetime
from typing import Optional, List, Tuple

from pydantic import BaseModel, Field

# ====== Config ======
SYSLOG_HOST = "0.0.0.0"
SYSLOG_PORT = 514  # Si tienes problema de permisos, usa 5514
OUTBOX_DIR = "alerts_outbox"  # Fallback si no está disponible el troubleshooter
os.environ["OPENAI_API_KEY"] = "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" 


# ====== Fallback LLM (opcional) ======
USE_LLM_FALLBACK = True
try:
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.messages import SystemMessage, HumanMessage
except Exception:
    USE_LLM_FALLBACK = False

if USE_LLM_FALLBACK and not os.getenv("OPENAI_API_KEY"):
    print("INFO: OPENAI_API_KEY no está definido; se desactiva el fallback LLM.")
    USE_LLM_FALLBACK = False

# ====== Modelos (Pydantic v2) ======
class DeviceCommandPlan(BaseModel):
    device_name: str = Field(..., description="Dispositivo destino para los comandos.")
    interfaces: List[str] = Field(default_factory=list, description="Interfaces relacionadas en el dispositivo.")
    commands: List[str] = Field(default_factory=list, description="Comandos 'show' a verificar.")

class AlertReport(BaseModel):
    raw_log: str = Field(..., description="Syslog original (línea completa)")
    timestamp_utc: str = Field(..., description="Momento de recepción (UTC)")
    technology: Optional[str] = Field(None, description="Tecnología: ISIS, OSPF, BGP, Interface, etc.")
    severity: str = Field(..., description="CRITICAL, MAJOR, MINOR, INFORMATIONAL o DEBUG")
    summary: str = Field(..., description="Resumen en una línea")
    # Campos solicitados:
    local_device_name: Optional[str] = Field(None, description="Hostname del emisor (si se puede extraer)")
    source_ip: Optional[str] = Field(None, description="IP origen del paquete UDP")
    source_port: Optional[int] = Field(None, description="Puerto origen del paquete UDP")
    neighbor: Optional[str] = Field(None, description="Vecino implicado en el log (si aplica)")
    interface: Optional[str] = Field(None, description="Interfaz local implicada (si aplica)")
    # Extras útiles:
    potential_causes: List[str] = Field(default_factory=list, description="Causas potenciales")
    devices: List[str] = Field(default_factory=list, description="Lista de dispositivos implicados (local + vecino)")
    next_steps: List[DeviceCommandPlan] = Field(default_factory=list, description="Siguientes pasos por dispositivo")

# ====== Regex y helpers de parsing determinista ======
# Ejemplos:
# %ROUTING-ISIS-5-ADJCHANGE : Adjacency to P-1 (GigabitEthernet0/0/0/0) (L2) Up, New adjacency
# %ROUTING-ISIS-5-ADJCHANGE : Adjacency to P-1 (GigabitEthernet0/0/0/0) (L2) Down, Interface state down
ISIS_ADJCHANGE_RE = re.compile(
    r"Adjacency to\s+(?P<neighbor>[A-Za-z0-9._/-]+)\s+\((?P<iface>[A-Za-z]+[A-Za-z0-9/\.]+)\)\s+\(L\d\)\s+(?P<state>Up|Down)",
    re.IGNORECASE
)

# Hostname cuando aparece antes del RP/...:
# "<189>382: PE-1 RP/0/0/CPU0:Aug ..."  -> captura "PE-1"
HOSTNAME_BEFORE_LOCATION_RE = re.compile(
    r"^\s*(?:<\d+>\d+:\s*)?(?P<host>[A-Za-z0-9._/-]+)\s+RP/\d+/\d+/CPU\d+:"
)

# Token inicial antes de ":" (cuando sólo aparece la localización, p.ej. "RP/0/0/CPU0")
HOST_PREFIX_RE = re.compile(r"^\s*(?:<\d+>\d+:\s*)?(?P<host>[A-Za-z0-9._/-]+):")

def extract_local_device_name(raw_log: str, source_ip: Optional[str]) -> Optional[str]:
    """
    Orden de detección:
      1) Hostname explícito antes de 'RP/...'
      2) Token inicial antes del primer ':'
      3) Reverse DNS del source_ip (o devolver la IP si no hay PTR)
    """
    m1 = HOSTNAME_BEFORE_LOCATION_RE.search(raw_log)
    if m1:
        return m1.group("host")

    m2 = HOST_PREFIX_RE.search(raw_log)
    if m2:
        return m2.group("host")

    if source_ip:
        try:
            return socket.gethostbyaddr(source_ip)[0]
        except Exception:
            return source_ip

    return None

def try_parse_isis_adjchange(raw_log: str,
                             source: Optional[Tuple[str, int]]) -> Optional[AlertReport]:
    m = ISIS_ADJCHANGE_RE.search(raw_log)
    if not m:
        return None

    src_ip, src_port = (source if source else (None, None))
    local_host = extract_local_device_name(raw_log, src_ip)

    neighbor = m.group("neighbor")
    iface = m.group("iface")
    state = m.group("state").capitalize()

    if state == "Up":
        severity = "INFORMATIONAL"
        summary = "ISIS session UP — adjacency established; no action required."
        next_cmds = ["show isis neighbors"]   # opcional, verificación ligera
        causes = []
    else:
        severity = "CRITICAL"
        summary = "ISIS adjacency DOWN — service impact possible; immediate checks required."
        next_cmds = [
            "show isis neighbors",
            f"show interface {iface} brief",
            f"show controllers {iface}",
            "show log | include ISIS",
            "show isis adjacency detail"
        ]
        causes = [
            "Physical link failure",
            "Neighbor or local device reload",
            "Interface admin down or err-disabled",
            "MTU/authentication/config mismatch"
        ]

    next_steps: List[DeviceCommandPlan] = []
    if local_host:
        # Para UP dejamos sólo verificación mínima; para DOWN, la lista completa
        cmds = ["show isis neighbors"] if state == "Up" else next_cmds
        next_steps.append(DeviceCommandPlan(
            device_name=local_host,
            interfaces=[iface],
            commands=cmds
        ))

    return AlertReport(
        raw_log=raw_log,
        timestamp_utc=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        technology="ISIS",
        severity=severity,
        summary=summary,
        local_device_name=local_host,
        source_ip=src_ip,
        source_port=src_port,
        neighbor=neighbor,
        interface=iface,
        potential_causes=causes,
        devices=[d for d in [local_host, neighbor] if d],
        next_steps=next_steps
    )

# ====== Fallback LLM (opcional) para formatos desconocidos ======
if USE_LLM_FALLBACK:
    llm_alert = ChatOpenAI(model="gpt-4o", temperature=0.1)
    alert_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "Eres un agente NOC experto. Recibes UNA línea de syslog y debes:\n"
            "1) Identificar TECNOLOGÍA (ISIS, OSPF, BGP, Interface/Link, Platform, Auth/AAA).\n"
            "2) Extraer dispositivo local (prefijo), vecino(s) e interfaz(es).\n"
            "3) Asignar severidad: CRITICAL, MAJOR, MINOR, INFORMATIONAL, DEBUG.\n"
            "4) Resumen conciso.\n"
            "5) Siguientes pasos como comandos 'show' en el dispositivo correcto.\n"
            "Devuelve JSON con: technology, severity, summary, devices, neighbor, interfaces, next_steps."
        )),
        HumanMessage(content="Raw syslog: {raw_log}")
    ])

def llm_parse_generic(raw_log: str,
                      source: Optional[Tuple[str, int]]) -> AlertReport:
    local_host = extract_local_device_name(raw_log, source[0] if source else None)
    src_ip, src_port = (source if source else (None, None))
    devices_guess: List[str] = [local_host] if local_host else []

    if not USE_LLM_FALLBACK:
        return AlertReport(
            raw_log=raw_log,
            timestamp_utc=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            technology=None,
            severity="INFORMATIONAL",
            summary="Unrecognized format; recorded for reference.",
            local_device_name=local_host,
            source_ip=src_ip,
            source_port=src_port,
            neighbor=None,
            interface=None,
            potential_causes=[],
            devices=devices_guess,
            next_steps=[]
        )

    try:
        resp = llm_alert.invoke(alert_prompt.format_messages(raw_log=raw_log))
        data = json.loads(resp.content)
    except Exception:
        data = {
            "technology": None,
            "severity": "INFORMATIONAL",
            "summary": "Could not parse; informational fallback.",
            "devices": devices_guess,
            "neighbor": None,
            "interfaces": [],
            "next_steps": []
        }

    devices = data.get("devices") or devices_guess
    neighbor = data.get("neighbor")
    if neighbor and neighbor not in devices:
        devices.append(neighbor)

    interfaces = data.get("interfaces") or []
    next_cmds = data.get("next_steps") or []
    next_steps: List[DeviceCommandPlan] = []
    if local_host and next_cmds:
        next_steps.append(DeviceCommandPlan(
            device_name=local_host,
            interfaces=interfaces,
            commands=next_cmds
        ))

    return AlertReport(
        raw_log=raw_log,
        timestamp_utc=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        technology=data.get("technology"),
        severity=(data.get("severity") or "INFORMATIONAL").upper(),
        summary=data.get("summary") or "",
        local_device_name=local_host,
        source_ip=src_ip,
        source_port=src_port,
        neighbor=neighbor,
        interface=(interfaces[0] if interfaces else None),
        potential_causes=[],
        devices=devices,
        next_steps=next_steps
    )

# ====== Handoff al Troubleshooter ======
def handoff_to_troubleshooter(report: AlertReport) -> None:
    """
    Preferido: llamar al troubleshooter si está disponible (import directo).
    Fallback: escribir JSON en OUTBOX_DIR para que otro proceso lo recoja.
    """
    try:
        from troubleshooter_agent import handle_alert  # tu agente separado
        handle_alert(report.model_dump())
    except Exception:
        os.makedirs(OUTBOX_DIR, exist_ok=True)
        fname = os.path.join(
            OUTBOX_DIR, f"alert_{datetime.utcnow().strftime('%Y%m%dT%H%M%S%fZ')}.json"
        )
        with open(fname, "w", encoding="utf-8") as f:
            f.write(json.dumps(report.model_dump(), indent=2))
        print(f"[Agent Alert] Troubleshooter no disponible. Se escribió: {fname}")

# ====== Bucle principal del servidor Syslog ======
def start_syslog_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind((SYSLOG_HOST, SYSLOG_PORT))
    except OSError as e:
        if e.errno == 13:
            raise SystemExit(f"Permiso denegado en el puerto {SYSLOG_PORT}. Prueba 5514 o ejecuta con privilegios.")
        elif e.errno == 98:
            raise SystemExit(f"El puerto {SYSLOG_PORT} ya está en uso.")
        else:
            raise

    print(f"[Agent Alert] Escuchando syslog en {SYSLOG_HOST}:{SYSLOG_PORT} ...")
    try:
        while True:
            data, addr = sock.recvfrom(8192)
            log_line = data.decode("utf-8", errors="ignore").strip()
            if not log_line:
                continue

            src_ip, src_port = addr[0], addr[1]
            print(f"\n--- [{datetime.utcnow().strftime('%Y-%m-%d %H:%M:%SZ')}] SYSLOG from {addr} ---")
            print(log_line)

            # 1) Parser determinista para ISIS
            report = try_parse_isis_adjchange(log_line, (src_ip, src_port))
            # 2) Si no aplica, fallback (LLM o mínimo)
            if report is None:
                report = llm_parse_generic(log_line, (src_ip, src_port))

            print("\n[Agent Alert] Parsed report:")
            print(json.dumps(report.model_dump(), indent=2))

            handoff_to_troubleshooter(report)

    except KeyboardInterrupt:
        print("\n[Agent Alert] Detenido por el usuario.")
    finally:
        sock.close()

if __name__ == "__main__":
    start_syslog_server()
