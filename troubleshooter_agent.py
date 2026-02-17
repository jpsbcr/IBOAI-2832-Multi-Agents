# troubleshooter_agent.py
# ISIS Troubleshooter with Local RAG
# - Builds local RAG index from JSON folders
# - Runs "show running-config router isis" on local/neighbor
# - Compares LIVE vs GOLDEN (RAG) and computes minimal remediation plan
# - LLM produces report + optional RemediationCommands
# - auto_apply=True applies the plan (whitelist) via pyATS configure()

import os
import re
import json
import argparse
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel, Field

# ========= Local RAG Data =========
LLM_DATA_ISIS_DB_DETAIL_DIR = "llm_data_isis_db_detail"
LLM_DATA_ISIS_NEIGHBORS_DIR = "llm_data_isis_neighbors"
LLM_DATA_IP_ROUTE_ISIS_DIR = "llm_data_ip_route_isis"
LLM_DATA_IP_INT_BRIEF_DIR = "llm_data_ip_int_brief"
LLM_DATA_RUN_ROUTER_ISIS_DIR = "llm_data_run_router_isis"
CHROMA_DB_DIR = "chroma_db_troubleshooter_local_v1"

# ========= OpenAI (DO NOT hardcode keys) =========
# export OPENAI_API_KEY="..."
if not os.getenv("OPENAI_API_KEY"):
    raise SystemExit("ERROR: Please export OPENAI_API_KEY in your environment.")

from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# ========= LangChain / RAG =========
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# ========= pyATS =========
TESTBED = None
try:
    testbed_path = os.getenv("TESTBED_PATH", "testbed.yaml")
    from pyats.topology import loader
    TESTBED = loader.load(testbed_path)
except Exception:
    TESTBED = None

DEBUG_TS = os.getenv("DEBUG_TS", "0").lower() in ("1", "true", "yes")
AUTO_APPLY_DEFAULT = os.getenv("AUTO_APPLY", "0").lower() in ("1", "true", "yes")

# =========================
# Models
# =========================
class DeviceCommandPlan(BaseModel):
    device_name: str
    interfaces: List[str] = Field(default_factory=list)
    commands: List[str] = Field(default_factory=list)

class AlertReport(BaseModel):
    raw_log: str
    timestamp_utc: str
    technology: Optional[str] = None
    severity: str
    summary: str
    local_device_name: Optional[str] = None
    source_ip: Optional[str] = None
    source_port: Optional[int] = None
    neighbor: Optional[str] = None
    interface: Optional[str] = None
    potential_causes: List[str] = Field(default_factory=list)
    devices: List[str] = Field(default_factory=list)
    next_steps: List[DeviceCommandPlan] = Field(default_factory=list)

# =========================
# Execution Policy
# =========================
RUN_ON_SEVERITIES = {"CRITICAL", "MAJOR"}
VERIFY_LIGHT_ON_UP = False

def _is_isis_up(alert: AlertReport) -> bool:
    return (alert.technology or "").upper() == "ISIS" and "up" in f"{alert.summary} {alert.raw_log}".lower()

def _should_run(alert: AlertReport) -> bool:
    sev = (alert.severity or "").upper()
    tech = (alert.technology or "").upper()
    if sev in RUN_ON_SEVERITIES:
        return True
    if tech == "ISIS":
        return "down" in f"{alert.summary} {alert.raw_log}".lower()
    return False

def _extract_level_from_alert(alert: AlertReport) -> Optional[str]:
    m = re.search(r"\((L[12])\)", alert.raw_log)
    return m.group(1) if m else None

# =========================
# pyATS Helpers
# =========================
def _get_device(name: str):
    if TESTBED is None:
        return None, "ERROR: pyATS testbed not initialized or not accessible."
    if name not in TESTBED.devices:
        return None, f"ERROR: device '{name}' not found in testbed."
    dev = TESTBED.devices[name]
    try:
        if not dev.is_connected():
            dev.connect(log_stdout=False)
        return dev, None
    except Exception as e:
        return None, f"ERROR connecting to {name}: {e}"

def run_cli(device_name: str, command: str) -> str:
    dev, err = _get_device(device_name)
    if err:
        return err
    try:
        out = dev.execute(command)
        return out if isinstance(out, str) and out else "(no output)"
    except Exception as e:
        return f"ERROR executing '{command}' on {device_name}: {e}"

def apply_config(device_name: str, commands: List[str]) -> str:
    """
    Applies safe IOS XR configuration and commits it.
    """
    SAFE_PREFIXES = (
        "router isis",
        " interface ",
        "  point-to-point",
        " address-family",
        "  metric-style",
        "  authentication",
        " is-type",
        " metric-style",
        " authentication",
        " net ",
        " segment-routing",
        "!",
    )

    BLOCKLIST = [
        "reload", "format", "write erase", "erase",
        "copy ", "configure replace",
        "shutdown", "no shutdown", "commit replace"
    ]

    def is_safe(line: str) -> bool:
        l = line.rstrip()
        low = l.lower()
        if any(b in low for b in BLOCKLIST):
            return False
        if l.strip() == "!":
            return True
        return any(l.startswith(p) for p in SAFE_PREFIXES)

    dev, err = _get_device(device_name)
    if err:
        return err

    try:
        safe_lines = [ln for ln in commands if is_safe(ln)]
        if not safe_lines:
            return "(no lines passed safety whitelist)"

        payload = "\n".join(safe_lines)
        result = dev.configure(payload) or "(configuration pushed)"

        # Explicit commit for IOS XR
        os_name = getattr(dev, "os", "") or ""
        if "xr" in os_name.lower():
            commit = dev.execute("commit") or "(commit done)"
            return f"{result}\n{commit}"

        return result

    except Exception as e:
        return f"ERROR applying configuration on {device_name}: {e}"

# =========================
# Parse: "show running-config router isis" (LIVE and GOLDEN)
# =========================
class IsisProcessCfg(BaseModel):
    device: str
    raw: str
    tag: Optional[str] = None
    nets: List[str] = Field(default_factory=list)
    is_type: Optional[str] = None
    metric_style: Optional[str] = None
    af_ipv4: bool = False
    af_ipv6: bool = False
    auth_mode: Optional[str] = None
    seg_routing: Optional[str] = None

    # NEW: track per-interface point-to-point presence
    interfaces_p2p: Dict[str, bool] = Field(default_factory=dict)

    notes: List[str] = Field(default_factory=list)

def _normalize_cfg_text(txt: str) -> List[str]:
    lines = [ln.rstrip() for ln in (txt or "").splitlines()]
    return [ln for ln in lines if ln.strip() and not ln.strip().startswith("Building ")]

def parse_router_isis_config(device: str, text: str) -> IsisProcessCfg:
    cfg = IsisProcessCfg(device=device, raw=text or "")

    if (text or "").startswith("ERROR"):
        cfg.notes.append(text.strip())
        return cfg

    lines = _normalize_cfg_text(text)
    if not lines:
        cfg.notes.append("No output or command error.")
        return cfg

    # Router ISIS tag
    for ln in lines:
        m = re.match(r"^router\s+isis(?:\s+(\S+))?$", ln.strip(), re.IGNORECASE)
        if m:
            cfg.tag = m.group(1) or None
            break

    last_iface: Optional[str] = None

    for ln in lines:
        s = ln.strip()

        # Address families
        if re.match(r"^address-family\s+ipv4\s+unicast", s, re.IGNORECASE):
            cfg.af_ipv4 = True
        if re.match(r"^address-family\s+ipv6\s+unicast", s, re.IGNORECASE):
            cfg.af_ipv6 = True

        # NET
        m = re.search(r"\bnet\s+(\S+)", s, re.IGNORECASE)
        if m:
            cfg.nets.append(m.group(1))

        # IS-type
        m = re.search(r"\bis-type\s+(level-\d(?:-\d)?(?:-only)?)", s, re.IGNORECASE)
        if m:
            cfg.is_type = m.group(1).lower()

        # Metric style
        m = re.search(r"\bmetric-style\s+(wide|narrow)", s, re.IGNORECASE)
        if m:
            cfg.metric_style = m.group(1).lower()

        # Authentication
        m = re.search(r"authentication\s+(mode\s+(\S+)|keychain\s+(\S+))", s, re.IGNORECASE)
        if m:
            if m.group(2):
                cfg.auth_mode = m.group(2).lower()
            elif m.group(3):
                cfg.auth_mode = f"keychain:{m.group(3)}"

        # Segment routing
        if re.search(r"\bsegment-routing\s+mpls\b", s, re.IGNORECASE):
            cfg.seg_routing = "mpls"
        if re.search(r"\bsegment-routing\s+srv6\b", s, re.IGNORECASE):
            cfg.seg_routing = "srv6"

        # Interface context
        if s.lower().startswith("interface "):
            last_iface = s.split(" ", 1)[1].strip()
            cfg.interfaces_p2p.setdefault(last_iface, False)

        # Point-to-point under an ISIS interface stanza
        if s.lower() == "point-to-point":
            if last_iface:
                cfg.interfaces_p2p[last_iface] = True

    return cfg

# =========================
# Persistence
# =========================
def _persist_live_outputs(alert: AlertReport, outputs: Dict[str, str]) -> str:
    os.makedirs("live_captures", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = f"live_captures/live_{alert.local_device_name or 'unknown'}_isis_run_{ts}.json"
    payload = {"alert": alert.model_dump(), "shows": outputs}
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2))
    print(f"[Troubleshooter] Saved live outputs: {path}")
    return path

def _persist_report(alert: AlertReport, report_md: str, live_path: Optional[str]) -> str:
    os.makedirs("ts_reports", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = f"ts_reports/ts_{alert.local_device_name or 'unknown'}_isis_run_{ts}.md"
    header = f"> Live JSON: {live_path}\n\n" if live_path else ""
    with open(path, "w", encoding="utf-8") as f:
        f.write(header + report_md)
    print(f"[Troubleshooter] Saved report: {path}")
    return path

def _persist_remediation(alert: AlertReport, lines: List[str]) -> str:
    os.makedirs("ts_reports", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = f"ts_reports/ts_{alert.local_device_name or 'unknown'}_isis_run_{ts}_remediation.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    if DEBUG_TS:
        print(f"[Troubleshooter] Saved remediation plan: {path}")
    return path

def _persist_prompt_preview(alert: AlertReport, prompt_text: str) -> str:
    os.makedirs("ts_reports", exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    path = f"ts_reports/ts_{alert.local_device_name or 'unknown'}_isis_run_{ts}_prompt_preview.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(prompt_text)
    if DEBUG_TS:
        print(f"[Troubleshooter] Saved prompt preview: {path}")
    return path

def _mark_empty(s: str, label: str) -> str:
    if not s or not s.strip() or s.strip() in ("{}", "[]"):
        return f"(missing {label})"
    return s

# =========================
# Local RAG Utilities
# =========================
def _as_text(blob) -> str:
    if blob is None:
        return ""
    if isinstance(blob, str):
        return blob
    try:
        return json.dumps(blob, ensure_ascii=False, indent=2)
    except Exception:
        return str(blob)

def parse_isis_db_detail(data, device_name) -> List[Document]:
    docs = []
    dev = data.get(device_name)
    if not dev:
        return docs
    inst = dev.get("instance", {}).get("1")
    lvl2 = inst.get("level", {}).get("2") if inst else None
    lspids = lvl2.get("lspid", {}) if lvl2 else {}
    for lsp_id, lsp_full in lspids.items():
        lsp_hdr = lsp_full.get("lsp", {})
        combined = {**lsp_full, **lsp_hdr}
        ip_reach = combined.get("extended_ipv4_reachability", {}) or {}
        is_nbr = combined.get("extended_is_neighbor", {}) or {}
        docs.append(Document(
            page_content=f"ISIS LSP from device {device_name}, LSP ID {lsp_id}:\n{_as_text(combined)}",
            metadata={
                "source": f"{device_name}_isis_db_detail",
                "type": "isis_lsp_detail",
                "device_name": device_name,
                "lsp_id": lsp_id,
                "level": lsp_hdr.get("level"),
                "ip_reachability_prefixes": ", ".join(ip_reach.keys()),
                "is_neighbor_ids": ", ".join(is_nbr.keys()),
            }
        ))
    return docs

def parse_isis_neighbors(data, device_name) -> List[Document]:
    docs = []
    dev = data.get(device_name)
    if not dev:
        return docs
    isis1 = dev.get("isis", {}).get("1", {})
    vrf_def = isis1.get("vrf", {}).get("default", {})
    ifaces = vrf_def.get("interfaces", {}) or {}
    for if_name, if_data in ifaces.items():
        nbrs = if_data.get("neighbors", {}) or {}
        for sysid, nbr in nbrs.items():
            docs.append(Document(
                page_content=f"ISIS Neighbor from device {device_name}, interface {if_name}, System ID {sysid}:\n{_as_text(nbr)}",
                metadata={
                    "source": f"{device_name}_isis_neighbors",
                    "type": "isis_neighbor_info",
                    "device_name": device_name,
                    "interface": if_name,
                    "neighbor_system_id": sysid,
                    "state": nbr.get("state"),
                    "circuit_type": nbr.get("type"),
                    "ip_address": nbr.get("ip_address"),
                }
            ))
    return docs

def parse_ip_route_isis(data, device_name) -> List[Document]:
    docs = []
    dev = data.get(device_name)
    if not dev:
        return docs
    vrfs = dev.get("vrfs", {}) or {}
    for vrf, vrf_data in vrfs.items():
        af = vrf_data.get("address_family", {}).get("ipv4", {})
        routes = af.get("routes", {}) or {}
        for prefix, entry in routes.items():
            for route in entry.get("route", []) or []:
                nh_list = route.get("next_hops", []) or []
                nh_summary = "; ".join([
                    f"via {nh.get('next_hop_address','?')} on {nh.get('outgoing_interface','?')}"
                    for nh in nh_list
                ])
                docs.append(Document(
                    page_content=f"ISIS IPv4 Route {prefix} on {device_name} (VRF {vrf}):\n{_as_text(route)}",
                    metadata={
                        "source": f"{device_name}_ip_route_isis",
                        "type": "isis_ipv4_route",
                        "device_name": device_name,
                        "vrf": vrf,
                        "prefix": prefix,
                        "metric": route.get("metric"),
                        "admin_distance": route.get("administrative_distance"),
                        "next_hops_summary": nh_summary,
                    }
                ))
    return docs

def parse_ip_int_brief(data, device_name) -> List[Document]:
    docs = []
    dev = data.get(device_name)
    if not dev:
        return docs
    ifaces = dev.get("interface", {}) or {}
    for if_name, if_data in ifaces.items():
        docs.append(Document(
            page_content=f"Interface details for device {device_name}, interface {if_name}:\n{_as_text(if_data)}",
            metadata={
                "source": f"{device_name}_ip_int_brief",
                "type": "device_interface_info",
                "device_name": device_name,
                "interface_name": if_name,
                "status": if_data.get("status"),
                "protocol_status": if_data.get("protocol_status"),
                "ip_address": if_data.get("ip_address"),
            }
        ))
    return docs

def parse_run_router_isis(data, device_name) -> List[Document]:
    """
    Expects { "<dev>": { "command": "...", "parsed": {...}, "raw_text": "..." } }
    Indexes head, AF and interface subtrees as separate docs (better retrieval).
    """
    docs: List[Document] = []
    dev = data.get(device_name)
    if not dev:
        return docs

    parsed = dev.get("parsed") or {}
    raw = dev.get("raw_text") or ""

    router_blocks = [k for k in parsed.keys() if k.startswith("router isis ")]
    if not router_blocks:
        if raw:
            docs.append(Document(
                page_content=f"ISIS Running Configuration (raw) for device {device_name}:\n{raw}",
                metadata={"source": f"{device_name}_run_router_isis", "type": "isis_running_config_raw", "device_name": device_name}
            ))
        return docs

    def _head_flags(head_block: dict) -> dict:
        flags = {}
        for k in head_block.keys():
            if k.startswith("is-type "): flags["is_type"] = k.split(" ", 1)[1]
            if k.startswith("net "): flags["net"] = k.split(" ", 1)[1]
            if k == "log adjacency changes": flags["log_adj_changes"] = "true"
        return flags

    for head in router_blocks:
        head_block = parsed.get(head, {})
        head_kv = _head_flags(head_block)
        docs.append(Document(
            page_content=f"{head} on {device_name}:\n{_as_text(head_block)}",
            metadata={"source": f"{device_name}_run_router_isis", "type": "run_router_isis", "device_name": device_name, **head_kv}
        ))

        for k, v in head_block.items():
            if k == "address-family ipv4 unicast" and isinstance(v, dict):
                docs.append(Document(
                    page_content=f"{head} address-family ipv4 unicast on {device_name}:\n{_as_text(v)}",
                    metadata={"source": f"{device_name}_run_router_isis", "type": "run_router_isis_af_ipv4", "device_name": device_name, **head_kv}
                ))

            if k.startswith("interface ") and isinstance(v, dict):
                if_name = k.split(" ", 1)[1]
                p2p = "true" if "point-to-point" in v else "false"
                docs.append(Document(
                    page_content=f"{head} interface {if_name} on {device_name}:\n{_as_text(v)}",
                    metadata={
                        "source": f"{device_name}_run_router_isis",
                        "type": "run_router_isis_interface",
                        "device_name": device_name,
                        "interface": if_name,
                        "p2p": p2p,
                        **head_kv
                    }
                ))

    if raw:
        docs.append(Document(
            page_content=f"RAW TEXT (running-config router isis) {device_name}:\n{raw}",
            metadata={"source": f"{device_name}_run_router_isis", "type": "isis_running_config_raw", "device_name": device_name}
        ))
    return docs

def _load_and_make_docs(*directories) -> List[Document]:
    all_docs: List[Document] = []
    parsers = {
        LLM_DATA_ISIS_DB_DETAIL_DIR: parse_isis_db_detail,
        LLM_DATA_ISIS_NEIGHBORS_DIR: parse_isis_neighbors,
        LLM_DATA_IP_ROUTE_ISIS_DIR: parse_ip_route_isis,
        LLM_DATA_IP_INT_BRIEF_DIR: parse_ip_int_brief,
        LLM_DATA_RUN_ROUTER_ISIS_DIR: parse_run_router_isis,
    }
    prefix_map = {
        "isis_db_detail": LLM_DATA_ISIS_DB_DETAIL_DIR,
        "isis_neighbors": LLM_DATA_ISIS_NEIGHBORS_DIR,
        "ip_route_isis": LLM_DATA_IP_ROUTE_ISIS_DIR,
        "ip_int_brief": LLM_DATA_IP_INT_BRIEF_DIR,
        "run_router_isis": LLM_DATA_RUN_ROUTER_ISIS_DIR,
    }

    for directory in directories:
        if not os.path.exists(directory):
            print(f"[RAG] Warning: directory '{directory}' not found. Skipping.")
            continue
        parser_func = parsers.get(directory)
        if not parser_func:
            print(f"[RAG] Warning: no parser for '{directory}'. Skipping.")
            continue

        for filename in os.listdir(directory):
            if not filename.endswith(".json"):
                continue
            path = os.path.join(directory, filename)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                detected_prefix = None
                for prefix_key, dir_path in prefix_map.items():
                    if dir_path == directory and filename.startswith(f"{prefix_key}_"):
                        detected_prefix = prefix_key
                        break
                if not detected_prefix:
                    print(f"[RAG] Warning: file '{filename}' prefix not recognized in '{directory}'. Skipping.")
                    continue

                device_name = filename[len(f"{detected_prefix}_"):-len(".json")]
                new_docs = parser_func(data, device_name)
                all_docs.extend(new_docs)
                if DEBUG_TS:
                    print(f"[RAG] Parsed {parser_func.__name__} for '{device_name}' from {filename} ({len(new_docs)} docs)")
            except Exception as e:
                print(f"[RAG] Error parsing {filename} in '{directory}': {e}")

    return all_docs

_RAG_CTX = None

def setup_local_rag():
    global _RAG_CTX
    if _RAG_CTX:
        return _RAG_CTX

    print("[RAG] Loading and processing JSON documents...")
    parent_docs = _load_and_make_docs(
        LLM_DATA_ISIS_DB_DETAIL_DIR,
        LLM_DATA_ISIS_NEIGHBORS_DIR,
        LLM_DATA_IP_ROUTE_ISIS_DIR,
        LLM_DATA_IP_INT_BRIEF_DIR,
        LLM_DATA_RUN_ROUTER_ISIS_DIR,
    )
    if not parent_docs:
        print("[RAG] No valid documents found. Check your JSON folders.")
        _RAG_CTX = None
        return None

    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=100)

    embeddings = OpenAIEmbeddings()
    store = InMemoryStore()

    vectordb = Chroma(
        collection_name="troubleshooter_parent_doc_retriever_v1",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    retriever = ParentDocumentRetriever(
        vectorstore=vectordb,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter
    )

    need_index = False
    try:
        need_index = not os.path.exists(CHROMA_DB_DIR) or not os.listdir(CHROMA_DB_DIR)
    except Exception:
        need_index = True

    if need_index:
        print(f"[RAG] Creating/reindexing '{CHROMA_DB_DIR}' with {len(parent_docs)} documents...")
        retriever.add_documents(parent_docs)
        print("[RAG] Index created.")
    else:
        print(f"[RAG] Using existing index in '{CHROMA_DB_DIR}'.")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer using ONLY the provided context (ISIS DB detail, ISIS neighbors, ISIS routes, interface brief, and running-config router isis). "
         "If required info is missing, say so explicitly."),
        ("human", "Context: {context}\n\nQuestion: {input}")
    ])

    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)

    _RAG_CTX = {"rag_chain": rag_chain, "retriever": retriever, "llm": llm}
    return _RAG_CTX

def query_baseline(rag_chain, question: str) -> Dict:
    try:
        return rag_chain.invoke({"input": question})
    except Exception as e:
        return {"answer": f"(RAG error) {e}", "context": []}

def extract_golden_router_isis(rag_resp: Dict, local: Optional[str], neighbor: Optional[str]) -> Dict[str, str]:
    per_dev: Dict[str, str] = {}
    ctx = rag_resp.get("context") or []
    wanted = set([d for d in [local, neighbor] if d])

    for doc in ctx:
        if isinstance(doc, dict):
            md = doc.get("metadata", {}) or {}
            txt = doc.get("page_content", "") or ""
        else:
            md = getattr(doc, "metadata", {}) or {}
            txt = getattr(doc, "page_content", "") or ""

        dev = (md.get("device_name") or md.get("device") or "").strip()
        tp = (md.get("type") or "").lower()

        score = 0
        if tp == "run_router_isis": score += 3
        if "run_router_isis" in tp or "isis_running_config" in tp: score += 2
        if "router isis" in (txt or "").lower(): score += 1
        if dev and dev in wanted: score += 2
        if score == 0:
            continue

        key = dev or "unknown"
        per_dev[key] = (per_dev.get(key, "") + "\n" + txt).strip()

    if not per_dev and ctx:
        for doc in ctx:
            if isinstance(doc, dict):
                md = doc.get("metadata", {}) or {}
                txt = doc.get("page_content", "") or ""
            else:
                md = getattr(doc, "metadata", {}) or {}
                txt = getattr(doc, "page_content", "") or ""
            if "router isis" in (txt or "").lower():
                key = (md.get("device_name") or "unknown").strip() or "unknown"
                per_dev[key] = (per_dev.get(key, "") + "\n" + txt).strip()

    return per_dev

def _maybe_json_to_cli(text: str, device_hint: Optional[str] = None) -> str:
    if not text or "{" not in text:
        return text
    try:
        start = text.find("{")
        data = json.loads(text[start:])
    except Exception:
        return text

    node = None
    if device_hint and isinstance(data, dict) and device_hint in data:
        node = data.get(device_hint)
    if node is None:
        node = data

    if isinstance(node, dict) and "parsed" in node and isinstance(node["parsed"], dict):
        node = node["parsed"]

    if isinstance(node, dict):
        def _dict_to_cli_lines(d: dict, indent: int = 0) -> list:
            lines = []
            for key, val in d.items():
                lines.append((" " * indent) + key)
                if isinstance(val, dict) and val:
                    lines.extend(_dict_to_cli_lines(val, indent + 1))
            return lines
        return "\n".join(_dict_to_cli_lines(node))

    return text

def parse_golden_map(golden_per_dev: Dict[str, str]) -> Dict[str, IsisProcessCfg]:
    out: Dict[str, IsisProcessCfg] = {}
    for dev, txt in (golden_per_dev or {}).items():
        cli_txt = _maybe_json_to_cli(txt, device_hint=dev)
        out[dev] = parse_router_isis_config(dev, cli_txt)
    return out

# =========================
# Remediation Planning
# =========================
def _first_tag(*tags) -> str:
    for t in tags:
        if t:
            return t
    return "1"

def _line_unique(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for l in seq:
        if l not in seen:
            seen.add(l)
            out.append(l)
    return out

def _area_ids_from_nets(nets: List[str]) -> List[str]:
    out = []
    for n in nets or []:
        m = re.match(r"^(\d{2})\.(\d{4})", n.replace(" ", ""))
        if m:
            out.append(f"{m.group(1)}.{m.group(2)}")
    return sorted(set(out))

def build_missing_from_golden(local: IsisProcessCfg, golden: IsisProcessCfg, level_hint: Optional[str]) -> List[str]:
    if not golden or not golden.raw:
        return []
    lines: List[str] = []
    tag = _first_tag(local.tag, golden.tag)
    lines.append(f"router isis {tag}")
    if golden.is_type and golden.is_type != local.is_type:
        lines.append(f" is-type {golden.is_type}")
    if golden.af_ipv4 and not local.af_ipv4:
        lines.append(" address-family ipv4 unicast")
        if golden.metric_style:
            lines.append(f"  metric-style {golden.metric_style}")
        lines.append(" !")
    if golden.metric_style and golden.metric_style != local.metric_style:
        lines.append(f" metric-style {golden.metric_style}")
    if golden.auth_mode and golden.auth_mode != local.auth_mode:
        if golden.auth_mode.startswith("keychain:"):
            kc = golden.auth_mode.split(":", 1)[1]
            lines.append(f" authentication keychain {kc}")
        else:
            lines.append(f" authentication mode {golden.auth_mode}")
    if golden.seg_routing and golden.seg_routing != local.seg_routing:
        lines.append(" segment-routing " + golden.seg_routing)
    if level_hint == "L1":
        if golden.nets and set(_area_ids_from_nets(local.nets)).isdisjoint(set(_area_ids_from_nets(golden.nets))):
            lines.append(f"! net {golden.nets[0]}   <<< REVIEW MANUALLY")
    return _line_unique(lines)

def build_missing_from_neighbor(local: IsisProcessCfg, neighbor: IsisProcessCfg, level_hint: Optional[str]) -> List[str]:
    if not neighbor or not neighbor.raw:
        return []
    lines: List[str] = []
    tag = _first_tag(local.tag, neighbor.tag)
    lines.append(f"router isis {tag}")
    if neighbor.is_type and neighbor.is_type != local.is_type:
        if level_hint == "L2" and "level-2" in neighbor.is_type:
            lines.append(f" is-type {neighbor.is_type}")
        elif level_hint == "L1" and "level-1" in neighbor.is_type:
            lines.append(f" is-type {neighbor.is_type}")
        elif not level_hint:
            lines.append(f" is-type {neighbor.is_type}")
    if neighbor.af_ipv4 and not local.af_ipv4:
        lines.append(" address-family ipv4 unicast")
        if neighbor.metric_style:
            lines.append(f"  metric-style {neighbor.metric_style}")
        lines.append(" !")
    if neighbor.metric_style and neighbor.metric_style != local.metric_style:
        lines.append(f" metric-style {neighbor.metric_style}")
    if neighbor.auth_mode and neighbor.auth_mode != local.auth_mode:
        if neighbor.auth_mode.startswith("keychain:"):
            kc = neighbor.auth_mode.split(":", 1)[1]
            lines.append(f" authentication keychain {kc}")
        else:
            lines.append(f" authentication mode {neighbor.auth_mode}")
    if neighbor.seg_routing and neighbor.seg_routing != local.seg_routing:
        lines.append(" segment-routing " + neighbor.seg_routing)
    if level_hint == "L1":
        if neighbor.nets and set(_area_ids_from_nets(local.nets)).isdisjoint(set(_area_ids_from_nets(neighbor.nets))):
            lines.append(f"! net {neighbor.nets[0]}   <<< REVIEW MANUALLY")
    return _line_unique(lines)

def merge_golden_neighbor_plan(local: IsisProcessCfg,
                               golden: Optional[IsisProcessCfg],
                               neighbor: Optional[IsisProcessCfg],
                               level_hint: Optional[str]) -> List[str]:
    from_golden = build_missing_from_golden(local, golden or IsisProcessCfg(device="golden", raw=""), level_hint)
    from_neighbor = build_missing_from_neighbor(local, neighbor or IsisProcessCfg(device="neighbor", raw=""), level_hint)
    if not from_golden and not from_neighbor:
        return []
    header = None
    for l in (from_golden or from_neighbor):
        if l.startswith("router isis "):
            header = l
            break
    merged = [header or f"router isis {_first_tag(local.tag, (golden and golden.tag), (neighbor and neighbor.tag))}"]

    def add_unique(lines):
        for l in lines:
            if l.startswith("router isis "):
                continue
            if l not in merged:
                merged.append(l)

    add_unique(from_golden)
    add_unique(from_neighbor)
    return _line_unique(merged)

# =========================
# NEW: Point-to-Point Fix Logic
# =========================
def build_p2p_fix(local: IsisProcessCfg,
                  golden: Optional[IsisProcessCfg],
                  neighbor: Optional[IsisProcessCfg],
                  iface: Optional[str]) -> List[str]:
    """
    If neighbor or golden expects 'point-to-point' under ISIS interface, but local does not have it,
    generate a minimal interface-level fix.
    """
    if not iface:
        return []

    local_has_p2p = local.interfaces_p2p.get(iface, False)

    expected_p2p = False
    if golden and iface in golden.interfaces_p2p:
        expected_p2p = expected_p2p or golden.interfaces_p2p.get(iface, False)
    if neighbor and iface in neighbor.interfaces_p2p:
        expected_p2p = expected_p2p or neighbor.interfaces_p2p.get(iface, False)

    if expected_p2p and not local_has_p2p:
        tag = _first_tag(local.tag, (golden and golden.tag), (neighbor and neighbor.tag))
        return [
            f"router isis {tag}",
            f" interface {iface}",
            "  point-to-point",
            " !"
        ]

    return []

# =========================
# LLM (analysis/report)
# =========================
llm = ChatOpenAI(model="gpt-4o", temperature=0.0)

analysis_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a Senior Network Architect expert in Cisco IOS XR and ISIS.\n"
     "Validate ISIS process consistency between LOCAL and NEIGHBOR using:\n"
     "- GOLDEN (RAG) router isis snippets\n"
     "- LIVE RAW and LIVE PARSED from both devices\n"
     "- Candidate plan already computed\n"
     "If something is missing, say it explicitly and continue.\n"
     "Always return:\n"
     "```RemediationCommands\n<IOS XR lines to align the LOCAL>\n```"),
    ("human",
     "ALERT:\n{alert_json}\n\n"
     "GOLDEN (RAG):\n{golden_json}\n\n"
     "LIVE RAW:\n{live_raw_json}\n\n"
     "LIVE PARSED:\n{live_parsed_json}\n\n"
     "CANDIDATE PLAN (computed):\n{candidate_plan}\n\n"
     "Generate: Diagnosis, Evidence, Workaround, Permanent Fix, Verification.")
])

def _extract_remediation_commands(markdown_text: str) -> List[str]:
    """
    Robust extraction of remediation commands from LLM output.
    Supports:
      - ```RemediationCommands
      - ```plaintext / ```text / ```bash
      - Any fenced block containing 'router isis'
    """
    if not markdown_text:
        return []

    txt = markdown_text.replace("\r\n", "\n")

    # 1) Prefer block after 'RemediationCommands'
    section = re.search(r"RemediationCommands\s*\n", txt, re.IGNORECASE)
    if section:
        after = txt[section.end():]
        code = re.search(r"```[a-zA-Z0-9_-]*\s*\n(.*?)\n```", after, re.DOTALL)
        if code:
            return [l.rstrip() for l in code.group(1).splitlines() if l.strip()]

    # 2) Otherwise, first block containing 'router isis'
    for m in re.finditer(r"```[a-zA-Z0-9_-]*\s*\n(.*?)\n```", txt, re.DOTALL):
        block = m.group(1)
        if "router isis" in block.lower():
            return [l.rstrip() for l in block.splitlines() if l.strip()]

    return []

# =========================
# Main entry
# =========================
def handle_alert(report: dict, auto_apply: bool = True) -> None:
    try:
        alert = AlertReport(**report)
        print("\n[Troubleshooter] Alert received:")
        print(json.dumps(alert.model_dump(), indent=2))

        if not _should_run(alert):
            if VERIFY_LIGHT_ON_UP and _is_isis_up(alert):
                dev = alert.local_device_name or (alert.devices[0] if alert.devices else None)
                if dev:
                    print(f"[Troubleshooter] Light verification on {dev}: show running-config router isis")
                    print(run_cli(dev, "show running-config router isis"))
            else:
                print("[Troubleshooter] No action needed for this alert.")
            return

        local = alert.local_device_name or (alert.devices[0] if alert.devices else None)
        neighbor = alert.neighbor or (alert.devices[1] if len(alert.devices) > 1 else None)
        if not local:
            print("[Troubleshooter] Missing local device in alert — cannot continue.")
            return

        # 1) LIVE: run command on local (+ neighbor if present)
        outputs: Dict[str, str] = {}
        for dev in [d for d in [local, neighbor] if d]:
            print(f"[Troubleshooter] Running on {dev}: show running-config router isis")
            out = run_cli(dev, "show running-config router isis")
            outputs[dev] = out
            if out.startswith("ERROR"):
                print(out)

        # 2) Persist LIVE
        live_path = _persist_live_outputs(alert, outputs)

        # 3) Parse LIVE
        parsed: Dict[str, IsisProcessCfg] = {
            dev: parse_router_isis_config(dev, txt) for dev, txt in outputs.items()
        }

        # 4) Local RAG: GOLDEN
        ctx = setup_local_rag()
        if ctx is None:
            print("[Troubleshooter] RAG not initialized — continuing without GOLDEN.")
            rag_resp = {"answer": "(no RAG)", "context": []}
            golden_per_dev = {}
        else:
            rag_chain = ctx["rag_chain"]
            q = (
                "Return GOLDEN 'router isis' snippets per device for: "
                f"{', '.join([d for d in [local, neighbor] if d])}. "
                "Prioritize docs where metadata.type includes 'run_router_isis' or 'isis_running_config'."
            )
            rag_resp = query_baseline(rag_chain, q)

            ctx_docs = rag_resp.get("context") or []
            if DEBUG_TS:
                print(f"[Troubleshooter][RAG] context items: {len(ctx_docs)}")
                for i, d in enumerate(ctx_docs[:10]):
                    if isinstance(d, dict):
                        md = d.get("metadata", {})
                        txt = d.get("page_content", "")[:120].replace("\n", " ")
                    else:
                        md = getattr(d, "metadata", {}) or {}
                        txt = (getattr(d, "page_content", "") or "")[:120].replace("\n", " ")
                    print(f"  - [{i}] device_name={md.get('device_name')} type={md.get('type')} source={md.get('source')} text~={txt}")

            golden_per_dev = extract_golden_router_isis(rag_resp, local, neighbor)

        golden_parsed: Dict[str, IsisProcessCfg] = parse_golden_map(golden_per_dev)
        golden_local = golden_parsed.get(local)
        golden_nei = golden_parsed.get(neighbor) if neighbor else None

        level_hint = _extract_level_from_alert(alert)

        # 5) Candidate plan (base)
        local_cfg = parsed.get(local) or IsisProcessCfg(device=local, raw="")
        nei_cfg = parsed.get(neighbor) if neighbor else None
        candidate_plan = merge_golden_neighbor_plan(local_cfg, golden_local, nei_cfg, level_hint)

        # 5b) NEW: add point-to-point fix if mismatch detected
        p2p_fix = build_p2p_fix(local_cfg, golden_local, nei_cfg, alert.interface)
        if p2p_fix:
            print("[Troubleshooter] Detected possible ISIS point-to-point mismatch — adding remediation.")
            candidate_plan.extend(p2p_fix)
            candidate_plan = _line_unique(candidate_plan)

        _persist_remediation(alert, candidate_plan)

        # 6) Prepare inputs for LLM
        alert_json = json.dumps(alert.model_dump(), indent=2)
        golden_json = json.dumps({"answer": rag_resp.get("answer", ""), "per_device_snippets": golden_per_dev}, indent=2)
        live_raw_json = json.dumps(outputs, indent=2)
        live_parsed_json = json.dumps({k: v.model_dump() for k, v in parsed.items()}, indent=2)
        candidate_text = "\n".join(candidate_plan) if candidate_plan else "(EMPTY PLAN)"

        golden_json = _mark_empty(golden_json, "GOLDEN")
        live_raw_json = _mark_empty(live_raw_json, "LIVE RAW")
        live_parsed_json = _mark_empty(live_parsed_json, "LIVE PARSED")

        prompt_preview_text = (
            "ALERT:\n" + alert_json + "\n\n" +
            "GOLDEN (RAG):\n" + golden_json + "\n\n" +
            "LIVE RAW:\n" + live_raw_json + "\n\n" +
            "LIVE PARSED:\n" + live_parsed_json + "\n\n" +
            "CANDIDATE PLAN:\n" + candidate_text + "\n"
        )
        _persist_prompt_preview(alert, prompt_preview_text)

        chain = analysis_prompt | llm
        llm_resp = chain.invoke({
            "alert_json": alert_json,
            "golden_json": golden_json,
            "live_raw_json": live_raw_json,
            "live_parsed_json": live_parsed_json,
            "candidate_plan": candidate_text
        })
        final_md = getattr(llm_resp, "content", None) or str(llm_resp) or "(no LLM output)"

        print("\n[Troubleshooter] Final report (LLM):\n")
        print(final_md)

        _persist_report(alert, final_md, live_path)

        # 7) Auto remediation
        if auto_apply:
            cmds = _extract_remediation_commands(final_md) or candidate_plan
            if cmds:
                print(f"\n[Troubleshooter] auto_apply=True → Applying configuration on {local} ...")
                res = apply_config(local, cmds)
                print(res)
            else:
                print("[Troubleshooter] No commands to apply (LLM + plan are empty).")

    except SystemExit as se:
        print(f"[Troubleshooter] SystemExit: {se}")
        raise
    except Exception as e:
        import traceback
        print(f"[Troubleshooter] ERROR in handle_alert: {e.__class__.__name__}: {e}")
        traceback.print_exc()
        raise

# =========================
# CLI
# =========================
def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="ISIS Troubleshooter Agent (Local RAG)")
    parser.add_argument("--alert-file", help="Path to alert JSON (from Agent Alert)")
    parser.add_argument("--auto-apply", action="store_true", help="Apply remediation automatically")
    parser.add_argument("--no-auto-apply", action="store_true", help="Do NOT apply remediation")
    parser.add_argument("--demo", action="store_true", help="Run a local demo alert")
    args = parser.parse_args()

    auto_apply = AUTO_APPLY_DEFAULT
    if args.auto_apply:
        auto_apply = True
    elif args.no_auto_apply:
        auto_apply = False

    if args.alert_file:
        report = _load_json(args.alert_file)
        handle_alert(report, auto_apply=auto_apply)
        return

    if args.demo:
        demo = {
            "raw_log": "%ROUTING-ISIS-5-ADJCHANGE : Adjacency to P-1 (GigabitEthernet0/0/0/0) (L2) Down, Interface state down",
            "timestamp_utc": "2025-08-10T20:33:07Z",
            "technology": "ISIS",
            "severity": "CRITICAL",
            "summary": "ISIS adjacency DOWN — service impact possible; immediate checks required.",
            "local_device_name": "PE-1",
            "neighbor": "P-1",
            "interface": "GigabitEthernet0/0/0/0",
            "devices": ["PE-1", "P-1"],
            "next_steps": []
        }
        handle_alert(demo, auto_apply=auto_apply)
        return

    parser.print_help()

if __name__ == "__main__":
    main()
