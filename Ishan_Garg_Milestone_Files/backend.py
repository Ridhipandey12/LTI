# backend.py
# FastAPI wrapper for the SOC RAG Assistant (uses the same langchain imports you requested)

import os
import re
import json
import time
import threading
import traceback
from typing import Dict, Any, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# === ONLY these LangChain imports (as requested) ===
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.retrievers import BM25Retriever
# EnsembleRetriever may be present in langchain_classic; attempt import but fallback is provided
try:
    from langchain_classic.retrievers import EnsembleRetriever
except Exception:
    try:
        from langchain.retrievers import EnsembleRetriever
    except Exception:
        EnsembleRetriever = None
# ===================================================

# Standard libs
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="SOC RAG Backend")

# ---------- Config ----------
TICKET_FILE = "security_incidents.txt"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
VECTOR_K = 4
LLM_MODEL_NAME = "mistral"

# ---------- Global state ----------
_texts: List[str] = []
_metadatas: List[Dict[str, Any]] = []
_faiss_index = None
_vector_retriever = None
_bm25_retriever = None
_hybrid_retriever = None
_embeddings = None
_llm = None

# session memory
_store: Dict[str, InMemoryChatMessageHistory] = {}

# prompt
prompt_template_text = """
You are a SOC Incident Assistant.

Relevant retrieved incident snippets:
{context}

Analyst memory / session prefs:
{entities}

Recent analyst chat history:
{history}

Question:
{question}

Provide:
1) A concise diagnostic summary.
2) Suggested next steps / remediation.
3) Any notable indicators (IPs, hostnames, commands, MITRE tags) found.
Give the answer as plain text. Keep it actionable and succinct.
"""
prompt = ChatPromptTemplate.from_template(prompt_template_text)
parser = StrOutputParser()


def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def safe_read_tickets(path: str):
    lines = []
    if not os.path.exists(path):
        log(f"Ticket file '{path}' not found; continuing with empty dataset.")
        return lines
    with open(path, "r", encoding="utf-8") as f:
        for i, l in enumerate(f):
            l = l.strip()
            if not l:
                continue
            lines.append({"ticket_id": f"t{i+1}", "text": l})
    return lines


def build_docs(lines: List[Dict[str, Any]]):
    docs = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    for row in lines:
        chunks = splitter.split_text(row["text"])
        for chunk in chunks:
            docs.append({"text": chunk, "metadata": {"ticket_id": row["ticket_id"], "orig_line": row["text"]}})
    return docs


def build_indexes():
    global _faiss_index, _vector_retriever, _bm25_retriever, _embeddings, _texts, _metadatas, _hybrid_retriever
    if not _texts:
        log("No texts to index.")
        return
    try:
        _embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        _faiss_index = FAISS.from_texts(_texts, _embeddings, metadatas=_metadatas)
        try:
            _vector_retriever = _faiss_index.as_retriever(search_kwargs={"k": VECTOR_K})
        except Exception:
            _vector_retriever = None
        log("FAISS index built.")
    except Exception as e:
        log(f"FAISS build failed: {e}\n{traceback.format_exc()}")
        _faiss_index = None
        _vector_retriever = None

    try:
        _bm25_retriever = BM25Retriever.from_texts(_texts, metadatas=_metadatas)
        log("BM25 retriever built.")
    except Exception as e:
        log(f"BM25 build failed: {e}\n{traceback.format_exc()}")
        _bm25_retriever = None

    # Try EnsembleRetriever if available, else SimpleHybrid
    if EnsembleRetriever is not None and _vector_retriever and _bm25_retriever:
        try:
            _hybrid_retriever = EnsembleRetriever(retrievers=[_vector_retriever, _bm25_retriever], weights=[0.7, 0.3])
            log("Ensemble retriever created.")
            return
        except Exception:
            _hybrid_retriever = None

    # Fallback hybrid wrapper
    class SimpleHybrid:
        def __init__(self, vec, bm, k=VECTOR_K):
            self.vec = vec
            self.bm = bm
            self.k = k

        def get_relevant_documents(self, query):
            vecs = []
            try:
                if self.vec:
                    vecs = self.vec.get_relevant_documents(query)
            except Exception:
                try:
                    vecs = self.vec.retrieve(query)
                except Exception:
                    vecs = []
            bms = []
            try:
                if self.bm:
                    bms = self.bm.get_relevant_documents(query)
            except Exception:
                try:
                    bms = self.bm.invoke(query)
                except Exception:
                    bms = []
            merged = []
            seen = set()
            for r in (vecs + bms):
                content = getattr(r, "page_content", None) or (r.get("page_content") if isinstance(r, dict) else str(r))
                meta = getattr(r, "metadata", None) or (r.get("metadata") if isinstance(r, dict) else {})
                key = (content, meta.get("ticket_id", meta.get("source", "")))
                if key in seen:
                    continue
                seen.add(key)
                merged.append({"page_content": content, "metadata": meta})
                if len(merged) >= self.k:
                    break
            return merged

    _hybrid_retriever = SimpleHybrid(_vector_retriever, _bm25_retriever, k=VECTOR_K)
    log("Simple hybrid retriever ready.")


def initialize_data():
    global _texts, _metadatas
    log("Initializing data...")
    lines = safe_read_tickets(TICKET_FILE)
    docs = build_docs(lines)
    _texts = [d["text"] for d in docs]
    _metadatas = [d["metadata"] for d in docs]
    build_indexes()
    log(f"Initialization done. {_texts.__len__()} chunks loaded.")


# init in background
_init_thread = threading.Thread(target=initialize_data, daemon=True)
_init_thread.start()

# LLM init
try:
    _llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0.3)
    log("ChatOllama initialized.")
except Exception as e:
    _llm = None
    log(f"ChatOllama init failed: {e}")


def call_llm(messages):
    if not _llm:
        return "(LLM unavailable; configure Ollama)"
    try:
        llm_response = _llm.invoke(messages)
        if hasattr(llm_response, "generations"):
            return llm_response.generations[0][0].text
        elif isinstance(llm_response, dict) and "text" in llm_response:
            return llm_response["text"]
        else:
            return getattr(llm_response, "content", None) or str(llm_response)
    except Exception as e:
        log(f"LLM call error: {e}\n{traceback.format_exc()}")
        return f"(LLM call failed: {e})"


# ---------------- Retrieval helper ----------------
def retriever_get_docs(retriever, query, top_k=VECTOR_K):
    try:
        if hasattr(retriever, "get_relevant_documents"):
            docs = retriever.get_relevant_documents(query)
            return docs[:top_k]
        if hasattr(retriever, "retrieve"):
            docs = retriever.retrieve(query)
            return docs[:top_k]
        # fallback None
        return []
    except Exception:
        try:
            return retriever.retrieve(query)[:top_k]
        except Exception:
            return []


# ---------------- Scoring and helpers ----------------
def threat_enrich_tool(ip: str) -> Dict[str, Any]:
    ip = ip.strip()
    if not ip:
        return {"ip": ip, "reputation": "unknown"}
    # mock intel
    if ip.endswith(".13") or ip.startswith("192.0"):
        return {"ip": ip, "reputation": "malicious", "last_seen": "2025-09-01", "confidence": "medium"}
    return {"ip": ip, "reputation": "unknown", "last_seen": None, "confidence": "low"}


def extract_entities(query: str, context_text: str = "") -> Dict[str, Any]:
    text = (query or "") + "\n" + (context_text or "")
    out: Dict[str, Any] = {}
    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ips = re.findall(ip_pattern, text)
    if ips:
        out["ips"] = list(dict.fromkeys(ips))
    host_pattern = r"\b[a-zA-Z0-9\-\_\.]{3,}\b"
    hosts = []
    for tok in re.findall(host_pattern, text):
        if "." in tok or re.search(r"[A-Za-z]+[0-9]+", tok) or tok.lower().startswith(("host", "srv")):
            if len(tok) <= 64:
                hosts.append(tok)
    if hosts:
        out["hosts"] = list(dict.fromkeys(hosts))
    os_matches = re.findall(r"\b(Windows\s*(?:10|11|7|8|8\.1)|Windows|Linux|Ubuntu|CentOS|macOS|darwin)\b", text, flags=re.IGNORECASE)
    if os_matches:
        out["os"] = list(dict.fromkeys([m.strip() for m in os_matches]))
    t_matches = re.findall(r"\bT\d{4}\b", text, flags=re.IGNORECASE)
    if t_matches:
        out["mitre"] = list(dict.fromkeys([t.upper() for t in t_matches]))
    sev_matches = re.findall(r"\b(Critical|High|Medium|Low)\b", text, flags=re.IGNORECASE)
    if sev_matches:
        out["severity"] = list(dict.fromkeys([s.capitalize() for s in sev_matches]))
    return out


def calculate_threat_score_v2(entities: dict, retrieved_snippets: list, query_text: str = "") -> dict:
    score = 0
    reasons = []
    sev_map = {"Critical": 35, "High": 25, "Medium": 12, "Low": 4}
    if entities.get("severity"):
        for s in entities["severity"]:
            v = sev_map.get(s.capitalize(), 0)
            score += v
            reasons.append(f"Severity '{s}' => +{v}")
    mitre_weights = {
        "credential": 25, "access": 20, "execution": 18, "persistence": 18,
        "lateral": 20, "privilege": 18, "defense-evasion": 15, "discovery": 8, "exfiltration": 20
    }
    if entities.get("mitre"):
        for t in entities["mitre"]:
            t_upper = t.upper()
            added = 10
            if any(k in t_upper for k in ["1003", "CREDENTIAL", "CRED", "PASSWORD"]):
                added = mitre_weights.get("credential", 25)
            elif any(k in t_upper for k in ["1059", "POWERSHELL", "EXECUTE", "CMD"]):
                added = mitre_weights.get("execution", 18)
            elif any(k in t_upper for k in ["T1078", "ACCOUNT", "LOGON", "ACCESS"]):
                added = mitre_weights.get("access", 20)
            elif any(k in t_upper for k in ["LATERAL", "RDP", "SMB", "PASS THE HASH"]):
                added = mitre_weights.get("lateral", 20)
            score += added
            reasons.append(f"MITRE {t_upper} => +{added}")
    ips = entities.get("ips", [])
    for ip in ips:
        ip = ip.strip()
        if ip.endswith(".13") or ip.startswith("192.0"):
            ip_score = 30
            score += ip_score
            reasons.append(f"Malicious IP {ip} => +{ip_score}")
        else:
            intel = threat_enrich_tool(ip)
            rep = intel.get("reputation", "unknown").lower()
            if rep == "malicious":
                score += 30
                reasons.append(f"Threat intel: {ip} reputation=malicious => +30")
            elif rep == "suspicious":
                score += 18
                reasons.append(f"Threat intel: {ip} reputation=suspicious => +18")
            else:
                score += 6
                reasons.append(f"Threat intel: {ip} reputation=unknown => +6")
    search_text = " ".join([query_text or ""]) + " " + " ".join(retrieved_snippets or [])
    search_text = search_text.lower()
    ps_indicators = ["powershell", "encodedcommand", "iex ", "invoke-expression", "frombase64string", "base64", "-enc "]
    ps_hits = sum(1 for kw in ps_indicators if kw in search_text)
    if ps_hits > 0:
        add_ps = min(30, 10 + (ps_hits - 1) * 8)
        score += add_ps
        reasons.append(f"Suspicious PowerShell indicators ({ps_hits}) => +{add_ps}")
    bf_indicators = ["failed login", "failed logins", "authentication failure", "multiple failed", "failed password", "brute force", "account lock"]
    bf_hits = sum(1 for kw in bf_indicators if kw in search_text)
    if bf_hits > 0:
        add_bf = min(30, 8 * bf_hits)
        score += add_bf
        reasons.append(f"Brute-force indicators ({bf_hits}) => +{add_bf}")
    try:
        snippets_count = len(retrieved_snippets) if retrieved_snippets else 0
    except Exception:
        snippets_count = 0
    snippet_bonus = min(10, snippets_count * 2)
    if snippet_bonus:
        score += snippet_bonus
        reasons.append(f"{snippets_count} retrieved snippets => +{snippet_bonus}")
    score = max(0, min(100, int(score)))
    if score >= 80:
        level = "Severe"
    elif score >= 60:
        level = "High"
    elif score >= 40:
        level = "Moderate"
    else:
        level = "Low"
    return {"score": score, "level": level, "reasons": reasons}


def create_json_response(rag_result: Dict[str, Any], threat_info: Dict[str, Any], entities: Dict[str, Any], query: str) -> Dict[str, Any]:
    summary = rag_result.get("answer", "(no answer)")
    score = threat_info.get("score", 0)
    level = threat_info.get("level", "Low")
    if level == "Severe":
        recommended_actions = [
            "Isolate affected systems immediately",
            "Block malicious IPs at network perimeter",
            "Start forensic capture and preserve logs",
            "Perform full malware/AV scan and credential reset"
        ]
    elif level == "High":
        recommended_actions = [
            "Increase monitoring and collect logs",
            "Block or throttle source IPs if confirmed",
            "Conduct targeted investigation on involved hosts"
        ]
    elif level == "Moderate":
        recommended_actions = [
            "Monitor affected accounts and systems closely",
            "Verify patch levels and review recent changes"
        ]
    else:
        recommended_actions = ["Continue monitoring, gather additional context if available"]
    confidence = min(1.0, max(0.0, score / 100.0 + min(0.1, len(rag_result.get("retrieved", [])) * 0.02)))
    related_incidents = []
    for idx, snippet in enumerate(rag_result.get("retrieved", [])):
        related_incidents.append({"incident_id": f"r{idx+1}", "content": snippet})
    return {
        "summary": summary,
        "recommended_actions": recommended_actions,
        "confidence": round(confidence, 2),
        "threat_score": score,
        "related_incidents": related_incidents,
        "entities": entities or {}
    }


# ---------- Memory helper ----------
def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in _store:
        _store[session_id] = InMemoryChatMessageHistory()
    return _store[session_id]


# ---------- FastAPI models ----------
class QueryRequest(BaseModel):
    session_id: str
    query: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "has_faiss": bool(_faiss_index),
        "has_bm25": bool(_bm25_retriever),
        "has_llm": bool(_llm),
        "chunks_loaded": len(_texts),
    }


@app.post("/query/")
def query_endpoint(req: QueryRequest):
    session_id = (req.session_id or "").strip()
    question = (req.query or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    if not question:
        raise HTTPException(status_code=400, detail="query is required")

    mem = get_session_history(session_id)
    hist_msgs = []
    try:
        if hasattr(mem, "messages"):
            hist_msgs = [getattr(m, "content", str(m)) for m in mem.messages]
        else:
            hist_msgs = mem.load_memory_variables({}).get("history", [])
    except Exception:
        hist_msgs = []

    chat_history_str = "\n".join(hist_msgs) if hist_msgs else "(no prior history)"

    # Retrieve
    retrieved_objs = retriever_get_docs(_hybrid_retriever or _vector_retriever or _bm25_retriever, question, top_k=VECTOR_K)
    context_lines = []
    for r in retrieved_objs:
        content = getattr(r, "page_content", None) or (r.get("page_content") if isinstance(r, dict) else str(r))
        meta = getattr(r, "metadata", None) or (r.get("metadata") if isinstance(r, dict) else {})
        tag = meta.get("ticket_id") or meta.get("id") or meta.get("source", "?")
        short = content if len(content) <= 1000 else content[:1000] + " ...[truncated]"
        context_lines.append(f"- [{tag}] {short}")
    context_str = "\n".join(context_lines) if context_lines else "(no retrieved snippets)"

    # Entities
    entities = extract_entities(question, context_str)

    # Build prompt text robustly
    # prefer message-based template if available
    template_string = None
    try:
        if hasattr(prompt, "messages") and len(prompt.messages) > 0:
            try:
                template_string = prompt.messages[0].prompt.template
            except Exception:
                template_string = None
    except Exception:
        template_string = None
    if template_string is None:
        try:
            template_string = prompt.template
        except Exception:
            template_string = None
    if template_string is None:
        template_string = prompt_template_text

    formatted = template_string.format(context=context_str, entities=json.dumps(entities, indent=2),
                                       history=chat_history_str, question=question)

    messages = [{"role": "system", "content": "You are a helpful SOC analyst assistant."},
                {"role": "user", "content": formatted}]

    answer = call_llm(messages)

    # Save to memory (best-effort)
    try:
        mem.add_user_message(question)
        mem.add_ai_message(answer)
    except Exception:
        try:
            mem.add_message(HumanMessage(content=question))
            mem.add_message(AIMessage(content=answer))
        except Exception:
            pass

    rag_result = {"answer": answer, "retrieved": context_lines}
    threat_info = calculate_threat_score_v2(entities, context_lines, query_text=question)
    json_resp = create_json_response(rag_result, threat_info, entities, question)

    # return both human answer and structured JSON
    return {"answer": answer, "retrieved": context_lines, "entities": entities, "structured": json_resp}


# ensure init has started
if _init_thread.is_alive():
    log("Background initialization started (building indexes). It may continue in background.")
