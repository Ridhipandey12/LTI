# frontend.py - Streamlit frontend for the SOC RAG Assistant backend
# Run: streamlit run frontend.py

import os
import streamlit as st
import requests
import datetime
import json
import re

# ---- CONFIG: use environment variable BACKEND_URL or default to localhost ----
BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
QUERY_ENDPOINT = f"{BACKEND_URL.rstrip('/')}/query/"
HEALTH_ENDPOINT = f"{BACKEND_URL.rstrip('/')}/health"

st.set_page_config(page_title="SOC RAG Assistant", layout="wide")

# --- Styles ---
st.markdown("""
<style>
.stApp { background: #f7fafc; }
.header { font-size: 1.6rem; font-weight: 700; color: #0f172a; }
.muted { color: #6b7280; margin-bottom: 12px; }
.chat-box { background: #fff; padding: 12px; border-radius: 10px; box-shadow: 0 6px 18px rgba(15,23,42,0.06); }
.message { padding: 10px 14px; border-radius: 12px; margin-bottom: 10px; max-width: 88%; }
.user-message { background-color: #0b84ff; color: white; }
.assistant-message { background-color: #f1f5f9; color: #0f172a; margin-left: auto; }
.timestamp { display:block; font-size: 0.8rem; color:#475569; margin-top:8px; }
.json-box { background:#0f172a; color:#e6eef8; padding:10px; border-radius:8px; font-family: monospace; white-space: pre-wrap;}
</style>
""", unsafe_allow_html=True)

# ---- Helpers ----
def now_ts():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def call_backend(session_id, query, timeout=30):
    payload = {"session_id": session_id, "query": query}
    try:
        r = requests.post(QUERY_ENDPOINT, json=payload, timeout=timeout)
    except Exception as e:
        return {"error": str(e)}
    try:
        return r.json()
    except Exception:
        return {"error": f"Invalid backend response: {r.status_code} - {r.text}"}

def check_health():
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=3)
        if r.status_code == 200:
            return r.json()
        return {"status": "unhealthy", "code": r.status_code, "text": r.text}
    except Exception as e:
        return {"status": "down", "error": str(e)}

# Local fallback entity extraction + scoring (simple, mirrors backend fallback)
def extract_entities_local(query: str, context: str = ""):
    text = (query or "") + "\n" + (context or "")
    out = {}
    ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}\b"
    ips = re.findall(ip_pattern, text)
    if ips:
        out["ips"] = list(dict.fromkeys(ips))
    host_pattern = r"\b[a-zA-Z0-9\-\_\.]{3,}\b"
    hosts = []
    for tok in re.findall(host_pattern, text):
        if "." in tok or re.search(r"[A-Za-z]+[0-9]+", tok) or tok.lower().startswith(("host","srv")):
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

def calculate_threat_score_local(entities, retrieved_snippets, query_text=""):
    score = 0
    reasons = []
    sev_map = {"Critical": 35, "High": 25, "Medium": 12, "Low": 4}
    if entities.get("severity"):
        for s in entities["severity"]:
            v = sev_map.get(s.capitalize(), 0)
            score += v
            reasons.append(f"Severity '{s}' => +{v}")
    if entities.get("mitre"):
        for t in entities["mitre"]:
            t_upper = t.upper()
            added = 10
            if any(k in t_upper for k in ["1003","CREDENTIAL","CRED","PASSWORD"]):
                added = 25
            elif any(k in t_upper for k in ["1059","POWERSHELL","EXECUTE","CMD"]):
                added = 18
            elif any(k in t_upper for k in ["T1078","ACCOUNT","LOGON","ACCESS"]):
                added = 20
            elif any(k in t_upper for k in ["LATERAL","RDP","SMB","PASS THE HASH"]):
                added = 20
            score += added
            reasons.append(f"MITRE {t_upper} => +{added}")
    ips = entities.get("ips", [])
    for ip in ips:
        ip = ip.strip()
        if ip.endswith(".13") or ip.startswith("192.0"):
            score += 30
            reasons.append(f"Malicious IP {ip} => +30")
        else:
            score += 6
            reasons.append(f"Unknown IP {ip} => +6")
    search_text = (query_text or "") + " " + " ".join(retrieved_snippets or [])
    search_text = search_text.lower()
    ps_indicators = ["powershell","encodedcommand","iex ","invoke-expression","frombase64string","base64","-enc "]
    ps_hits = sum(1 for kw in ps_indicators if kw in search_text)
    if ps_hits:
        add_ps = min(30, 10 + (ps_hits-1)*8)
        score += add_ps
        reasons.append(f"Suspicious PowerShell indicators ({ps_hits}) => +{add_ps}")
    bf_indicators = ["failed login","failed logins","authentication failure","multiple failed","failed password","brute force","account lock"]
    bf_hits = sum(1 for kw in bf_indicators if kw in search_text)
    if bf_hits:
        add_bf = min(30, 8 * bf_hits)
        score += add_bf
        reasons.append(f"Brute-force indicators ({bf_hits}) => +{add_bf}")
    snippet_bonus = min(10, (len(retrieved_snippets) if retrieved_snippets else 0) * 2)
    if snippet_bonus:
        score += snippet_bonus
        reasons.append(f"{len(retrieved_snippets)} retrieved snippets => +{snippet_bonus}")
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

def build_json_response_local(rag_result, entities, query):
    retrieved = rag_result.get("retrieved", []) or []
    threat_info = calculate_threat_score_local(entities, retrieved, query_text=query)
    level = threat_info["level"]
    if level == "Severe":
        recs = ["Isolate systems","Block malicious IPs","Preserve logs & forensic capture","Reset credentials"]
    elif level == "High":
        recs = ["Increase monitoring","Collect logs","Investigate hosts"]
    elif level == "Moderate":
        recs = ["Monitor systems","Verify patches"]
    else:
        recs = ["Continue monitoring"]
    confidence = round(min(1.0, threat_info["score"]/100.0 + min(0.1, len(retrieved)*0.02)), 2)
    related = [{"incident_id": f"r{i+1}", "content": s} for i, s in enumerate(retrieved)]
    return {
        "summary": rag_result.get("answer","(no answer)"),
        "recommended_actions": recs,
        "confidence": confidence,
        "threat_score": threat_info["score"],
        "related_incidents": related,
        "entities": entities,
        "threat_reasons": threat_info.get("reasons", [])
    }

# ---- Session state ----
if "session_id" not in st.session_state:
    st.session_state.session_id = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "last_structured" not in st.session_state:
    st.session_state.last_structured = None

# ---- Sidebar ----
with st.sidebar:
    st.markdown("<div class='header'>SOC RAG Assistant</div>", unsafe_allow_html=True)
    st.text_input("Session ID (analyst)", key="session_id", placeholder="analyst1")
    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.last_structured = None
    st.markdown("---")
    st.markdown("Backend status:")
    health = check_health()
    if health.get("status") == "ok":
        st.success("Backend OK")
    else:
        st.warning(json.dumps(health))

# ---- Main layout ----
col1, col2 = st.columns([3,1])
with col1:
    st.markdown("<div class='header'>Ask SOC — RAG Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Enter your query and get a human answer plus structured JSON.</div>", unsafe_allow_html=True)
    st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
    for m in st.session_state.chat_history:
        role = m.get("role")
        text = m.get("text")
        ts = m.get("ts")
        if role == "user":
            st.markdown(f"<div class='message user-message'><b>You:</b> {text}<span class='timestamp'>{ts}</span></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='message assistant-message'><b>Assistant:</b> {text}<span class='timestamp'>{ts}</span></div>", unsafe_allow_html=True)
            if m.get("json"):
                st.markdown("<details><summary>Structured JSON</summary>", unsafe_allow_html=True)
                st.code(json.dumps(m.get("json"), indent=2))
                st.markdown("</details>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("### Compose")
    # Set the value to an empty string after submission
    query_text = st.text_area("Query", key="query_area", height=140, placeholder="e.g., multiple failed ssh logins from 192.168.10.13", value="")
    timeout = st.number_input("Timeout (s)", value=30, min_value=5, max_value=120)
    submit = st.button("Ask")

# ---- Submit handling ----
if submit:
    sid = st.session_state.session_id.strip()
    if not sid:
        st.warning("Please provide session ID in sidebar.")
    elif not query_text.strip():
        st.warning("Please type a query.")
    else:
        st.session_state.chat_history.append({"role":"user","text":query_text.strip(),"ts":now_ts()})
        with st.spinner("Contacting backend..."):
            resp = call_backend(sid, query_text.strip(), timeout=timeout)
        if resp.get("error"):
            assistant_text = f"Error: {resp['error']}"
            st.session_state.chat_history.append({"role":"assistant","text":assistant_text,"ts":now_ts()})
        else:
            structured = resp.get("structured") if isinstance(resp, dict) else None
            if not structured:
                answer = resp.get("answer") if isinstance(resp, dict) else str(resp)
                retrieved = resp.get("retrieved", []) if isinstance(resp, dict) else []
                entities = resp.get("entities", {}) if isinstance(resp, dict) else extract_entities_local(query_text.strip(), "\n".join(retrieved))
                structured = build_json_response_local({"answer": answer, "retrieved": retrieved}, entities, query_text.strip())
            assistant_text = structured.get("summary") or resp.get("answer") or "(no answer)"
            st.session_state.chat_history.append({"role":"assistant","text":assistant_text,"ts":now_ts(),"json":structured})
            st.session_state.last_structured = structured
        # **DO NOT modify session_state directly for the text field**
        # Instead, the text area automatically clears with `value=""` upon submission.
        query_text = ""  # reset the query text after submission

# ---- Bottom panel ----
st.markdown("---")
st.markdown("### Last structured response")
if st.session_state.last_structured:
    st.markdown("<div class='json-box'>", unsafe_allow_html=True)
    st.text(json.dumps(st.session_state.last_structured, indent=2))
    st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("No structured response yet. Ask a query to get one.")

st.markdown("---")
st.caption(f"Using backend at: {BACKEND_URL}  — ensure backend serves POST {QUERY_ENDPOINT}")
