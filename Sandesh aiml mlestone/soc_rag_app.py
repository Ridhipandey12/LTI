
"""
soc_rag_app.py

SOC Analyst Assistant — Incident RAG Investigation System

This console-based application implements a complete RAG + Memory workflow
to help SOC analysts diagnose new security alerts using historical incident data.

Main Features:
1. Dataset Loading
   - Reads security_incidents.txt (1 incident per line)
   - Displays count and sample lines

2. Chunking Pipeline
   - Uses RecursiveCharacterTextSplitter (fallback: manual splitter)
   - Generates ~300–500 character chunks with overlap

3. Embedding & Vector Indexing
   - SentenceTransformer embeddings
   - FAISS index (if available) or NumPy cosine similarity fallback

4. Hybrid Retrieval
   - Vector similarity + BM25 (or TF-IDF fallback)
   - Weighted scoring: vector(0.7) + textual(0.3)

5. LCEL RAG Chain
   - retrieve → prompt → LLM (Ollama Mistral) → output parser
   - Injects retrieved context, entity memory, past analyst conversation

6. Memory System
   - RunnableWithMessageHistory for per-analyst chat history
   - Custom entity memory across sessions (IP, OS, MITRE tags, hostnames, severity)

7. Entity Extraction
   - Extracts IPs, OS names, hostnames, MITRE technique IDs (Txxxx), severity levels
   - Applies to queries, retrieved incidents, and assistant responses

8. Threat Scoring Engine
   - Scores incidents based on severity, MITRE tags, malicious indicators, suspicious text

9. Tool Integration
   - Threat Enrichment tool: mock IP intelligence (ASN / Geo / Reputation)

10. Structured Output
   - Generates final JSON-like dictionary with:
     - Retrieved refs
     - Entity memory
     - Threat score
     - Assistant response
     - Tool output (if used)
"""


import os
import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# LCEL / LangChain building blocks
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

# Text splitter
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    # fallback simple splitter if package not available
    RecursiveCharacterTextSplitter = None

# Embeddings
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    raise RuntimeError("Please install sentence-transformers: pip install sentence-transformers")

# Optional FAISS
_use_faiss = False
try:
    import faiss  # type: ignore
    _use_faiss = True
except Exception:
    _use_faiss = False

# Optional BM25 (rank_bm25) else fallback to TF-IDF
_has_bm25 = False
try:
    from rank_bm25 import BM25Okapi
    _has_bm25 = True
except Exception:
    _has_bm25 = False
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

# ---------------- Config ----------------
INCIDENTS_PATH = Path("security_incidents.txt")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K = 4
VECTOR_WEIGHT = 0.7
TEXT_WEIGHT = 0.3

# ---------------- Load incidents ----------------
if not INCIDENTS_PATH.exists():
    print(f"[error] {INCIDENTS_PATH} not found. Place your security_incidents.txt (200-300 lines) here.")
    sys.exit(1)

raw_text = INCIDENTS_PATH.read_text(encoding="utf-8", errors="replace")
raw_lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
print(f"[info] Loaded {len(raw_lines)} incident lines. Sample:")
for i, ln in enumerate(raw_lines[:3], 1):
    print(f" {i}. {ln[:200]}")

# ---------------- Chunking ----------------
if RecursiveCharacterTextSplitter:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    # it expects Documents but we'll feed text pieces directly
else:
    splitter = None

def chunk_line(line: str) -> List[str]:
    if splitter:
        # splitter expects langchain Document; emulate by splitting by characters
        docs = splitter.split_text(line)
        return docs
    # fallback simple char-based splitting
    if len(line) <= CHUNK_SIZE:
        return [line]
    out = []
    i = 0
    while i < len(line):
        out.append(line[i:i+CHUNK_SIZE])
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return out

chunks: List[Dict[str, Any]] = []
for idx, line in enumerate(raw_lines):
    pieces = chunk_line(line)
    for j, p in enumerate(pieces):
        chunks.append({"text": p, "meta": {"orig_line": idx, "chunk_idx": j}})

print(f"[info] Created {len(chunks)} chunks from incidents.")

# ---------------- Embeddings & Index ----------------
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
texts = [c["text"] for c in chunks]
print("[info] Computing embeddings...")
embeddings = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
print(f"[info] Embeddings shape: {embeddings.shape}")

if _use_faiss:
    dim = embeddings.shape[1]
    # use IndexFlatIP with normalized vectors (cosine)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print("[info] FAISS index built.")
else:
    # normalize numpy embeddings for cosine
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    index = None
    print("[warn] FAISS not available; using in-memory vector similarity fallback.")

# ---------------- Text retriever (BM25 or TF-IDF) ----------------
tokenized_corpus = None
bm25_obj = None
tfidf_vec = None
tfidf_matrix = None

if _has_bm25:
    # tokenization simple split
    tokenized_corpus = [re.findall(r"\w+", t.lower()) for t in texts]
    bm25_obj = BM25Okapi(tokenized_corpus)
    print("[info] BM25 index built.")
else:
    # TF-IDF fallback
    tfidf_vec = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vec.fit_transform(texts)
    # normalize rows
    tfidf_matrix = normalize(tfidf_matrix)
    print("[info] TF-IDF matrix built as BM25 fallback.")

# ---------------- Hybrid retriever function ----------------
def retrieve_hybrid(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    """Return top-k chunk dicts by hybrid score (vector similarity + BM25/TF-IDF)."""
    # Vector part
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    if _use_faiss:
        D, I = index.search(q_emb, min(k*5, len(texts)))  # fetch a larger candidate set
        cand_idxs = I[0].tolist()
        vec_sims = {i: D[0][j] for j, i in enumerate(cand_idxs)}
    else:
        sims = (embeddings @ q_emb.T).squeeze()
        # take top 50 candidates
        cand_idxs = list(np.argsort(-sims)[:min(200, len(sims))])
        vec_sims = {i: float(sims[i]) for i in cand_idxs}

    # Text part
    text_scores = {}
    if _has_bm25:
        q_tok = re.findall(r"\w+", query.lower())
        bm25_scores = bm25_obj.get_scores(q_tok)
        for i in cand_idxs:
            text_scores[i] = float(bm25_scores[i])
    else:
        q_vec = tfidf_vec.transform([query])
        q_vec = normalize(q_vec)
        sims_t = (tfidf_matrix @ q_vec.T).toarray().squeeze()
        for i in cand_idxs:
            text_scores[i] = float(sims_t[i])

    # Combine and pick top-k
    combined = []
    for i in cand_idxs:
        v = vec_sims.get(i, 0.0)
        t = text_scores.get(i, 0.0)
        score = VECTOR_WEIGHT * v + TEXT_WEIGHT * t
        combined.append((i, score))
    combined.sort(key=lambda x: x[1], reverse=True)
    top = [idx for idx, _ in combined[:k]]
    return [chunks[i] for i in top]

# ---------------- Entity extraction ----------------
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
OS_RE = re.compile(r"\b(Windows\s*\d+|Linux|Ubuntu|CentOS|macOS|mac os|android|ios)\b", flags=re.I)
MITRE_RE = re.compile(r"\bT\d{4}\b", flags=re.I)
HOST_RE = re.compile(r"\bhost(?:name)?:\s*([A-Za-z0-9\-\._]+)\b", flags=re.I)
SEV_RE = re.compile(r"\b(low|medium|high|critical)\b", flags=re.I)

def extract_entities(text: str) -> Dict[str, List[str]]:
    ents: Dict[str, List[str]] = {"ips": [], "os": [], "hosts": [], "mitre": [], "severity": []}
    for ip in IP_RE.findall(text):
        ents["ips"].append(ip)
    for m in OS_RE.findall(text):
        ents["os"].append(m)
    for m in MITRE_RE.findall(text):
        ents["mitre"].append(m.upper())
    for m in HOST_RE.findall(text):
        ents["hosts"].append(m)
    for m in SEV_RE.findall(text):
        ents["severity"].append(m.lower())
    # dedupe
    for k in ents:
        ents[k] = list(dict.fromkeys(ents[k]))
    return ents

# ---------------- Threat scoring ----------------
def compute_threat_score(entities: Dict[str, List[str]], context_text: str) -> int:
    score = 0
    # severity influence
    sev = entities.get("severity", [])
    if "critical" in sev:
        score += 40
    elif "high" in sev:
        score += 25
    elif "medium" in sev:
        score += 10
    # presence of IPs
    if entities.get("ips"):
        score += 10
    # MITRE tag presence (suspicious techniques)
    if entities.get("mitre"):
        score += 10
    # suspicious keywords
    kws = ["powershell", "encoded", "ransomware", "brute-force", "credential dumping", "suspicious outbound"]
    for kw in kws:
        if kw in context_text.lower():
            score += 5
    # clamp 0-100
    return min(max(score, 0), 100)

# ---------------- LCEL chain build ----------------
extract = RunnableLambda(lambda x: x["question"] if isinstance(x, dict) and "question" in x else x)

def retriever_runnable_fn(x):
    q = x["question"] if isinstance(x, dict) and "question" in x else x
    hits = retrieve_hybrid(q, k=TOP_K)
    combined = "\n---\n".join([f"Incident[{h['meta']['orig_line']}]: {h['text']}" for h in hits])
    return combined

retriever_runnable = RunnableLambda(retriever_runnable_fn)

prompt_template = (
    "You are a SOC incident assistant.\n"
    "Retrieved incidents:\n{retrieved}\n\n"
    "Entity memory (extracted): {entities}\n"
    "Analyst history:\n{history}\n"
    "Query: {question}\n\n"
    "Provide:\n1) Short diagnosis\n2) Suggested resolution steps\n3) Key entities mention (ip/os/mitre)\n4) One-line justification\nReturn the response in concise text."
)
prompt = ChatPromptTemplate.from_template(prompt_template)
parser = StrOutputParser()

# LLM
llm = ChatOllama(model="mistral", temperature=0.2)

chain = (
    {
        "retrieved": retriever_runnable,
        "question": extract,
        "history": RunnablePassthrough(),
        "entities": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
)

# ---------------- Memory (per-analyst) ----------------
_sessions_hist: Dict[str, InMemoryChatMessageHistory] = {}
_sessions_entity_memory: Dict[str, Dict[str, List[str]]] = {}

def get_session_history(session_id: str):
    if session_id not in _sessions_hist:
        _sessions_hist[session_id] = InMemoryChatMessageHistory()
    return _sessions_hist[session_id]

# The memory wrapper will inject history at key "history"
memory_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

# ---------------- Tool: threat enrichment ----------------
@tool
def threat_enrich(text: str) -> str:
    """
    Perform mock threat enrichment for extracted IP addresses.
    Takes a text input, extracts IPs, and returns mock ASN/Geo/Reputation info.
    """
    ips = IP_RE.findall(text)
    if not ips:
        return "No IPs found to enrich."
    outs = []
    for ip in ips[:3]:
        outs.append(f"IP {ip}: ASN=AS{hash(ip)%10000}, Geo=CountryX, Reputation=Suspicious")
    return "\n".join(outs)


llm_with_tools = llm.bind_tools([threat_enrich])

# ---------------- Console loop ----------------
print("\n=== SOC RAG Assistant ===")
print("Type 'quit' as user_id or query to exit.\n")

while True:
    try:
        analyst = input("analyst_id> ").strip()
        if analyst.lower() in ("quit", "exit"):
            break
        query = input("query> ").strip()
        if query.lower() in ("quit", "exit"):
            break

        # 1) Retrieve relevant incidents (the retriever will also run inside chain; we run again for scoring/entities)
        hits = retrieve_hybrid(query, k=TOP_K)
        retrieved_text = "\n---\n".join([f"Incident[{h['meta']['orig_line']}]: {h['text']}" for h in hits])

        # 2) Extract entities from query + retrieved context
        ents_q = extract_entities(query)
        ents_ctx = extract_entities(retrieved_text)
        combined_entities = {
            "ips": list(dict.fromkeys(ents_q["ips"] + ents_ctx["ips"])),
            "os": list(dict.fromkeys(ents_q["os"] + ents_ctx["os"])),
            "hosts": list(dict.fromkeys(ents_q["hosts"] + ents_ctx["hosts"])),
            "mitre": list(dict.fromkeys(ents_q["mitre"] + ents_ctx["mitre"])),
            "severity": list(dict.fromkeys(ents_q["severity"] + ents_ctx["severity"]))
        }

        # 3) update per-analyst entity memory
        if analyst not in _sessions_entity_memory:
            _sessions_entity_memory[analyst] = {"ips": [], "os": [], "hosts": [], "mitre": [], "severity": []}
        # append discovered ents
        for k in _sessions_entity_memory[analyst]:
            merged = list(dict.fromkeys(_sessions_entity_memory[analyst][k] + combined_entities.get(k, [])))
            _sessions_entity_memory[analyst][k] = merged

        # 4) compute threat score (simple logic)
        threat_score = compute_threat_score(combined_entities, retrieved_text + "\n" + query)

        # 5) prepare input object for memory_chain (history injection)
        entities_str = json.dumps(_sessions_entity_memory[analyst], ensure_ascii=False)
        input_obj = {"question": query, "entities": entities_str, "prefs": "", "history": ""}  # history injected by wrapper
        config = {"configurable": {"session_id": analyst}}

        # 6) run the memory-wrapped chain (LLM returns a textual response)
        result = memory_chain.invoke(input_obj, config)
        assistant_text = str(result)

        # 7) If the LLM requested a tool call (tool-calling varies by wrapper), try to detect and resolve
        tool_calls = getattr(result, "tool_calls", None)
        if tool_calls is None and isinstance(result, dict):
            tool_calls = result.get("tool_calls")
        tool_info = None
        if tool_calls:
            # attempt resolve first tool call
            call = tool_calls[0]
            name = call.get("name") if isinstance(call, dict) else None
            args = call.get("args") if isinstance(call, dict) else None
            # resolve arg
            arg = None
            if isinstance(args, list) and args:
                arg = args[0]
            elif isinstance(args, str):
                arg = args
            else:
                arg = query + "\n" + retrieved_text
            if name == "threat_enrich" or name is None:
                tool_res = threat_enrich(arg)
                tool_info = tool_res
                assistant_text += "\n\n[Threat Enrichment]\n" + tool_res

        # 8) Structured JSON result construction
        structured = {
            "analyst": analyst,
            "query": query,
            "retrieved_count": len(hits),
            "retrieved_refs": [h["meta"]["orig_line"] for h in hits],
            "entities": _sessions_entity_memory[analyst],
            "threat_score": threat_score,
            "assistant_text": assistant_text,
            "tool_info": tool_info
        }

        # 9) Print outputs
        print("\n--- Retrieved Context ---")
        print(retrieved_text or "(none)")
        print("\n--- Injected Memory (entities) ---")
        print(json.dumps(_sessions_entity_memory[analyst], indent=2))
        print("\n--- Threat Score ---")
        print(threat_score)
        print("\n--- Assistant (final) ---")
        print(assistant_text)
        print("\n--- Structured Output (JSON) ---")
        print(json.dumps(structured, ensure_ascii=False, indent=2))
        print("\n-----------------------------\n")

        # 10) Update analyst history: RunnableWithMessageHistory already appended; also store entities from assistant text
        more_ents = extract_entities(assistant_text)
        for k in _sessions_entity_memory[analyst]:
            merged = list(dict.fromkeys(_sessions_entity_memory[analyst][k] + more_ents.get(k, [])))
            _sessions_entity_memory[analyst][k] = merged

    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
        break
    except Exception as e:
        print("[error]", repr(e))
        import traceback
        traceback.print_exc()
        continue



