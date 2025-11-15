
import os
import sys
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

print("spplitting the texts for embeddings")
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except Exception:
    RecursiveCharacterTextSplitter = None

print("embedding to pass in vectorstore")
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    raise RuntimeError("Please install sentence-transformers: pip install sentence-transformers")

print("putting in Faiss, faiss is a vectorstore")
_use_faiss = False
try:
    import faiss 
    _use_faiss = True
except Exception:
    _use_faiss = False

_has_bm25 = False
try:
    from rank_bm25 import BM25Okapi
    _has_bm25 = True
except Exception:
    _has_bm25 = False
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

INCIDENTS_PATH = Path("security_incidents.txt")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
TOP_K = 4
VECTOR_WEIGHT = 0.7
TEXT_WEIGHT = 0.3

if not INCIDENTS_PATH.exists():
    print(f"[error] {INCIDENTS_PATH} not found. Place your security_incidents.txt (200-300 lines) here.")
    sys.exit(1)

raw_text = INCIDENTS_PATH.read_text(encoding="utf-8", errors="replace")
raw_lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
print(f"[info] Loaded {len(raw_lines)} incident lines. Sample:")
for i, ln in enumerate(raw_lines[:3], 1):
    print(f" {i}. {ln[:200]}")

print("chunk process")
if RecursiveCharacterTextSplitter:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
else:
    splitter = None

def chunk_line(line: str) -> List[str]:
    if splitter:
        docs = splitter.split_text(line)
        return docs
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

print("embedding it")
embed_model = SentenceTransformer(EMBED_MODEL_NAME)
texts = [c["text"] for c in chunks]
print("[info] Computing embeddings...")
embeddings = embed_model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
print(f"[info] Embeddings shape: {embeddings.shape}")

if _use_faiss:
    dim = embeddings.shape[1]
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print("[info] FAISS index built.")
else:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    embeddings = embeddings / norms
    index = None
    print("[warn] FAISS not available; using in-memory vector similarity fallback.")

print("BM25 operation")
tokenized_corpus = None
bm25_obj = None
tfidf_vec = None
tfidf_matrix = None

if _has_bm25:
    tokenized_corpus = [re.findall(r"\w+", t.lower()) for t in texts]
    bm25_obj = BM25Okapi(tokenized_corpus)
    print("[info] BM25 index built.")
else:
    tfidf_vec = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf_vec.fit_transform(texts)
    tfidf_matrix = normalize(tfidf_matrix)
    print("[info] TF-IDF matrix built as BM25 fallback.")

print("hybrid")
def retrieve_hybrid(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    """Return top-k chunk dicts by hybrid score (vector similarity + BM25/TF-IDF)."""
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-12)
    if _use_faiss:
        D, I = index.search(q_emb, min(k*5, len(texts))) 
        cand_idxs = I[0].tolist()
        vec_sims = {i: D[0][j] for j, i in enumerate(cand_idxs)}
    else:
        sims = (embeddings @ q_emb.T).squeeze()
        cand_idxs = list(np.argsort(-sims)[:min(200, len(sims))])
        vec_sims = {i: float(sims[i]) for i in cand_idxs}

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

    combined = []
    for i in cand_idxs:
        v = vec_sims.get(i, 0.0)
        t = text_scores.get(i, 0.0)
        score = VECTOR_WEIGHT * v + TEXT_WEIGHT * t
        combined.append((i, score))
    combined.sort(key=lambda x: x[1], reverse=True)
    top = [idx for idx, _ in combined[:k]]
    return [chunks[i] for i in top]

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
    for k in ents:
        ents[k] = list(dict.fromkeys(ents[k]))
    return ents

def compute_threat_score(entities: Dict[str, List[str]], context_text: str) -> int:
    score = 0
    sev = entities.get("severity", [])
    if "critical" in sev:
        score += 40
    elif "high" in sev:
        score += 25
    elif "medium" in sev:
        score += 10
    if entities.get("ips"):
        score += 10
    if entities.get("mitre"):
        score += 10
    kws = ["powershell", "encoded", "ransomware", "brute-force", "credential dumping", "suspicious outbound"]
    for kw in kws:
        if kw in context_text.lower():
            score += 5
    return min(max(score, 0), 100)

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

print("memory")
_sessions_hist: Dict[str, InMemoryChatMessageHistory] = {}
_sessions_entity_memory: Dict[str, Dict[str, List[str]]] = {}

def get_session_history(session_id: str):
    if session_id not in _sessions_hist:
        _sessions_hist[session_id] = InMemoryChatMessageHistory()
    return _sessions_hist[session_id]

memory_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)



print("\n=== SOC RAG Assistant ===")
print("Type 'quit' as user_id or query to exit.\n")

while True:
    try:
        analyst = input("analyst_id> ").strip()
        if analyst.lower() in ("quit", "exit","q"):
            break
        query = input("query> ").strip()
        if query.lower() in ("quit", "exit"):
            break

        hits = retrieve_hybrid(query, k=TOP_K)
        retrieved_text = "\n---\n".join([f"Incident[{h['meta']['orig_line']}]: {h['text']}" for h in hits])

        ents_q = extract_entities(query)
        ents_ctx = extract_entities(retrieved_text)
        combined_entities = {
            "ips": list(dict.fromkeys(ents_q["ips"] + ents_ctx["ips"])),
            "os": list(dict.fromkeys(ents_q["os"] + ents_ctx["os"])),
            "hosts": list(dict.fromkeys(ents_q["hosts"] + ents_ctx["hosts"])),
            "mitre": list(dict.fromkeys(ents_q["mitre"] + ents_ctx["mitre"])),
            "severity": list(dict.fromkeys(ents_q["severity"] + ents_ctx["severity"]))
        }

        if analyst not in _sessions_entity_memory:
            _sessions_entity_memory[analyst] = {"ips": [], "os": [], "hosts": [], "mitre": [], "severity": []}
        for k in _sessions_entity_memory[analyst]:
            merged = list(dict.fromkeys(_sessions_entity_memory[analyst][k] + combined_entities.get(k, [])))
            _sessions_entity_memory[analyst][k] = merged

        threat_score = compute_threat_score(combined_entities, retrieved_text + "\n" + query)

        entities_str = json.dumps(_sessions_entity_memory[analyst], ensure_ascii=False)
        input_obj = {"question": query, "entities": entities_str, "prefs": "", "history": ""}  # history injected by wrapper
        config = {"configurable": {"session_id": analyst}}

        result = memory_chain.invoke(input_obj, config)
        assistant_text = str(result)

        tool_calls = getattr(result, "tool_calls", None)
        if tool_calls is None and isinstance(result, dict):
            tool_calls = result.get("tool_calls")
        tool_info = None
        if tool_calls:
            call = tool_calls[0]
            name = call.get("name") if isinstance(call, dict) else None
            args = call.get("args") if isinstance(call, dict) else None
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



