# SOC RAG Assistant
 
# STEP 0: IMPORTS
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
import re
import json
from typing import List
from operator import itemgetter

# LLM
llm = ChatOllama(model="mistral", temperature=0.3)

# STEP 1: LOAD DATA

print("\n=== STEP 1: LOAD DATA ===")
loader = TextLoader("security_incidents.txt", encoding="utf-8")
raw_docs = loader.load()

docs: List = []
for line in raw_docs[0].page_content.strip().splitlines():
    if not line.strip():
        continue
    parts = [p.strip() for p in line.split("|")]
    if len(parts) != 8:
        continue
    metadata = {
        "user": parts[0],
        "hostname": parts[1],
        "os": parts[2],
        "source_ip": parts[3],
        "summary": parts[4],
        "mitre": parts[5],
        "severity": parts[6],
        "resolution": parts[7],
    }
    content = (
        f"User:{metadata['user']} Host:{metadata['hostname']} OS:{metadata['os']} "
        f"IP:{metadata['source_ip']} Summary:{metadata['summary']} "
        f"MITRE:{metadata['mitre']} Severity:{metadata['severity']} "
        f"Resolution:{metadata['resolution']}"
    )
    docs.append(type("Document", (), {"page_content": content, "metadata": metadata})())

print(f"Loaded {len(docs)} incidents.")
print("Sample →", docs[0].page_content[:200] + "..." if docs else "No data")



# STEP 2: CHUNKING

print("\n=== STEP 2: CHUNKING ===")
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50, length_function=len)
splits = splitter.split_documents(docs)
print(f"Created {len(splits)} chunks.")



# STEP 3: EMBEDDINGS + FAISS

print("\n=== STEP 3: EMBEDDINGS + INDEX ===")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, emb)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})



# BONUS: HYBRID RETRIEVAL

print("\n=== BONUS: HYBRID RETRIEVAL ===")
bm25 = BM25Retriever.from_documents(splits)
bm25.k = 4

hybrid = EnsembleRetriever(
    retrievers=[vector_retriever, bm25],
    weights=[0.7, 0.3]
)
print("Hybrid retriever ready.\n")



# HELPERS

def format_docs(docs):
    return "\n\n".join([f"---\nIncident: {d.page_content}" for d in docs])


# ---------- ENTITY EXTRACTION ----------
def extract_entities(query: str, context: str) -> dict:
    """Extract IPs, OS, hostnames, MITRE tags, severity from query+context."""
    text = f"{query}\n{context}"

    # IP addresses
    ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)

    # OS (Windows|Linux|macOS)
    os_list = re.findall(r"\b(Windows|Linux|macOS)\b", text, re.I)

    # Hostnames – now captures everything after "Host:" (including hyphens & numbers)
    hostnames = re.findall(r"Host:([A-Za-z0-9-]+)", text)

    # MITRE technique IDs
    mitre = re.findall(r"\b(T\d{4}(?:\.\d+)?)\b", text)

    # Severity
    severity = re.findall(r"\b(Low|Medium|High|Critical)\b", text, re.I)

    return {
        "ips": list(set(ips)),          # dedup
        "os": list(set(os_list)),
        "hostnames": list(set(hostnames)),
        "mitre": list(set(mitre)),
        "severity": list(set(severity)),
    }


# ---------- THREAT SCORE ----------
def calculate_threat_score(entities: dict, context: str) -> int:
    score = 0
    sev_map = {"Low": 1, "Medium": 3, "High": 6, "Critical": 10}
    for s in entities.get("severity", []):
        score += sev_map.get(s.capitalize(), 0)

    critical_mitre = {"T1486", "T1566", "T1078", "T1059"}
    for m in entities.get("mitre", []):
        if m.split(".")[0] in critical_mitre:
            score += 5

    bad_ips = {"185.123.45.67", "198.51.100.22"}
    if any(ip in bad_ips for ip in entities.get("ips", [])):
        score += 7

    keywords = ["PowerShell", "brute-force", "credential dumping", "ransomware"]
    ctx_lower = context.lower()
    score += sum(4 for kw in keywords if kw.lower() in ctx_lower)

    return min(score, 100)



# PROMPT

prompt = ChatPromptTemplate.from_template("""
You are a senior SOC analyst assistant.

**Retrieved Context**
{context}

**Extracted Entities**
{entities}

**Conversation History**
{history}

**Query:** {question}

Provide a concise technical answer with:
- analysis
- recommended resolution steps
- similar past incident IDs (if any)
- threat score (0-100)

Return **valid JSON** with keys: analysis, resolution, similar_incidents, threat_score
""")



# CORE RAG CHAIN (includes history)

core_chain = (
    {
        "question": itemgetter("question"),
        "context": itemgetter("question") | hybrid | format_docs,
        "history": itemgetter("history"),
    }
    | RunnableLambda(lambda x: {
        **x,
        "entities": extract_entities(x["question"], x["context"]),
    })
    | RunnableLambda(lambda x: {
        **x,
        "threat_score": calculate_threat_score(
            x["entities"],
            "\n".join([d.page_content for d in hybrid.invoke(x["question"])])
        ),
    })
    | prompt
    | llm
    | JsonOutputParser()
)

print("RAG chain constructed.\n")



# BONUS: TOOL – THREAT ENRICHMENT

@tool
def threat_enrich(ip: str) -> str:
    """Mock threat intel for a given IP."""
    known = {
        "185.123.45.67": "Known brute-force C2 (SSH)",
        "198.51.100.22": "Emotet dropper, high risk",
    }
    return known.get(ip, "No known reputation")



# MEMORY

store: dict = {}

def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

memory_chain = RunnableWithMessageHistory(
    core_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

print("Memory wrapper ready.\n")



# INTERACTIVE LOOP

print("\n=== STEP 7: CONSOLE LOOP ===")
print("Format: <analyst_id> <your query>   (type 'q' to quit)\n")

while True:
    raw = input("Enter: ").strip()
    if raw.lower() == "q":
        break

    parts = raw.split(" ", 1)
    if len(parts) < 2:
        print("Error: Please use: analyst42 suspicious SSH activity")
        continue

    analyst_id, query = parts[0], parts[1]

    # Retrieve for display
    retrieved_docs = hybrid.invoke(query)
    context_str = format_docs(retrieved_docs)
    entities = extract_entities(query, context_str)

    # Invoke chain
    try:
        response = memory_chain.invoke(
            {"question": query},
            config={"configurable": {"session_id": analyst_id}}
        )
    except Exception as e:
        print(f"Chain error: {e}")
        continue

    # ---------- OUTPUT ----------
    print("\n" + "="*60)
    print("FINAL ANSWER (JSON)")
    print(json.dumps(response, indent=2))

    print("\n--- Retrieved Context (Hybrid) ---")
    for i, doc in enumerate(retrieved_docs, 1):
        print(f"{i}. {doc.page_content[:300]}...")

    print("\n--- Extracted Entities ---")
    print(json.dumps(entities, indent=2))

    print("\n--- Conversation History (last 2 turns) ---")
    hist = get_session_history(analyst_id)
    for msg in hist.messages[-4:]:
        role = "USER" if isinstance(msg, HumanMessage) else "AI"
        print(f"{role}: {msg.content[:120]}...")

    print("\n--- Threat Score ---")
    print(f"{response.get('threat_score', 'N/A')}/100")

    if entities.get("ips"):
        ip = entities["ips"][0]
        enrich = threat_enrich.invoke({"ip": ip})
        print(f"\n--- Threat Enrichment for {ip} ---")
        print(enrich)

    print("="*60 + "\n")