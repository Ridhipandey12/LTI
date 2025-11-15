# ================================================================
# SOC RAG Assistant - Complete Implementation
# ================================================================

# ===========
# STEP 0: IMPORTS (DO NOT REMOVE — Add only if needed)
# ===========
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
import re
import json
import redis

# LLM Setup (Students keep as-is)
llm = ChatOllama(model="mistral", temperature=0.3)

redis_client = redis.Redis(host='localhost', port=6379, db=0)

# ===========
# STEP 1: LOAD DATA
# ===========
print("\n=== STEP 1: LOAD DATA ===")
# Load security_incidents.txt using TextLoader
loader = TextLoader("security_incidents.txt")
docs = loader.load()
print(f"Loaded {len(docs)} document(s)")
print(f"Sample content (first 200 chars):\n{docs[0].page_content[:200]}...\n")


# ===========
# STEP 2: CHUNK DATA
# ===========
print("\n=== STEP 2: CHUNKING ===")
# Implement RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
splits = splitter.split_documents(docs)
print(f"Created {len(splits)} chunks")
print(f"Sample chunk:\n{splits[0].page_content[:150]}...\n")


# ===========
# STEP 3: EMBEDDINGS + FAISS
# ===========
print("\n=== STEP 3: EMBEDDINGS + INDEX ===")
# Create embeddings using MiniLM-L6-v2
emb = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embeddings model loaded")

# Create FAISS vectorstore
vectorstore = FAISS.from_documents(splits, emb)
print("FAISS index created")

# Create vector retriever (k=4)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("Vector retriever ready (k=4)\n")


# ===========
# BONUS: HYBRID RETRIEVAL
# ===========
print("\n=== BONUS: HYBRID RETRIEVAL ===")
# Create BM25 retriever
bm25 = BM25Retriever.from_documents(splits)
bm25.k = 4

# Create EnsembleRetriever with weights [0.7, 0.3]
hybrid = EnsembleRetriever(
    retrievers=[vector_retriever, bm25],
    weights=[0.7, 0.3]
)
print("Hybrid retriever ready (FAISS: 0.7, BM25: 0.3)\n")


# ===========
# STEP 6: ENTITY EXTRACTION (BONUS)
# ===========
def extract_entities(query: str, context: str):
    """Extract IPs, hostname, OS, MITRE tags, severity from query and context"""
    combined_text = f"{query} {context}"
    entities = {}
    
    # Extract IP addresses
    ip_pattern = r'\b(?:d{1, 3}\.){3}\d{1, 3}\b'
    ips = re.findall(ip_pattern, combined_text)
    if ips:
        entities['ips'] = list(set(ips))

    # Extract MITRE tags (Txxxx format)
    mitre_pattern = r'T\d{4}'
    mitre_tags = re.findall(mitre_pattern, combined_text)
    if mitre_tags:
        entities['mitre_tags'] = list(set(mitre_tags))

    # Extract OS (common patterns)
    os_keywords = ['Windows', 'Linux', 'Ubuntu', 'CentOS', 'macOS', 'Debian']
    found_os = [os for os in os_keywords if os.lower() in combined_text.lower()]
    if found_os:
        entities['os'] = list(set(found_os))
    
    # Extract Severity
    severity_pattern = r'\b(Low|Medium|High|Critical)\b'
    severities = re.findall(severity_pattern, combined_text, re.IGNORECASE)
    if severities:
        entities['severity'] = list(set([s.capitalize() for s in severities]))
    
    # Extract hostnames (basic pattern)
    hostname_pattern = r'\b[a-zA-Z0-9]+-[a-zA-Z0-9]+\b|\b[a-zA-Z0-9]+\.[a-zA-Z0-9]+\.[a-zA-Z]{2,}\b'
    hostnames = re.findall(hostname_pattern, combined_text)
    # Filter out IPs from hostnames
    hostnames = [h for h in hostnames if not re.match(ip_pattern, h)]
    if hostnames:
        entities['hostnames'] = list(set(hostnames))[:3]  # Limit to 3
    
    return entities


# ===========
# BONUS: THREAT SCORE
# ===========
def calculate_threat_score(entities: dict, context: str):
    """Calculate threat score based on severity, MITRE tags, and indicators"""
    score = 0
    reasons = []
    
    # Severity scoring
    if 'severity' in entities:
        severity_map = {'Low': 10, 'Medium': 25, 'High': 50, 'Critical': 75}
        for sev in entities['severity']:
            score += severity_map.get(sev, 0)
            reasons.append(f"Severity: {sev}")
    
    # MITRE tag presence
    if 'mitre_tags' in entities:
        score += len(entities['mitre_tags']) * 15
        reasons.append(f"MITRE techniques detected: {', '.join(entities['mitre_tags'])}")
    
    # Suspicious keywords
    suspicious_keywords = ['powershell', 'brute-force', 'malware', 'ransomware', 
                          'credential', 'dump', 'suspicious', 'unauthorized']
    for keyword in suspicious_keywords:
        if keyword.lower() in context.lower():
            score += 10
            if keyword not in [r.split(':')[0].lower() for r in reasons]:
                reasons.append(f"Suspicious activity: {keyword}")
    
    # Cap score at 100
    score = min(score, 100)
    
    return {"threat_score": score, "reasons": reasons}


# ===========
# BONUS: TOOL
# ===========
print("\n=== BONUS: TOOL ===")

@tool
def threat_enrich(ip: str):
    """Return mock threat intel for IP address"""
    # Mock threat intelligence database
    threat_db = {
        "192.168.1.100": "Known malicious IP - Associated with brute-force attacks",
        "10.0.0.50": "Suspicious IP - Multiple failed login attempts",
        "172.16.0.200": "Clean IP - No threat indicators found",
    }
    
    result = threat_db.get(ip, f"No specific threat intel for {ip} - Recommend further investigation")
    return f"Threat Intel for {ip}: {result}"

# Bind tool to LLM
llm_with_tools = llm.bind_tools([threat_enrich])
print("Threat enrichment tool ready\n")


# ===========
# STEP 4: PROMPT + LCEL RAG CHAIN
# ===========
print("\n=== STEP 4: RAG CHAIN ===")

# Build prompt template with context, question, entities, history
prompt = ChatPromptTemplate.from_template("""
You are an expert SOC (Security Operations Center) analyst assistant. Your role is to help analysts investigate security incidents by providing relevant information from past incidents and actionable recommendations.

## Retrieved Context from Similar Past Incidents:
{context}

## Extracted Entities:
{entities}

## Conversation History:
{history}

## Current Query:
{question}

## Instructions:
1. Analyze the retrieved context and identify similar past incidents
2. Consider the extracted entities (IPs, MITRE tags, OS, severity) in your analysis
3. Provide clear, actionable recommendations for the SOC analyst
4. Reference specific past incidents when relevant
5. If IP addresses are mentioned, suggest using the threat_enrich tool for additional intelligence
6. Be concise but comprehensive - analysts need quick, accurate answers

Provide your analysis and recommendations:
""")

parser = StrOutputParser()

# Build LCEL runnable chain
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def get_entities_from_query(inputs):
    """Extract entities from query and retrieved context"""
    query = inputs.get("question", "")
    docs = inputs.get("context", [])
    context = format_docs(docs) if docs else ""
    entities = extract_entities(query, context)
    return json.dumps(entities, indent=2) if entities else "No entities extracted"

def get_history_string(inputs):
    """Format chat history as string"""
    history = inputs.get("history", [])
    if not history:
        return "No previous conversation"
    
    history_str = []
    for msg in history[-6:]:  # Last 6 messages (3 turns)
        if isinstance(msg, HumanMessage):
            history_str.append(f"Analyst: {msg.content}")
        elif isinstance(msg, AIMessage):
            history_str.append(f"Assistant: {msg.content}")
    return "\n".join(history_str) if history_str else "No previous conversation"

chain = (
    {
        "context": lambda x: hybrid.invoke(x["question"]),
        "question": lambda x: x["question"],
        "entities": lambda x: get_entities_from_query({
            "question": x["question"],
            "context": hybrid.invoke(x["question"])
        }),
        "history": lambda x: get_history_string(x)
    }
    | prompt
    | llm
    | parser
)
print("RAG chain constructed\n")


# ===========
# STEP 5: MEMORY (RunnableWithMessageHistory)
# ===========
print("\n=== STEP 5: MEMORY ===")

store = {}  # DO NOT CHANGE

def get_session_history(session_id: str):
    """Create InMemoryChatMessageHistory if missing, return history object"""
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Wrap chain with RunnableWithMessageHistory
memory_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)
print("Memory wrapper ready\n")


print("\n==== PERSISTENCE - SAVE INDEX ===")
vectorstore.save_local("security_incidents_db")  # What: Disk save. Why: Survives restarts. How: FAISS folder (index.faiss + index.pkl).
print("Index saved to 'security_incidents_db'.")

# ===========
# STEP 7: INTERACTIVE CONSOLE LOOP
# ===========
print("\n" + "="*70)
print("SOC RAG ASSISTANT - INCIDENT INVESTIGATION SYSTEM")
print("="*70)
print("Format: <a_id> <query>")
print("Example: a42 suspicious ssh activity from 192.168.1.100")
print("Type 'q' to quit")
print("="*70 + "\n")

while True:
    raw = input("🔍 Enter: ")
    if raw.strip().lower() == "q":
        print("\n👋 Exiting SOC Assistant. Stay secure!")
        break
    
    parts = raw.split(" ", 1)
    if len(parts) < 2:
        print("⚠️  Format: a42 suspicious ssh activity")
        continue

    a_id, query = parts
    print(f"\n{'='*70}")
    print(f"Analyst: {a_id}")
    print(f"Query: {query}")
    print(f"{'='*70}\n")

    try:
        # Retrieve context for entity extraction
        retrieved_docs = hybrid.invoke(query)
        context = format_docs(retrieved_docs)
        
        # Extract entities
        entities = extract_entities(query, context)
        
        if entities:
            print("EXTRACTED ENTITIES:")
            for key, values in entities.items():
                print(f"   • {key.upper()}: {', '.join(map(str, values))}")
            
            # Calculate threat score
            threat_info = calculate_threat_score(entities, context)
            print(f"\nTHREAT SCORE: {threat_info['threat_score']}/100")
            if threat_info['reasons']:
                print("   Reasons:")
                for reason in threat_info['reasons'][:5]:
                    print(f"   • {reason}")
            print()
        
        # Invoke the memory-wrapped RAG chain
        response = memory_chain.invoke(
            {"question": query},
            config={"configurable": {"session_id": a_id}}
        )
        
        # Print the response
        print("ASSISTANT RESPONSE:")
        print("-" * 70)
        print(response)
        print("-" * 70)
        
        # Suggest threat enrichment if IPs found
        if entities.get('ips'):
            print(f"\n💡 Tip: You can enrich IPs using threat intelligence")
            print(f"   Available IPs: {', '.join(entities['ips'])}")
        
        print("\n")

        # show caching info
        cache_key = f"rag:{query}"
        cached = redis_client.get(cache_key)
        if cached:
            print("Cache Hit:", cached.decode())
        else:
            redis_client.setex(cache_key, 3600, response)
        print("Cache updated.")
        print("Cache key:", cache_key)
        print("Cached response:", redis_client.get(cache_key).decode())
        print("Caching done.")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Please try again.\n")