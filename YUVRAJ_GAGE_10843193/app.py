# ================================================================
# SOC RAG Assistant - Complete Implementation
# ================================================================

# ===========
# STEP 0: IMPORTS
# ===========
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
import re
import json

# LLM Setup
llm = ChatOllama(model="mistral", temperature=0.3)


# ===========
# STEP 1: LOAD DATA
# ===========
print("\n=== STEP 1: LOAD DATA ===")
loader = TextLoader("security.csv")
docs = loader.load()
print(f"✓ Loaded {len(docs)} document(s)")
print(f"✓ Sample (first 200 chars): {docs[0].page_content[:200]}...\n")


# ===========
# STEP 2: CHUNK DATA
# ===========
print("\n=== STEP 2: CHUNKING ===")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", "|", " ", ""]
)
splits = splitter.split_documents(docs)
print(f"✓ Created {len(splits)} chunks")
print(f"✓ Sample chunk: {splits[0].page_content[:150]}...\n")


# ===========
# STEP 3: EMBEDDINGS + FAISS
# ===========
print("\n=== STEP 3: EMBEDDINGS + INDEX ===")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, emb)
vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print(f"✓ FAISS index built with {len(splits)} vectors")
print(f"✓ Vector retriever ready (k=4)\n")


# ===========
# BONUS: HYBRID RETRIEVAL
# ===========
print("\n=== BONUS: HYBRID RETRIEVAL ===")
bm25_retriever = BM25Retriever.from_documents(splits)
bm25_retriever.k = 4

hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.7, 0.3]
)
print("✓ Hybrid retriever ready (Vector: 0.7, BM25: 0.3)\n")


# ===========
# BONUS: ENTITY EXTRACTION
# ===========
def extract_entities(query: str, context: str = ""):
    """Extract IPs, hostname, OS, MITRE tags, severity from query and context"""
    combined_text = f"{query} {context}"
    entities = {}
    
    # Extract IP addresses
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    ips = re.findall(ip_pattern, combined_text)
    if ips:
        entities['IPs'] = list(set(ips))
    
    # Extract MITRE tags
    mitre_pattern = r'T\d{4}(?:\.\d{3})?'
    mitre_tags = re.findall(mitre_pattern, combined_text)
    if mitre_tags:
        entities['MITRE'] = list(set(mitre_tags))
    
    # Extract OS
    os_pattern = r'(?:Ubuntu|Windows|CentOS|RedHat|Debian)\s*\d*'
    os_matches = re.findall(os_pattern, combined_text, re.IGNORECASE)
    if os_matches:
        entities['OS'] = list(set(os_matches))
    
    # Extract Hostname
    host_pattern = r'Host=([A-Z0-9\-]+)|(?:SRV|WKS|LAB|DEV|WEB|FIN)\-[A-Z0-9\-]+'
    hosts = re.findall(host_pattern, combined_text)
    hosts = [h[0] if isinstance(h, tuple) else h for h in hosts]
    if hosts:
        entities['Hostnames'] = list(set([h for h in hosts if h]))
    
    # Extract Severity
    severity_pattern = r'Severity=(Low|Medium|High|Critical)'
    severities = re.findall(severity_pattern, combined_text, re.IGNORECASE)
    if severities:
        entities['Severity'] = list(set(severities))
    
    return entities


# ===========
# BONUS: THREAT SCORE
# ===========
def calculate_threat_score(query: str, context: str, entities: dict):
    """Calculate threat score based on indicators"""
    score = 0
    reasons = []
    
    # Check severity
    if 'Severity' in entities:
        if 'Critical' in str(entities['Severity']):
            score += 40
            reasons.append("Critical severity (+40)")
        elif 'High' in str(entities['Severity']):
            score += 30
            reasons.append("High severity (+30)")
        elif 'Medium' in str(entities['Severity']):
            score += 15
            reasons.append("Medium severity (+15)")
    
    # Check MITRE tags - high-risk techniques
    high_risk_mitre = ['T1486', 'T1003', 'T1059', 'T1071', 'T1110']
    if 'MITRE' in entities:
        for tag in entities['MITRE']:
            if any(risk in tag for risk in high_risk_mitre):
                score += 20
                reasons.append(f"High-risk MITRE {tag} (+20)")
                break
    
    # Check for suspicious keywords
    suspicious_keywords = ['ransomware', 'powershell', 'brute-force', 'malicious', 
                          'credential', 'dump', 'C2', 'beacon', 'encryption']
    combined = f"{query} {context}".lower()
    for keyword in suspicious_keywords:
        if keyword in combined:
            score += 10
            reasons.append(f"Suspicious keyword '{keyword}' (+10)")
            break
    
    # Check for malicious IPs (mock check)
    if 'IPs' in entities:
        for ip in entities['IPs']:
            if ip.startswith('10.') or ip.startswith('172.'):
                score += 5
                reasons.append(f"Internal IP in alert (+5)")
                break
    
    return min(score, 100), reasons


# ===========
# BONUS: TOOL
# ===========
print("\n=== BONUS: TOOL ===")

@tool
def threat_enrich(ip: str):
    """Enrich IP with threat intelligence data"""
    # Mock threat intelligence
    threat_db = {
        "10.1.1.9": "Known brute-force source | Last seen: 2 days ago | Risk: High",
        "10.2.4.9": "SSH attack origin | Geolocation: Unknown | Risk: High",
        "172.22.9.54": "Port scanning activity | ASN: Internal | Risk: Medium",
        "10.22.3.9": "C2 communication detected | Blacklisted: Yes | Risk: Critical",
        "192.168.55.20": "Blacklisted country origin | Risk: High"
    }
    
    return threat_db.get(ip, f"No threat intel available for {ip} | Status: Clean")

llm_with_tools = llm.bind_tools([threat_enrich])
print("✓ Threat enrichment tool ready\n")


# ===========
# STEP 4: PROMPT + LCEL RAG CHAIN
# ===========
print("\n=== STEP 4: RAG CHAIN ===")

prompt = ChatPromptTemplate.from_template("""You are an expert SOC analyst assistant. Analyze the security incident and provide actionable recommendations.

**Retrieved Context from Past Incidents:**
{context}

**Extracted Entities:**
{entities}

**Conversation History:**
{history}

**Current Query:**
{question}

Provide a comprehensive analysis including:
1. Incident summary and severity assessment
2. Similar past incidents and their resolutions
3. Recommended immediate actions
4. Long-term mitigation steps
5. Relevant MITRE ATT&CK techniques

Be concise but thorough. Focus on actionable insights.
""")

parser = StrOutputParser()

# Helper function to format retrieved documents
def format_docs(docs):
    return "\n\n".join([f"[Incident {i+1}] {doc.page_content}" for i, doc in enumerate(docs)])

# Helper function to format entities
def format_entities(entities):
    if not entities:
        return "None detected"
    return "\n".join([f"- {k}: {', '.join(map(str, v)) if isinstance(v, list) else v}" 
                      for k, v in entities.items()])

# Build RAG chain
def process_input(input_dict):
    """Process input and extract entities"""
    query = input_dict["question"]
    docs = hybrid_retriever.invoke(query)
    context = format_docs(docs)
    entities = extract_entities(query, context)
    
    return {
        "context": context,
        "question": query,
        "entities": format_entities(entities),
        "history": input_dict.get("history", ""),
        "_raw_entities": entities,
        "_raw_context": context
    }

def convertLLMOutputToJSON(output: str) -> dict:
    """Convert LLM output string to JSON dict"""
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        return {"response": output}

chain = (
    RunnableLambda(process_input)
    | RunnablePassthrough.assign(
        answer=lambda x: (prompt | llm | parser).invoke({
            "context": x["context"],
            "question": x["question"],
            "entities": x["entities"],
            "history": x["history"]
        })
    )
)

print("✓ RAG chain constructed\n")


# ===========
# STEP 5: MEMORY (RunnableWithMessageHistory)
# ===========
print("\n=== STEP 5: MEMORY ===")

store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

memory_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)

print("✓ Memory wrapper ready\n")


# ===========
# STEP 7: INTERACTIVE CONSOLE LOOP
# ===========
print("\n" + "="*70)
print("🛡️  SOC RAG ASSISTANT - INCIDENT INVESTIGATION SYSTEM")
print("="*70)
print("Format: <analyst_id> <query>")
print("Example: analyst42 suspicious ssh activity on ubuntu server")
print("Type 'q' to quit")
print("="*70 + "\n")

while True:
    raw = input("🔍 Enter: ").strip()
    if raw.lower() == "q":
        print("\n✓ Exiting SOC Assistant. Stay vigilant! 🛡️\n")
        break

    if raw == "reset":
        print("🔄 Resetting conversation...")
        memory_chain.reset()
        continue

    if raw == "json":
        #give output in json format
        print("🔄 Switching to JSON output mode...")
        memory_chain = memory_chain.map_output(convertLLMOutputToJSON)
        continue

   

    parts = raw.split(" ", 1)
    if len(parts) < 2:
        print("❌ Format: analyst42 suspicious ssh activity\n")
        continue

    analyst_id, query = parts
    
    print(f"\n{'='*70}")
    print(f"📊 Analyst: {analyst_id} | Query: {query}")
    print(f"{'='*70}\n")
    
    try:
        # Invoke the memory-wrapped RAG chain
        config = {"configurable": {"session_id": analyst_id}}
        response = memory_chain.invoke({"question": query}, config=config)
        
        # Display entities
        if response.get("_raw_entities"):
            print("🔎 **EXTRACTED ENTITIES:**")
            for k, v in response["_raw_entities"].items():
                print(f"   • {k}: {', '.join(map(str, v)) if isinstance(v, list) else v}")
            print()
        
        # Calculate threat score
        score, reasons = calculate_threat_score(
            query, 
            response.get("_raw_context", ""), 
            response.get("_raw_entities", {})
        )
        print(f"⚠️  **THREAT SCORE: {score}/100**")
        if reasons:
            for reason in reasons:
                print(f"   • {reason}")
        print()
        
        # Display main response
        print("🤖 **SOC ASSISTANT ANALYSIS:**")
        print("-" * 70)
        print(response["answer"])

        #print in json format
        print(json.dumps(convertLLMOutputToJSON(response["answer"]), indent=4))
        print("-" * 70)
        
        # Tool enrichment (if IPs detected)
        if response.get("_raw_entities", {}).get("IPs"):
            print("\n🌐 **THREAT ENRICHMENT:**")
            for ip in response["_raw_entities"]["IPs"][:2]:  # Limit to 2 IPs
                intel = threat_enrich.invoke({"ip": ip})
                print(f"   • {ip}: {intel}")
        
        print("\n" + "="*70 + "\n")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}\n")
        continue