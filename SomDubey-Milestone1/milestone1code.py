# ================================================================
# SOC RAG Assistant - Assessment Template (Students: Fill TODOs)
# ================================================================

# ===========
# STEP 0: IMPORTS (DO NOT REMOVE — Add only if needed)
# ===========
import warnings
import os

# Suppress LangSmith UUID warning
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.v1.main")
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
import re

# LLM Setup (Students keep as-is)
llm = ChatOllama(model="mistral", temperature=0.3)


# ===========
# STEP 1: LOAD DATA
# ===========
print("\n=== STEP 1: LOAD DATA ===")
# TODO: Load security_incidents.txt using TextLoader
# TODO: Print doc count + sample record
loader = TextLoader("security_incidents.txt")
# STUDENT IMPLEMENTATION:
docs = loader.load()
print(f"Loaded {len(docs)} document(s)")
print(f"Sample content (first 200 chars): {docs[0].page_content[:200]}")


# ===========
# STEP 2: CHUNK DATA
# ===========
print("\n=== STEP 2: CHUNKING ===")
# TODO: Implement RecursiveCharacterTextSplitter
# TODO: Produce splits
# STUDENT IMPLEMENTATION:
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
splits = splitter.split_documents(docs)
print(f"Created {len(splits)} chunks")
print(f"Sample chunk: {splits[0].page_content[:150]}")


# ===========
# STEP 3: EMBEDDINGS + FAISS
# ===========
print("\n=== STEP 3: EMBEDDINGS + INDEX ===")
# TODO: Create embeddings using MiniLM-L6-v2
# TODO: Create FAISS vectorstore
# TODO: Create vector retriever (k=4)
# STUDENT IMPLEMENTATION:
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, emb)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
print("FAISS vectorstore created with vector retriever (k=4)")


# ===========
# BONUS: HYBRID RETRIEVAL
# ===========
print("\n=== BONUS: HYBRID RETRIEVAL ===")
# TODO: Create BM25 retriever
# TODO: Create EnsembleRetriever with weights [0.7, 0.3]
# STUDENT IMPLEMENTATION:
bm25 = BM25Retriever.from_documents(splits)
bm25.k = 4
hybrid = EnsembleRetriever(
    retrievers=[retriever, bm25],
    weights=[0.7, 0.3]
)
print("Hybrid retriever ready.\n")


# ===========
# STEP 4: PROMPT + LCEL RAG CHAIN
# ===========
print("\n=== STEP 4: RAG CHAIN ===")

# TODO: Build prompt template (you must include placeholders for context, question, entities, history)
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a Security Operations Center (SOC) analyst assistant. Your role is to help analysts investigate security incidents using the provided context.

Context from Security Incidents Database:
{context}

Extracted Entities:
{entities}

Instructions:
- Provide clear, concise answers based on the context
- Reference specific incident numbers when applicable
- Highlight MITRE ATT&CK techniques, severity levels, and resolutions
- If the context doesn't contain relevant information, say so
- Maintain professional security analyst tone"""),
    ("placeholder", "{history}"),
    ("human", "{question}")
])

parser = StrOutputParser()

# TODO: Build LCEL runnable chain
# STUDENT IMPLEMENTATION:
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def format_entities(entities_dict):
    if isinstance(entities_dict, dict) and entities_dict:
        return "\n".join([f"- {k}: {v}" for k, v in entities_dict.items()])
    return "No entities extracted"

def get_question_string(x):
    """Extract question string from input"""
    if isinstance(x, dict):
        return x.get("question", "")
    return str(x)

chain = {
    "context": lambda x: format_docs(hybrid.invoke(get_question_string(x))),
    "question": get_question_string,
    "entities": lambda x: format_entities(x.get("entities", {}) if isinstance(x, dict) else {}),
} | prompt | llm | parser

print("RAG chain constructed.\n")


# ===========
# BONUS: TOOL
# ===========
print("\n=== BONUS: TOOL ===")

@tool
def threat_enrich(ip: str):
    """Students: Return mock threat intel for IP"""
    # TODO: Implement enrichment logic
    mock_threat_db = {
        "10.1.1.9": "Known SSH brute-force source | Reputation: Malicious | First seen: 2024-01",
        "10.2.4.9": "SSH attack source | ASN: AS12345 | Country: Unknown",
        "172.22.9.54": "Port scanning activity detected | Threat Score: 8/10",
        "10.22.3.9": "C2 communication IP | Malware family: Cobalt Strike",
        "192.168.55.20": "Blacklisted country source | VPN exit node suspected"
    }
    return mock_threat_db.get(ip, f"No threat intelligence found for IP: {ip}")

# TODO: Bind tool to LLM
llm_with_tools = llm.bind_tools([threat_enrich])


# ===========
# STEP 5: MEMORY (RunnableWithMessageHistory)
# ===========
print("\n=== STEP 5: MEMORY ===")

store = {}  # DO NOT CHANGE

def get_session_history(session_id: str):
    # TODO: Create InMemoryChatMessageHistory if missing
    # TODO: Return the history object
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# TODO: Wrap chain with RunnableWithMessageHistory
memory_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history"
)
print("Memory wrapper ready.\n")


# ===========
# STEP 6: ENTITY EXTRACTION (BONUS)
# ===========
def extract_entities(query: str, context: str):
    """Students: Extract IPs, hostname, OS, MITRE tags, severity"""
    # TODO: Your custom extraction logic
    entities = {}
    
    # Extract IPs
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    ips = re.findall(ip_pattern, query + " " + context)
    if ips:
        entities["IPs"] = ", ".join(set(ips))
    
    # Extract hostnames
    host_pattern = r'\b(?:SRV|WKS|LAB|DEV|WEB|FIN)-[A-Z0-9-]+\b'
    hosts = re.findall(host_pattern, query.upper() + " " + context.upper())
    if hosts:
        entities["Hosts"] = ", ".join(set(hosts))
    
    # Extract MITRE techniques
    mitre_pattern = r'\bT\d{4}(?:\.\d{3})?\b'
    mitre = re.findall(mitre_pattern, query + " " + context)
    if mitre:
        entities["MITRE"] = ", ".join(set(mitre))
    
    # Extract severity
    severity_pattern = r'\b(Critical|High|Medium|Low)\b'
    severity = re.findall(severity_pattern, query + " " + context, re.IGNORECASE)
    if severity:
        entities["Severity"] = ", ".join(set(severity))
    
    # Extract OS
    os_pattern = r'\b(Ubuntu|Windows|CentOS|RedHat|Debian)\s*\d*\b'
    os_list = re.findall(os_pattern, query + " " + context, re.IGNORECASE)
    if os_list:
        entities["OS"] = ", ".join(set(os_list))
    
    return entities


# ===========
# BONUS 3: THREAT SCORE (5 MARKS)
# ===========
def calculate_threat_score(entities: dict, context: str) -> dict:
    """
    Calculate threat score based on:
    - MITRE tag severity
    - Severity level
    - Presence of malicious IP
    - Suspicious PowerShell activity
    - Brute-force indicators
    """
    score = 0
    factors = []
    
    # High-risk MITRE techniques
    high_risk_mitre = {
        "T1486": 10,  # Ransomware
        "T1003": 9,   # Credential Dumping
        "T1071": 9,   # C2 Communication
        "T1059": 8,   # Command Execution
        "T1110": 7,   # Brute Force
        "T1562": 8,   # Disable AV
        "T1078": 7,   # Valid Accounts
    }
    
    # Check MITRE techniques
    if "MITRE" in entities:
        mitre_tags = entities["MITRE"].split(", ")
        for tag in mitre_tags:
            base_tag = tag.split(".")[0]  # T1059.001 -> T1059
            if base_tag in high_risk_mitre:
                score += high_risk_mitre[base_tag]
                factors.append(f"High-risk MITRE: {tag} (+{high_risk_mitre[base_tag]})")
    
    # Check Severity
    if "Severity" in entities:
        severity = entities["Severity"].lower()
        if "critical" in severity:
            score += 15
            factors.append("Critical severity (+15)")
        elif "high" in severity:
            score += 10
            factors.append("High severity (+10)")
        elif "medium" in severity:
            score += 5
            factors.append("Medium severity (+5)")
    
    # Check for malicious IPs (from threat_enrich mock data)
    malicious_ips = ["10.1.1.9", "10.2.4.9", "172.22.9.54", "10.22.3.9", "192.168.55.20"]
    if "IPs" in entities:
        ips = entities["IPs"].split(", ")
        for ip in ips:
            if ip in malicious_ips:
                score += 8
                factors.append(f"Known malicious IP: {ip} (+8)")
    
    # Check for PowerShell activity
    powershell_indicators = ["powershell", "empire", "iex", "encoded command"]
    for indicator in powershell_indicators:
        if indicator.lower() in context.lower():
            score += 6
            factors.append(f"PowerShell activity detected (+6)")
            break
    
    # Check for brute-force indicators
    bruteforce_indicators = ["brute-force", "brute force", "failed login", "failed ssh"]
    for indicator in bruteforce_indicators:
        if indicator.lower() in context.lower():
            score += 5
            factors.append(f"Brute-force indicator (+5)")
            break
    
    # Check for ransomware
    if "ransomware" in context.lower() or "encryption" in context.lower():
        score += 12
        factors.append("Ransomware indicator (+12)")
    
    # Determine threat level
    if score >= 30:
        threat_level = "CRITICAL"
    elif score >= 20:
        threat_level = "HIGH"
    elif score >= 10:
        threat_level = "MEDIUM"
    else:
        threat_level = "LOW"
    
    return {
        "score": score,
        "threat_level": threat_level,
        "factors": factors
    }


# ===========
# BONUS 4: STRUCTURED OUTPUT (5 MARKS)
# ===========
def parse_to_json(response: str, entities: dict, threat_score: dict) -> dict:
    """
    Convert LLM response to structured JSON format
    """
    structured_output = {
        "response": response,
        "entities": entities,
        "threat_analysis": {
            "score": threat_score["score"],
            "level": threat_score["threat_level"],
            "risk_factors": threat_score["factors"]
        },
        "metadata": {
            "incident_count": response.count("Incident #"),
            "has_resolution": "resolution" in response.lower() or "resolved" in response.lower(),
            "mitre_techniques": entities.get("MITRE", "None").split(", ") if "MITRE" in entities else []
        }
    }
    
    return structured_output


# ===========
# STEP 7: INTERACTIVE CONSOLE LOOP
# ===========
print("\n=== STEP 7: CONSOLE LOOP ===")
print("\n📋 Example queries:")
print("  analyst42 Show me all SSH brute force incidents")
print("  analyst42 What happened with user johns?")
print("  analyst42 Find Critical severity incidents")
print("  analyst42 Show me ransomware attacks")
print("  analyst42 What MITRE techniques are most common?")
print("  analyst42 Tell me about PowerShell attacks")

while True:
    raw = input("\nEnter: <analyst_id> <query> (or q): ")
    if raw.strip().lower() == "q":
        print("Exiting SOC RAG Assistant. Stay secure!")
        break
    
    parts = raw.split(" ", 1)
    if len(parts) < 2:
        print("❌ Format: analyst42 suspicious ssh activity")
        continue

    analyst_id, query = parts

    # TODO: Retrieve entities from query or context
    entities = extract_entities(query, "")

    # TODO: Invoke the memory-wrapped RAG chain
    try:
        print("\n⏳ Processing your query...")
        
        # Invoke with session-based memory
        response = memory_chain.invoke(
            {"question": query, "entities": entities},
            config={"configurable": {"session_id": analyst_id}}
        )
        
        # Get retrieved context for threat scoring
        retrieved_docs = hybrid.invoke(query)
        context = "\n".join([doc.page_content for doc in retrieved_docs])
        
        # Calculate threat score (BONUS 3)
        threat_score = calculate_threat_score(entities, context)
        
        # Create structured JSON output (BONUS 4)
        json_output = parse_to_json(response, entities, threat_score)
        
        # TODO: Print the response
        print(f"\n{'='*80}")
        print(f"🔍 Assistant Response for {analyst_id}:")
        print(f"{'='*80}")
        print(response)
        print(f"{'='*80}")
        
        # Show extracted entities
        if entities:
            print(f"\n📊 Extracted Entities:")
            for key, value in entities.items():
                print(f"   • {key}: {value}")
        
        # Show threat score (BONUS 3)
        print(f"\n🎯 Threat Score Analysis:")
        print(f"   • Score: {threat_score['score']}/100")
        print(f"   • Level: {threat_score['threat_level']}")
        print(f"   • Risk Factors:")
        for factor in threat_score['factors']:
            print(f"      - {factor}")
        
        # Show structured JSON (BONUS 4)
        print(f"\n📋 Structured JSON Output:")
        import json
        print(json.dumps(json_output, indent=2))
        
        # Show session info
        session_history = store.get(analyst_id)
        if session_history:
            msg_count = len(session_history.messages)
            print(f"\n💬 Session Info: {msg_count} messages in history for {analyst_id}")
            
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        print(f"\n🔍 Full traceback:")
        traceback.print_exc()
        print("\nPlease try again or check your query format.")