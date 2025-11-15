import re
import json
from typing import Optional, Dict, List, Any
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.embeddings import Embeddings
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from datetime import datetime

class LocalEmbeddings(Embeddings):
    """Local embeddings using SentenceTransformer"""
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    
    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True).tolist()

def load_security_incidents(file_path: str) -> List[Dict[str, str]]:
    """Load and parse security incidents from text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        lines = content.strip().split('\n')
        incidents = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            incidents.append(parse_incident_line(line))
        
        print(f"\n✓ Loaded {len(incidents)} security incidents from {file_path}")
        if incidents:
            sample = incidents[0]
            print(f" Sample incident:")
            print(f"   ID: {sample.get('Incident')} | User: {sample.get('User')} | Severity: {sample.get('Severity')}")
            print(f"   Alert: {sample.get('Alert', 'N/A')[:70]}...")
        print()
        return incidents
    except FileNotFoundError:
        print(f" ERROR: File not found at '{file_path}'")
        print("Please ensure the file exists in the current directory.")
        return []
    except Exception as e:
        print(f" ERROR loading file: {e}")
        return []

def parse_incident_line(line: str) -> Dict[str, str]:
    """Parse pipe-delimited incident line into dictionary"""
    incident = {}
    parts = line.split('|')
    
    for part in parts:
        part = part.strip()
        if '=' in part:
            key, value = part.split('=', 1)
            incident[key.strip()] = value.strip()
    
    return incident

def incidents_to_documents(incidents: List[Dict]) -> List[str]:
    """Convert parsed incidents to document strings for chunking"""
    documents = []
    for inc in incidents:
        doc_text = f"""Incident ID: {inc.get('Incident', 'Unknown')}
User: {inc.get('User', 'Unknown')}
Alert: {inc.get('Alert', 'No alert')}
Source IP: {inc.get('SourceIP', 'N/A')}
Hostname: {inc.get('Host', 'Unknown')}
Operating System: {inc.get('OS', 'Unknown')}
MITRE Technique: {inc.get('MITRE', 'N/A')}
Severity: {inc.get('Severity', 'Unknown')}
Resolution Steps: {inc.get('Resolution', 'No action taken')}"""
        documents.append(doc_text)
    
    return documents

class EntityExtractor:
    """Extract security-relevant entities from text"""
    
    @staticmethod
    def extract_entities(text: str) -> Dict[str, List[str]]:
        """Extract IPs, OS, Hostname, MITRE tags, Severity"""
        entities = {
            "ip_addresses": [],
            "operating_systems": [],
            "hostnames": [],
            "mitre_techniques": [],
            "severity_levels": [],
            "users": []
        }
        
        # IP addresses (including CIDR notation)
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b'
        ips = re.findall(ip_pattern, text)
        entities["ip_addresses"] = list(set(ips))
        
        # OS detection - comprehensive patterns
        os_keywords = {
            "Ubuntu": r"Ubuntu\s+\d+",
            "Windows": r"Windows\s+(?:\d+|Server\s+\d+)",
            "CentOS": r"CentOS\s+\d+",
            "RedHat": r"RedHat\s+\d+|Red\s+Hat\s+\d+",
            "Debian": r"Debian\s+\d+",
            "macOS": r"macOS\s+\d+",
            "Linux": r"Linux"
        }
        for label, pattern in os_keywords.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities["operating_systems"].extend(matches)
        
        # Hostname extraction
        host_pattern = r'(?:Host|Hostname)(?:\s*=|\s*:)\s*([a-zA-Z0-9\-_.]+)'
        hostnames = re.findall(host_pattern, text, re.IGNORECASE)
        entities["hostnames"] = list(set(hostnames))
        
        # MITRE techniques (T#### or T####.###)
        mitre_pattern = r'T\d{4}(?:\.\d{3})?'
        mitre_tags = re.findall(mitre_pattern, text)
        entities["mitre_techniques"] = list(set(mitre_tags))
        
        # Severity levels
        severity_keywords = ["Critical", "High", "Medium", "Low"]
        for sev in severity_keywords:
            if re.search(rf'\b{sev}\b', text, re.IGNORECASE):
                entities["severity_levels"].append(sev)
        
        # User extraction
        user_pattern = r'User(?:\s*=|\s*:)\s*([a-zA-Z0-9_.]+)'
        users = re.findall(user_pattern, text, re.IGNORECASE)
        entities["users"] = list(set(users))
        
        return entities

class ThreatScorer:
    """Calculate threat score based on multiple factors"""
    
    MITRE_SCORES = {
        "T1110": 8,
        "T1110.001": 8,
        "T1059": 7,
        "T1059.001": 7,
        "T1059.004": 7,
        "T1059.005": 7,
        "T1059.006": 7,
        "T1059.007": 7,
        "T1003": 9,
        "T1021": 7,
        "T1021.001": 7,
        "T1486": 10,
        "T1071": 9,
        "T1568": 8,
        "T1068": 8,
        "T1046": 7,
        "T1005": 8,
        "T1106": 7,
        "T1552": 8,
        "T1133": 7,
        "T1053": 7,
        "T1053.003": 7,
        "T1105": 8,
        "T1090": 8,
        "T1091": 8,
        "T1047": 7,
        "T1041": 9,
        "T1190": 8,
        "T1562": 8,
        "T1078": 7,
        "T1078.001": 7,
    }
    
    SEVERITY_SCORES = {
        "Critical": 15,
        "High": 10,
        "Medium": 5,
        "Low": 2
    }
    
    @staticmethod
    def calculate_threat_score(entities: Dict, text: str) -> int:
        """Calculate composite threat score (0-100)"""
        score = 0
        
        # MITRE technique score
        for mitre in entities.get("mitre_techniques", []):
            score += ThreatScorer.MITRE_SCORES.get(mitre, 5)
        
        # Severity score
        for sev in entities.get("severity_levels", []):
            score += ThreatScorer.SEVERITY_SCORES.get(sev, 3)
        
        # Malicious IP presence
        if entities.get("ip_addresses"):
            score += 3
        
        # Suspicious keywords
        suspicious_keywords = ["malware", "ransomware", "c2", "brute", "exploit", 
                              "credential", "dump", "reverse shell", "obfuscated", 
                              "persistence", "lateral", "exfiltration", "encryption"]
        score += sum(4 for kw in suspicious_keywords if kw in text.lower())
        
        # Multiple failed logins indicator
        if any(x in text.lower() for x in ["failed", "attempt", "spike", "multiple"]):
            score += 3
        
        return min(score, 100)

@tool
def threat_enrichment_tool(ip_address: str) -> str:
    """Enriches threat information for a given IP address.
    
    Args:
        ip_address: IP address to enrich
    
    Returns:
        Enrichment data including reputation, known threats, and recommendations
    """
    threat_db = {
        "10.1.1.9": {
            "reputation": "Suspicious",
            "known_campaigns": ["SSH Brute Force Attack"],
            "last_seen": "2024-01-10",
            "action": "Monitor login attempts, enforce MFA"
        },
        "172.22.9.54": {
            "reputation": "Malicious",
            "known_campaigns": ["Network Reconnaissance", "Port Scanning"],
            "last_seen": "2024-01-12",
            "action": "Block immediately at firewall"
        },
        "10.22.3.9": {
            "reputation": "High Risk",
            "known_campaigns": ["C2 Communication", "Data Exfiltration"],
            "last_seen": "2024-01-15",
            "action": "Block all outbound traffic, investigate compromised host"
        },
        "192.168.55.20": {
            "reputation": "Suspicious",
            "known_campaigns": ["Geographic Anomaly", "Credential Abuse"],
            "last_seen": "2024-01-13",
            "action": "Enforce MFA and geographic restrictions"
        }
    }
    
    enrichment = threat_db.get(ip_address, {
        "reputation": "Unknown",
        "known_campaigns": ["No known campaigns"],
        "last_seen": "Never",
        "action": "Continue baseline monitoring",
        "recommendation": "Monitor for anomalies"
    })
    
    return json.dumps(enrichment, indent=2)

def setup_retrievers(documents: List[str], embeddings: Embeddings):
    """Setup FAISS + BM25 ensemble retriever"""
    
    # Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n", " ", ""]
    )
    chunks = text_splitter.split_text("\n\n".join(documents))
    print(f"📊 Total document chunks created: {len(chunks)}")
    
    # FAISS vector store
    print("🔍 Building FAISS vector store...")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    faiss_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    
    # BM25 retriever
    print("📚 Building BM25 retriever...")
    bm25_retriever = BM25Retriever.from_texts(chunks)
    bm25_retriever.k = 4
    
    # Ensemble retriever with weights
    print("🔗 Creating ensemble retriever (FAISS: 0.7, BM25: 0.3)...")
    ensemble = EnsembleRetriever(
        retrievers=[faiss_retriever, bm25_retriever],
        weights=[0.7, 0.3]
    )
    
    print("✓ Retrieval system ready!\n")
    return ensemble, chunks

def format_docs(docs) -> str:
    """Format retrieved documents"""
    return "\n\n" + "="*70 + "\n\n".join([doc.page_content for doc in docs])

analyst_memories: Dict[str, InMemoryChatMessageHistory] = {}

def get_chat_history(analyst_id: str) -> InMemoryChatMessageHistory:
    """Retrieve or create chat history for an analyst."""
    if analyst_id not in analyst_memories:
        analyst_memories[analyst_id] = InMemoryChatMessageHistory()
    return analyst_memories[analyst_id]

def get_analyst_summary(analyst_id: str) -> str:
    """
    Retrieves a summary or context for a given analyst.
    This is a placeholder for a more complex lookup (e.g., from a user database).
    """
    analyst_profiles = {
        "alex_g": "Alex is a Tier 2 analyst specializing in network forensics.",
        "sara_k": "Sara is a junior analyst focusing on malware triage.",
        "default": "Analyst has no specific profile information available."
    }
    return analyst_profiles.get(analyst_id, analyst_profiles["default"])

def build_soc_chain(llm, ensemble_retriever, embeddings):
    """Build LCEL RAG chain with entity memory"""
    
    entity_extractor = EntityExtractor()
    threat_scorer = ThreatScorer()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Security Operations Center (SOC) Analyst Assistant specializing in incident response and threat analysis.

Your responsibilities:
1. Analyze security incidents using historical context
2. Extract and highlight critical security indicators
3. Provide actionable recommendations
4. Calculate and explain threat severity
5. Reference similar past incidents for context

Incident Analysis Framework:
- Classify incident type based on MITRE ATT&CK techniques
- Assess severity using multiple factors
- Retrieve and reference similar historical incidents
- Provide specific, actionable resolution steps
- Identify related security entities (IPs, hosts, users)
- Recommend preventive measures

Response Format:
 Incident Classification: [Type of incident]
 Threat Score: [Score/100 with explanation]
 Severity: [Critical/High/Medium/Low]
 Similar Historical Incidents: [References to past incidents]
 Recommended Actions: [Specific resolution steps]
 Extracted Entities: [IPs, Hosts, Users, Techniques]
 Preventive Measures: [Future prevention steps]
 Investigation Priorities: [Key areas to focus on]"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", """SECURITY INCIDENT ANALYSIS REQUEST

Analyst Query:
{question}

Retrieved Similar Historical Incidents:
{context}

Extracted Security Entities:
{entity_memory}

Threat Score: {threat_score}/100

Analyst Context:
{analyst_context}

Please provide a comprehensive incident analysis with recommendations.""")
    ])
    
    def extract_and_score(inputs):
        """Extract entities and calculate threat score"""
        question = inputs.get("question", "")
        context = inputs.get("context", "")
        analyst_id = inputs.get("analyst_id", "Unknown")
        full_text = question + " " + context
        
        entities = entity_extractor.extract_entities(full_text)
        threat_score = threat_scorer.calculate_threat_score(entities, full_text)
        analyst_context = get_analyst_summary(analyst_id)
        
        entity_text = json.dumps(entities, indent=2)
        return {
            "entity_memory": entity_text,
            "threat_score": threat_score,
            "analyst_context": analyst_context
        }
    
    chain = (
        {
            "context": RunnableLambda(lambda x: x["question"]) | ensemble_retriever | format_docs,
            "question": lambda x: x["question"],
            "chat_history": RunnableLambda(
                lambda x: x.get("messages", []) if isinstance(x, dict) else []
            ),
            "analyst_id": RunnableLambda(lambda x: x.get("analyst_id", "Unknown"))
        }
        | RunnableLambda(lambda x: {
            **x,
            **extract_and_score(x)
        })
        | RunnablePassthrough.assign(
            analysis=prompt | llm | StrOutputParser()
        )
        | RunnableLambda(
            lambda x: create_structured_report(
                x["analysis"], x["analyst_id"], x["question"], x["threat_score"], x["entity_memory"]
            )
        )
    )
    
    return chain

def create_structured_report(analysis: str, analyst_id: str, question: str, threat_score: int, entity_memory: str) -> Dict[str, Any]:
    """Convert LLM response to structured JSON report."""
    return {
        "report_metadata": {
            "analyst_id": analyst_id,
            "timestamp": datetime.now().isoformat(),
            "threat_score": threat_score
        },
        "incident_query": question,
        "extracted_entities": json.loads(entity_memory),
        "analysis": analysis,
        "report_format": "structured_json"
    }


def main():
    print("\n" + "="*80)
    print("  SOC ANALYST ASSISTANT - SECURITY INCIDENT RAG SYSTEM ")
    print("="*80)
    print("Powered by: LangChain + Ollama + FAISS + BM25 Hybrid Retrieval")
    print("="*80 + "\n")
    
    # Load dataset
    print("Loading security incidents from file...")
    incidents = load_security_incidents("security_incidents.txt")
    
    if not incidents:
        print("\n FATAL: Cannot proceed without incident data.")
        print("   Please create 'security_incidents.txt' in the current directory.")
        return
    
    # Convert to documents
    print(f" Processing {len(incidents)} incidents into searchable documents...")
    documents = incidents_to_documents(incidents)
    
    # Setup embeddings and retrievers
    print("\n Initializing Embedding & Retrieval Systems...")
    embeddings = LocalEmbeddings()
    ensemble_retriever, chunks = setup_retrievers(documents, embeddings)
    
    # Initialize LLM
    print("Initializing Ollama Language Model...")
    print("  IMPORTANT: Ollama must be running before proceeding!")
    print("   In another terminal, run: ollama serve")
    print("   Then select model: ollama pull mistral (or llama2)\n")
    
    try:
        llm = ChatOllama(model="mistral", temperature=0.7)
        # Test connection
        test_response = llm.invoke("Hello, say OK.")
        print("Ollama connection successful!\n")
    except Exception as e:
        print(f" FATAL: Cannot connect to Ollama")
        print(f"   Error: {e}")
        print("   Please start Ollama with: ollama serve")
        return
    
    # Build chain
    print(" Building RAG chain with entity extraction and threat scoring...")
    chain = build_soc_chain(llm, ensemble_retriever, embeddings)
    print("✓ RAG chain ready!\n")
    
    print("="*80)
    print("✓✓✓ SOC ASSISTANT FULLY INITIALIZED ✓✓✓")
    print("="*80)
    print("\n USAGE GUIDE:")
    print("    Enter your Analyst ID")
    print("    Describe the security incident or ask a query")
    print("    Your conversation history is saved per analyst")
    print("\n COMMANDS:")
    print("   'history' - View all active analysts and their message counts")
    print("   'clear <analyst_id>' - Clear specific analyst's memory")
    print("   'exit' - Quit the application\n")
    print("="*80 + "\n")
    
    # Main loop
    while True:
        try:
            analyst_id = input("\n👤 Enter Analyst ID (or 'exit'): ").strip()
            
            if analyst_id.lower() == "exit":
                print("\n Thank you for using SOC Assistant. Stay secure!")
                break

            if analyst_id.lower() == "history":
                if not analyst_memories:
                    print(" No active analysts yet.")
                else:
                    print("\n ACTIVE ANALYSTS:")
                    for aid, memory in analyst_memories.items():
                        print(f"   • {aid}: {len(memory.messages)} messages in history")
                continue

            if analyst_id.lower().startswith("clear"):
                parts = analyst_id.split(maxsplit=1)
                if len(parts) > 1:
                    clear_id = parts[1]
                    if clear_id in analyst_memories:
                        del analyst_memories[clear_id]
                        print(f"Cleared memory for analyst: {clear_id}")
                    else:
                        print(f" No memory found for analyst: {clear_id}")
                else:
                    print(" Usage: clear <analyst_id>")
                continue

            if not analyst_id:
                print("  Analyst ID cannot be empty.")
                continue

            query = input(" Enter Security Incident Query (or 'exit'): ").strip()

            if query.lower() == "exit":
                continue

            if not query:
                print("  Query cannot be empty.")
                continue
            
            print("\n ANALYZING INCIDENT...")
            print("   - Searching historical incidents...")
            print("   - Extracting entities...")
            print("   - Calculating threat score...")
            print("   - Generating analysis...\n")

            # Get analyst memory
            history = get_chat_history(analyst_id)

            # Invoke chain.
            structured_report = chain.invoke({
                "question": query,
                "messages": history.messages,
                "analyst_id": analyst_id
            })
            
            llm_response = structured_report["analysis"]
            
            # Save to memory            
            history.add_user_message(query)
            history.add_ai_message(llm_response)
            
            # Display results            
            print("="*80)
            print(" INCIDENT ANALYSIS REPORT")
            print("="*80)
            print(f"Analyst: {analyst_id}")
            print(f"Timestamp: {structured_report['report_metadata']['timestamp']}")
            print(f"Threat Score: {structured_report['report_metadata']['threat_score']}/100")
            print(f"Session Messages: {len(history.messages)}")
            print("-"*80 + "\n")
            print(llm_response)
            print("\n" + "="*80 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n Application interrupted. Stay secure!")
            break


if __name__ == "__main__":
    main()