# Sagar Agrawal
# PS ID - 10843112

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
import re, json, warnings

warnings.filterwarnings('ignore')  #To avoid hugging face warnings

llm = ChatOllama(model="mistral", temperature=0.5)

print("=== STEP 1: LOAD DATA ===")
loader = TextLoader("security_incidents.txt")
docs = loader.load()
print(f"Loaded {len(docs)} documents.")
print("Sample:\n", docs[0].page_content[:300])

print("\n=== STEP 2: CHUNKING ===")
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
splits = splitter.split_documents(docs)
print(f"Total chunks: {len(splits)}")

print("\n=== STEP 3: EMBEDDINGS + INDEX ===")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, emb)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("\n=== BONUS: HYBRID RETRIEVAL ===")
bm25 = BM25Retriever.from_documents(splits)
bm25.k = 4
hybrid = EnsembleRetriever(retrievers=[retriever, bm25], weights=[0.7, 0.3])
print("Hybrid retriever ready.")

print("\n=== STEP 4: RAG CHAIN ===")

prompt = ChatPromptTemplate.from_template("""
You are a SOC Analyst Assistant. Use the following context to help answer the analyst's query.

Context:
{context}

Entities:
{entities}

Conversation History:
{history}

Analyst Query:
{question}

Respond with a recommended resolution, similar past incidents, and any threat indicators.
""")

parser = StrOutputParser()

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

chain = (
    {
        "context": lambda x: format_docs(hybrid.invoke(x["question"])),
        "question": lambda x: x["question"],
        "entities": lambda x: x.get("entities", {}),
        "history": lambda x: x.get("history", []),
    }
    | prompt
    | llm
    | parser
)
print("RAG chain constructed.")


print("\n=== BONUS: TOOL ===")

@tool
def threat_enrich(ip: str):
    """Mock threat enrichment tool for IP addresses."""
    return f"IP {ip} is flagged in threat intel feeds for brute-force SSH attempts and malware C2 activity."

llm.bind_tools([threat_enrich])
print("@tool threat_enrich added successfully")

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
print("Memory wrapper ready.\n")

def extract_entities(query: str, context: str):
    text = query + " " + context
    entities = {}
    entities["ips"] = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
    entities["hostnames"] = re.findall(r"host(?:name)?[:=]?\s*(\S+)", text, re.IGNORECASE)
    entities["os"] = re.findall(r"\b(?:Windows|Linux|Ubuntu|macOS|CentOS|Debian)\b", text, re.IGNORECASE)
    entities["mitre"] = re.findall(r"T\d{4}", text)
    entities["severity"] = re.findall(r"\b(Critical|High|Medium|Low)\b", text, re.IGNORECASE)
    return entities

#BONUS - SCORE
def compute_threat_score(entities, query, context):
    score = 0

    severity_map = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}
    for sev in entities.get("severity", []):
        score += severity_map.get(sev.capitalize(), 0)

    score += len(entities.get("mitre", [])) * 2

    malicious_ips = {"192.168.1.100", "10.0.0.5", "172.16.0.9"}
    for ip in entities.get("ips", []):
        if ip in malicious_ips:
            score += 3

    if "powershell" in query.lower() or "powershell" in context.lower():
        score += 2

    brute_keywords = ["brute force", "failed login", "multiple attempts", "ssh"]
    if any(kw in query.lower() or kw in context.lower() for kw in brute_keywords):
        score += 2

    return score

print("=== STEP 7: CONSOLE LOOP ===")

while True:
    analyst_id = input("\nEnter analyst ID (or 'exit'): ").strip()
    if analyst_id.lower() == "exit": break
    query = input("Enter alert/query: ").strip()

       
    retrieved_docs = hybrid.invoke(query)
    context = format_docs(retrieved_docs)
    entities = extract_entities(query, context)

    threat_score = compute_threat_score(entities, query, context)

    response = memory_chain.invoke(
        {"question": query, "entities": entities},
        config={"configurable": {"session_id": analyst_id}}
    )

    
    print("---BONUS : JSON-RESPONSE ---")


    structured_output = {
        "analyst_id": analyst_id,
        "query": query,
        "entities": entities,
        "threat_score": threat_score,
        "response": response
    }

    print(json.dumps(structured_output, indent=2))
    print("-------------------------\n")
    print("Tools, Entities, Score, Memory, Json Response Implemented")
