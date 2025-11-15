from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

import re, json

#  Load and chunk dataset
loader = TextLoader("security_incidents.txt")
docs = loader.load()
print(f"Loaded {len(docs)} document(s). Sample:\n{docs[0].page_content[:300]}")

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
chunks = splitter.split_documents(docs)
print(f"Generated {len(chunks)} chunks.")

#  Embeddings + FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(chunks, embeddings)
faiss_retriever = vector_store.as_retriever(search_kwargs={"k": 4})

#  BM25 + Hybrid
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4
hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, faiss_retriever],
    weights=[0.3, 0.7]
)

#  Entity extraction 
def extract_entities(text):
    return {
        "ips": re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text),
        "mitre": re.findall(r"T\d{4}", text),
        "os": re.findall(r"\bWindows|Linux|Ubuntu|macOS\b", text),
        "severity": re.findall(r"\bLow|Medium|High|Critical\b", text),
        "hostnames": re.findall(r"\b(?:SRV|WKS|host)[\w\-]+\b", text)
    }

def threat_score(entities):
    score = 0
    if entities["mitre"]: score += 2
    if "Critical" in entities["severity"]: score += 3
    if entities["ips"]: score += 2
    return score

#  Prompt
prompt = PromptTemplate.from_template("""
You are a SOC Analyst Assistant.

Context:
{context}

Entities:
{entities}

Analyst History:
{chat_history}

Alert:
{question}

Respond in JSON with:
- summary
- resolution
- threat_score
- extracted_entities
""")

# LLM
llm = ChatOllama(model="mistral", temperature=0.7)

# Build RAG Chain
def format_input(inputs):
    retrieved = hybrid_retriever.invoke(inputs["question"])
    context = "\n".join([doc.page_content for doc in retrieved])
    entities = extract_entities(inputs["question"] + context)
    score = threat_score(entities)
    return {
        "context": context,
        "entities": json.dumps(entities),
        "chat_history": inputs["chat_history"],
        "question": inputs["question"],
        "threat_score": score
    }

base_chain = (
    RunnableLambda(format_input)
    | prompt
    | llm
)

# Memory
user_memory_store = {}
def get_memory(user_id):
    if user_id not in user_memory_store:
        user_memory_store[user_id] = InMemoryChatMessageHistory()
    return user_memory_store[user_id]

rag_chain = RunnableWithMessageHistory(
    base_chain,
    get_memory,
    input_messages_key="question",
    history_messages_key="chat_history"
)

#  JSON Output
def format_response(text):
    try:
        data = json.loads(text)

        # Normalize resolution steps
        resolution_steps = data.get("resolution", [])
        if isinstance(resolution_steps, str):
            resolution_steps = [step.strip() for step in re.split(r"\.\s+", resolution_steps) if step]

        # Normalize extracted entities
        entities = data.get("extracted_entities", {})
        structured = {
            "incident_summary": {
                "issue": data.get("summary", "N/A"),
                "affected_user": entities.get("user", "Not identified from the provided incidents.")
            },
            "recommended_resolution": {
                "steps": resolution_steps
            },
            "threat_assessment": {
                "threat_score": data.get("threat_score", "N/A"),
                "mitre_techniques": entities.get("mitre", [])
            },
            "extracted_entities": {
                "users": [entities.get("user")] if "user" in entities else [],
                "hosts": [entities.get("host")] if "host" in entities else [],
                "operating_systems": [entities.get("os")] if "os" in entities else [],
                "ip_addresses": entities.get("ips", []),
                "mitre": entities.get("mitre", [])
            }
        }

        print("\n Structured JSON Output:\n")
        print(json.dumps(structured, indent=2))

    except Exception as e:
        print(text)

# Console Loop
print("\nSOC RAG Assistant Ready. Type 'exit' to quit.\n")
while True:
    user_id = input("Analyst ID: ").strip()
    if user_id.lower() in ["exit", "quit"]: break
    query = input("Alert Summary: ").strip()
    response = rag_chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": user_id}}
    )

    try:
        format_response(response.content)
    except AttributeError:
        try:
            format_response(response["answer"])
        except:
            try:
                format_response(response.generations[0][0].text)
            except:
                print(f"\n🔍 Raw Response:\n{response}\n")