# STEP 1: IMPORTS 

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

# LLM setup
llm = ChatOllama(model="mistral", temperature=0.3)

# step 2: Load Data 
print("\n=== STEP 1: LOAD DATA ===")
loader = TextLoader("security_incidents.txt")
docs = loader.load()
print(f"Loaded {len(docs)} documents.")
print("Sample:\n", docs[0].page_content)

# step 3: Chunk Data
print("\n=== STEP 2: CHUNKING ===")
splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
splits = splitter.split_documents(docs)
print(f"Total chunks: {len(splits)}")

# step 4: Embeddings + FAISS
print("\n=== STEP 3: EMBEDDINGS + INDEX ===")
emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(splits, emb)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Bonus Hybrid Retrieval
print("\n=== BONUS: HYBRID RETRIEVAL ===")
bm25 = BM25Retriever.from_documents(splits)
bm25.k = 4
hybrid = EnsembleRetriever(retrievers=[retriever, bm25], weights=[0.7, 0.3])
print("Hybrid retriever ready.\n")

# STEP 5: PROMPT + LCEL RAG CHAIN
print("\n=== STEP 4: RAG CHAIN ===")

prompt = ChatPromptTemplate.from_template("""
You are a SOC Analyst Assistant. Use the following context and history to help answer the analyst's query.

Context:
{context}

Entities:
{entities}

History:
{history}

Query:
{question}

Respond with a recommended resolution and relevant insights.
""")

parser = StrOutputParser()

chain = (
    {
        "context": lambda x: "\n\n".join([doc.page_content for doc in hybrid.invoke(x["question"])]),
        "question": RunnablePassthrough(),
        "entities": lambda x: extract_entities(x["question"], ""),
        "history": lambda x: x.get("history", [])
    }
    | prompt
    | llm
    | parser
)
print("RAG chain constructed.\n")

# BONUS: TOOL
print("\n=== BONUS: TOOL ===")

@tool
def threat_enrich(ip: str):
    """Students: Return mock threat intel for IP"""
    return f"IP {ip} is flagged in 3 threat feeds. Associated with known C2 infrastructure."

# STEP 6: MEMORY (RunnableWithMessageHistory)
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

# STEP 7: ENTITY EXTRACTION (BONUS)
def extract_entities(query: str, context: str):
    text = query + " " + context
    ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
    mitre = re.findall(r"T\d{4}", text)
    os = re.findall(r"\b(?:Windows|Linux|macOS|Ubuntu)\b", text)
    severity = re.findall(r"\b(?:Low|Medium|High|Critical)\b", text)
    return {"IPs": ips, "MITRE": mitre, "OS": os, "Severity": severity}

# STEP 8: INTERACTIVE CONSOLE LOOP
print("\n=== STEP 7: CONSOLE LOOP ===")

while True:
    raw = input("Enter: <analyst_id> <query> (or q): ")
    if raw == "q":
        break
    
    parts = raw.split(" ", 1)
    if len(parts) < 2:
        print("Format: analyst42 suspicious ssh activity")
        continue

    analyst_id, query = parts
    entities = extract_entities(query, "")
    response = memory_chain.invoke(
        {"question": query},
        config={"configurable": {"session_id": analyst_id}}
    )
    print("\n--- Response ---")
    print(response)
