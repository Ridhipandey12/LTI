from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
import re, json

with open("security_incidents.txt", "r") as f:
    lines = f.readlines()
docs = [line.strip() for line in lines if line.strip()]
print(f"No. of incidents loaded: {len(docs)}")
for i in range(3): print(f"Sample {i+1}: {docs[i]}")

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = splitter.create_documents(docs)
print(f"No. of chunks generated: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
faiss = FAISS.from_documents(chunks, embeddings)
bm25 = BM25Retriever.from_documents(chunks)
retriever = EnsembleRetriever(retrievers=[faiss.as_retriever(search_kwargs={"k": 4}), bm25], weights=[0.7, 0.3])

def extract_entities(text):
    ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text)
    mitre = re.findall(r"\bT\d{4}\b", text)
    os = re.findall(r"\bWindows|Linux|macOS\b", text, re.IGNORECASE)
    severity = re.findall(r"\bLow|Medium|High|Critical\b", text, re.IGNORECASE)
    hostnames = re.findall(r"\bhost\d+\b", text)
    entities = {
        "IPs": list(set(ips)),
        "MITRE": list(set(mitre)),
        "OS": list(set(os)),
        "Severity": list(set(severity)),
        "Hostnames": list(set(hostnames))
    }
    return "\n".join([f"{k}: {', '.join(v)}" for k, v in entities.items() if v])

def enrich_threat(answer):
    if "powershell" in answer.lower():
        return answer + "\nThreat Enrichment: Encoded PowerShell detected, possible obfuscation."
    if "brute-force" in answer.lower():
        return answer + "\nThreat Enrichment: Brute-force SSH attempt, check login logs and block IP."
    return answer

def score_threat(text):
    score = 0
    if "T1059" in text or "powershell" in text.lower(): score += 2
    if "brute-force" in text.lower(): score += 2
    if "Critical" in text: score += 3
    if re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text): score += 1
    return f"Threat Score: {score}/10"

prompt = ChatPromptTemplate.from_template(
    "Retrieved Incidents:\n{context}\n\nEntity Memory:\n{entities}\n\nAnalyst History:\n{chat_history}\n\nQuery:\n{question}\n\nAnswer:"
)

llm = ChatOllama(model="mistral", temperature=0.3)
parser = StrOutputParser()

base_chain = (
    {
        "context": lambda x: retriever.invoke(x["question"]),
        "entities": lambda x: extract_entities(x["question"] + "\n" + "\n".join([doc.page_content for doc in retriever.invoke(x["question"])])),
        "chat_history": lambda x, config: get_memory(config["configurable"]["session_id"]).messages[-5:],
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
    | parser
    | RunnableLambda(lambda x: enrich_threat(x) + "\n" + score_threat(x))
)

memory_store = {}
def get_memory(user_id):
    if user_id not in memory_store:
        memory_store[user_id] = InMemoryChatMessageHistory()
    return memory_store[user_id]

rag_chain = RunnableWithMessageHistory(
    base_chain,
    get_memory,
    input_messages_key="question",
    history_messages_key="chat_history"
)


if __name__ == "__main__":
    while True:
        user_id = input("\nEnter analyst ID (or 'exit'): ").strip()
        if user_id.lower() == "exit": break
        query = input("Enter alert/query: ").strip()
        config = {"configurable": {"session_id": user_id}}
        response = rag_chain.invoke({"question": query}, config=config)
        get_memory(user_id).add_message(AIMessage(content=response))
        print("\n--- Response ---")
        print(response)
        print("\n--- JSON Output ---")
        print(json.dumps({"analyst": user_id, "response": response}, indent=2))