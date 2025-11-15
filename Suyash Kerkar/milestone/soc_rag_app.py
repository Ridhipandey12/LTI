from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_core.chat_history import InMemoryChatMessageHistory

from typing import List, Dict, Any  
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.llms import Ollama
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda, RunnableMap, RunnablePassthrough
#from langchain_community.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import StuffDocumentsChain
from langchain_core.prompts import PromptTemplate
import re
import json
import os 


import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_ollama import OllamaLLM
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnableMap

# -------------------------------
# 1. Load and Parse Dataset
# -------------------------------

def load_incidents(file_path: str) -> List[Document]:
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"✅ Loaded {len(lines)} incidents.")
    print("📄 Sample incident:\n", lines[0])
    docs = [Document(page_content=line.strip()) for line in lines if line.strip()]
    return docs

# -------------------------------
# 2. Chunking
# -------------------------------

def chunk_documents(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"✅ Total chunks created: {len(chunks)}")
    return chunks

# -------------------------------
# 3. Embeddings + FAISS
# -------------------------------

def build_vector_retriever(chunks: List[Document]):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    return retriever

# -------------------------------
# 4. Hybrid Retrieval (FAISS + BM25)
# -------------------------------

def build_hybrid_retriever(chunks: List[Document]):
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = 4
    vector = build_vector_retriever(chunks)
    hybrid = EnsembleRetriever(retrievers=[vector, bm25], weights=[0.7, 0.3])
    return hybrid

# -------------------------------
# 5. Entity Extraction
# -------------------------------

def extract_entities(text: str) -> Dict[str, Any]:
    return {
        "ips": re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", text),
        "os": re.findall(r"\b(?:Windows|Linux|Ubuntu|macOS|Debian|CentOS)\b", text, re.IGNORECASE),
        "mitre": re.findall(r"T\d{4}(?:\.\d+)?", text),
        "severity": re.findall(r"\b(Critical|High|Medium|Low)\b", text),
        "hostnames": re.findall(r"\b[A-Z]{2,5}-[A-Z0-9]+\b", text)
    }

# -------------------------------
# 6. Threat Score
# -------------------------------

def compute_threat_score(entities: Dict[str, Any]) -> float:
    score = 0
    if "Critical" in entities["severity"]:
        score += 4
    elif "High" in entities["severity"]:
        score += 3
    elif "Medium" in entities["severity"]:
        score += 2
    elif "Low" in entities["severity"]:
        score += 1
    if any("powershell" in x.lower() for x in entities.get("mitre", [])):
        score += 2
    if any("brute" in x.lower() for x in entities.get("mitre", [])):
        score += 2
    if entities["ips"]:
        score += 1
    return round(score, 2)

# -------------------------------
# 7. Prompt Template
# -------------------------------

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a SOC Analyst Assistant. Use the retrieved incidents, analyst memory, and extracted entities to recommend a resolution."),
    MessagesPlaceholder("history"),
    ("user", "{input}"),
    ("system", "Relevant Incidents:\n{context}\n\nEntities:\n{entities}\n\nThreat Score: {threat_score}")
])

# -------------------------------
# 8. Manual RAG Chain
# -------------------------------

def build_chain(retriever):
    llm = OllamaLLM(model="mistral")

    def enrich(inputs):
        query = inputs["input"]
        docs = retriever.invoke(query)
        context = "\n".join([doc.page_content for doc in docs])
        entities = extract_entities(query + "\n" + context)
        score = compute_threat_score(entities)
        return {
            "input": query,
            "context": context,
            "entities": entities,
            "threat_score": score,
            "history": inputs.get("history", [])
        }

    chain = (
        RunnableLambda(enrich)
        | prompt
        | llm
    )

    return chain

# -------------------------------
# 9. Memory Setup
# -------------------------------

store = {}

def get_history(user_id: str):
    if user_id not in store:
        store[user_id] = InMemoryChatMessageHistory()
    return store[user_id]

# -------------------------------
# 10. Main Console Loop
# -------------------------------

def main():
    print("🔄 Loading incidents...")
    docs = load_incidents("security_incidents.txt")

    print("🔄 Chunking documents...")
    chunks = chunk_documents(docs)

    print("🔄 Building retriever...")
    retriever = build_hybrid_retriever(chunks)

    print("🔄 Building RAG chain...")
    chain = build_chain(retriever)

    user_id = input("👤 Enter your analyst ID: ").strip()
    chain_with_memory = RunnableWithMessageHistory(
        chain,
        lambda session_id: get_history(session_id),
        input_messages_key="input",
        history_messages_key="history"
    )

    print("\nSOC RAG Assistant Ready. Type 'exit' to quit.\n")

    while True:
        query = input(f"[{user_id}] 🔍 Enter alert summary: ").strip()
        if query.lower() in ["exit", "quit"]:
            print(" Exiting SOC RAG Assistant.")
            break

        response = chain_with_memory.invoke(
            {"input": query},
            config={"configurable": {"session_id": user_id}}
        )

        print("\n Assistant Response:\n")
        print(response)
        print("\n" + "-" * 60 + "\n")

        # Save to JSON
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"response_{user_id}_{timestamp}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({
                "analyst_id": user_id,
                "query": query,
                "response": str(response)
            }, f, indent=2)
        print(f"💾 Response saved to {filename}")

if __name__ == "__main__":
    main()