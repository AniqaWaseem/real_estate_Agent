import os
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
import uuid

from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# ✅ Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ✅ Initialize FastAPI
app = FastAPI()

# ✅ Gemini Embeddings for Vector DB
embedding = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",  # ✅ Latest embedding model
    google_api_key=GOOGLE_API_KEY
)

# ✅ Gemini Chat Model (LLM)
chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",  # ✅ Best for RAG
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# ✅ Load Knowledge Base Files
knowledge_dir = "./knowledge_base"
docs = []

print("📚 Loading knowledge base files...")
for filename in os.listdir(knowledge_dir):
    path = os.path.join(knowledge_dir, filename)

    if filename.lower().endswith(".pdf"):
        loader = PyPDFLoader(path)
        docs.extend(loader.load())
        print(f"📄 Loaded PDF: {filename}")

    elif filename.lower().endswith(".txt"):
        loader = TextLoader(path, encoding="utf-8")
        docs.extend(loader.load())
        print(f"📝 Loaded TXT: {filename}")

    elif filename.lower().endswith(".csv"):
        loader = CSVLoader(path)
        docs.extend(loader.load())
        print(f"📊 Loaded CSV: {filename}")

print(f"✅ Total docs loaded: {len(docs)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)

# ✅ Persist Chroma (don’t rebuild every restart)
kb_vectorstore = Chroma.from_documents(
    texts=texts,
    embedding=embedding,
    persist_directory="./reva_realestate_kb"
)

memory_vectorstore = Chroma(
    persist_directory="./reva_chat_memory",
    embedding_function=embedding
)

# ✅ API Request/Response Models
class ChatRequest(BaseModel):
    message: str
    session_id: str = None
    history: list = []

class ChatResponse(BaseModel):
    reply: str
    session_id: str

# ✅ Chat Endpoint
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())

    kb_docs = kb_vectorstore.similarity_search(request.message, k=3)
    kb_context = "\n\n".join(doc.page_content for doc in kb_docs)

    memory_docs = memory_vectorstore.similarity_search(request.message, k=3)
    memory_context = "\n\n".join(doc.page_content for doc in memory_docs)

    system_prompt = f"""
You are REVA, a helpful real estate chatbot. 
Use the Knowledge Base and memory below to answer.

Knowledge:
{kb_context}

Past Memory:
{memory_context}
"""

    full_prompt = system_prompt + "\nUser: " + request.message

    response = chat_model.invoke(full_prompt)
    reply = response.content.strip()

    if "I’m REVA" not in reply:
        memory_vectorstore.add_texts([f"{request.message}"])

    return ChatResponse(reply=reply, session_id=session_id)


@app.get("/")
def index():
    return {"status": "✅ REVA Gemini RAG API Running!"}
