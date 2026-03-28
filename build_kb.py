import os
from dotenv import load_dotenv
load_dotenv()

GROQ_KEY = os.getenv("GROQ_KEY")
LANGSMITH_KEY = os.getenv("LANGSMITH_KEY")
GEMINI_KEY = os.getenv("GEMINI_KEY")

import os
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_KEY
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "engosoft-agent"

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
import shutil

print("1. Loading courses.md...")
loader = TextLoader("courses.md", encoding="utf-8")
docs = loader.load()
import re
for doc in docs:
    doc.page_content = re.sub(r'\\([#*\[\]()\-_])', r'\1', doc.page_content)
    doc.page_content = re.sub(r'\\\n', '\n', doc.page_content)
    doc.page_content = re.sub(r'\\_', '_', doc.page_content)
print("   Cleaned markdown escaping")
print(f"   Loaded {len(docs)} document")

print("2. Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=80,
    separators=["---", "\n\n", "\n"]
)
chunks = splitter.split_documents(docs)
print(f"   Created {len(chunks)} chunks")

print("3. Loading embedding model (first time: 2-3 min download)...")
embeddings = SentenceTransformerEmbeddings(
    model_name="paraphrase-multilingual-mpnet-base-v2"
)

print("4. Building ChromaDB...")
if os.path.exists("./vectordb"):
    shutil.rmtree("./vectordb")

vectordb = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./vectordb"
)
print(f"   Saved {len(chunks)} chunks to ChromaDB")

print("\n5. Testing search...")
tests = [
    "ETABS structural design",
    "تصميم الكباري",
    "Revit architecture",
]
for query in tests:
    results = vectordb.similarity_search(query, k=1)
    print(f"\n   Query: {query}")
    print(f"   Result: {results[0].page_content[:120]}...")

print("\nDone! ChromaDB ready in ./vectordb")