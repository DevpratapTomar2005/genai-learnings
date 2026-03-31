import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Implementation of similarity search retriever from in-memory(local) vector store using Chroma
# This is also called retrieval phase.
# We can also use other vector stores like FAISS, Pinecone, Weaviate, etc. but here we are using Chroma which is an open-source vector database that can be used locally or in the cloud. All vector store follow the same interface, so you can easily switch between them without changing your code in LangChain.

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

vector_store = Chroma(
    collection_name="python_basics",
    embedding_function=embeddings,
    persist_directory=os.path.join(BASE_DIR, "../embeddings"),
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2}  
)

query = "What are the common data types in Python?"
retrieved_docs = retriever.invoke(query)

for doc in retrieved_docs:
    print(doc, end="\n\n")
 