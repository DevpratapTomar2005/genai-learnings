
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Implementation of MMR (Maximal Marginal Relevance) search retriever using Chroma vector store and Google Gemini models. MMR search is a technique used to diversify the results returned by a retriever. It aims to balance relevance and diversity by selecting documents that are not only relevant to the query but also different from each other. This can help in scenarios where you want to get a broader range of information or perspectives on a topic, rather than just the most similar documents.

load_dotenv()

docs = [
    Document(
    page_content= "The universe is the totality of all space, time, matter, and energy in existence.",
    metadata={"source": "test_source", "author": "Devpratap Tomar"},
),
    Document(
    page_content= "The universe is very random and chaotic, and it is difficult to predict what will happen in the future.",
    metadata={"source": "test_source", "author": "Devpratap Tomar"},
),
Document(
    page_content= "The Earth is the third planet from the Sun and the only known astronomical object to harbor life.",
    metadata={"source": "test_source", "author": "Devpratap Tomar"},
),
Document(
    page_content= "The speed of light in a vacuum is approximately 299,792 kilometers per second (186,282 miles per second).",
    metadata={"source": "test_source", "author": "Devpratap Tomar"},
),
Document(
    page_content= "totality of all space, time, matter, and energy in existence is the universe.",
    metadata={"source": "test_source", "author": "Devpratap Tomar"},
)
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)

chuncks = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

vector_store = Chroma.from_documents(
    documents=chuncks,
    embedding=embeddings
)


retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2})

query = "What is the universe?"
results = retriever.invoke(query)

print(results)


#============================================================= Output =================================================================#
 

## Without MMR Search:
# [Document(id='4314aa8f-c636-4b90-b786-c4dfa1a2c049', metadata={'source': 'test_source', 'author': 'Devpratap Tomar'}, page_content='The universe is the totality of all space, time, matter, and energy in existence.'), Document(id='2656fe78-38c4-4528-8caa-4aa9042e34d7', metadata={'source': 'test_source', 'author': 'Devpratap Tomar'}, page_content='totality of all space, time, matter, and energy in existence is the universe.')]

## With MMR Search:
# [Document(id='400dde54-fdac-46fb-8f5f-664cef99853d', metadata={'author': 'Devpratap Tomar', 'source': 'test_source'}, page_content='The universe is the totality of all space, time, matter, and energy in existence.'), Document(id='f791d3f9-81d2-482e-9486-05b59fe859e7', metadata={'source': 'test_source', 'author': 'Devpratap Tomar'}, page_content='The universe is very random and chaotic, and it is difficult to predict what will happen in the future.')]