
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import MultiQueryRetriever

# Implementation of multi-query retriever using Chroma vector store and Google Gemini models.



load_dotenv()

docs = [
    Document(
        page_content="Fitness is more than just how you look; it’s about your body's ability to handle daily tasks with energy and ease.",
        metadata={"source": "test_source", "author": "Devpratap Tomar"},
    ),
    Document(
        page_content="Maintaining a regular routine can significantly lower your risk of chronic conditions like heart disease, type 2 diabetes, and certain cancers.",
        metadata={"source": "test_source", "author": "Devpratap Tomar"},
    ),
    Document(
        page_content="Physical activity also offers a powerful mental boost, as it releases endorphins that can help reduce feelings of stress, anxiety, and depression.",
        metadata={"source": "test_source", "author": "Devpratap Tomar"},
    ),
    Document(
        page_content="Experts generally suggest getting at least 150 minutes of moderate aerobic exercise and two days of muscle-strengthening work each week.",
        metadata={"source": "test_source", "author": "Devpratap Tomar"},
    ),
    Document(
        page_content="Beyond standard workouts, staying fit involves building an active lifestyle, such as taking the stairs or going for daily walks to keep your heart and muscles strong.",
        metadata={"source": "test_source", "author": "Devpratap Tomar"},
    ),
]

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)

chuncks = text_splitter.split_documents(docs)

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

vector_store = Chroma.from_documents(
    documents=chuncks,
    embedding=embeddings
)


similarity_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=similarity_retriever, 
    llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1.0)
)

query = "Tips for staying healthy"
results = multiquery_retriever.invoke(query)

print(results)


#============================================================= Output =================================================================#
 

#[Document(id='0f16ad95-1ab1-4e7a-a0d4-043167952c4e', metadata={'author': 'Devpratap Tomar', 'source': 'test_source'}, page_content='Maintaining a regular routine can significantly lower your risk of chronic conditions like heart disease, type 2 diabetes, and certain cancers.'), Document(id='57b6049a-bc62-468c-b8f1-6d44f309171d', metadata={'source': 'test_source', 'author': 'Devpratap Tomar'}, page_content='Beyond standard workouts, staying fit involves building an active lifestyle, such as taking the stairs or going for daily walks to keep your heart and muscles strong.')]