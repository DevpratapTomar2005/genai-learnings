from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
import os

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1.0)

system_message = """You are a helpful assistant which replies the question asked by the user based on the given text from the pdf file."""

prompt = ChatPromptTemplate(
    [
    ("system", system_message),
   
    ("human", """The text is: \n {file_text} \n and the question is "{question}" """)
    ]
)

loader = PyPDFLoader(file_path=os.path.join(BASE_DIR, "docs", "ds_notes.pdf"))

docs = loader.load()

parser = StrOutputParser()

doc_text = [doc.page_content for doc in docs]

while True:

    user_input = input("👤: ")

    if user_input.lower() == "/exit":
        break


    chain = prompt | model | parser

    result = chain.invoke(
        {
            "file_text": doc_text,
            
            "question": user_input
            
        }
    )

    
    print(f"🤖: {result}")



#=========================================================== Output ====================================================================#

# 👤: what is the first unit in file
# 🤖: The first unit mentioned in the "File" section is a **file**, which is defined as "a collection of records". A record, in turn, consists of one or more **fields**.
# 👤: I meant what is the name of first unit in pdf file
# 🤖: The name of the first unit in the PDF file is "Introduction to Data Structure".
# 👤: explain what is data structure
# 🤖: Based on the provided text, a data structure is:

# *   A representation of the logical relationship existing between individual elements of data.
# *   A way of organizing all data items that considers not only the elements stored but also their relationship to each other.
# *   A mathematical or logical model of a particular organization of data items.
# *   Defined as the way of storing and manipulating data in an organized form so that it can be used efficiently.

# It mainly specifies:
# *   Organization of Data
# *   Accessing methods
# *   Degree of associativity
# *   Processing alternatives for information

# The text also states that a program can be understood as "Algorithm + Data Structure = Program".
# The study of data structures covers the amount of memory and time required to process data, its representation in memory, and the operations performed on that data.
# 👤: what is teh page number of first unit first page
# 🤖: The first page of the first unit, "Introduction to Data Structure," is page 1. This is indicated by "2130702 – Data Structure 1" in the header.
# 👤: /exit