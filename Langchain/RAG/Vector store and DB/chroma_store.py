import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Implementation of in-memory(local) vector store using Chroma
# This is also called indexing phase.
# We can also use other vector stores like FAISS, Pinecone, Weaviate, etc. but here we are using Chroma which is an open-source vector database that can be used locally or in the cloud. All vector store follow the same interface, so you can easily switch between them without changing your code in LangChain.

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


text = """
================================================================
                     PYTHON BASICS GUIDE
================================================================
  Python is a beginner-friendly, high-level programming language
  known for its clean and readable syntax. It is widely used in
  web development, data science, automation, AI, and more.
  This guide covers the core building blocks of Python.
================================================================


----------------------------------------------------------------
1. VARIABLES & DATA TYPES
----------------------------------------------------------------
  EXPLANATION:
  A variable is a named container used to store data in memory.
  You don't need to declare the type — Python figures it out
  automatically (this is called dynamic typing). Every value in
  Python has a data type that determines what operations can be
  performed on it.

  Common data types:
  - int   : Whole numbers (e.g., 5, -10, 100)
  - float : Decimal numbers (e.g., 3.14, -0.5)
  - str   : Text (e.g., "hello", 'world')
  - bool  : True or False values
  - None  : Represents the absence of a value

# Integer
age = 25

# Float
price = 19.99

# String
name = "Alice"

# Boolean
is_active = True

# NoneType
value = None

# Check type
print(type(age))       # <class 'int'>


----------------------------------------------------------------
2. STRINGS
----------------------------------------------------------------
  EXPLANATION:
  A string is a sequence of characters enclosed in single (' ')
  or double (" ") quotes. Strings are one of the most commonly
  used data types in Python. They are immutable — once created,
  their content cannot be changed directly.

  Key concepts:
  - Indexing  : Access individual characters using [index]
  - Slicing   : Extract a portion using [start:end]
  - Methods   : Built-in functions like .upper(), .lower(),
                .replace(), .strip(), .split(), etc.
  - f-strings : A clean way to embed variables inside strings
                using the f"...{variable}..." syntax

greeting = "Hello, World!"

# String methods
print(greeting.upper())          # HELLO, WORLD!
print(greeting.lower())          # hello, world!
print(greeting.replace("Hello", "Hi"))  # Hi, World!
print(len(greeting))             # 13

# String formatting (f-string)
name = "Alice"
age = 25
print(f"My name is {name} and I am {age} years old.")

# String slicing
text = "Python"
print(text[0])      # P      -> first character
print(text[-1])     # n      -> last character
print(text[0:3])    # Pyt    -> characters from index 0 to 2


----------------------------------------------------------------
3. NUMBERS & MATH OPERATIONS
----------------------------------------------------------------
  EXPLANATION:
  Python supports two main numeric types: integers (int) and
  floating-point numbers (float). You can perform all standard
  arithmetic operations directly.

  Key operators:
  - +   : Addition
  - -   : Subtraction
  - *   : Multiplication
  - /   : Division (always returns a float)
  - //  : Floor division (rounds down to nearest whole number)
  - %   : Modulus (returns the remainder of division)
  - **  : Exponentiation (power)

  Python also follows standard order of operations (PEMDAS/BODMAS).
  Use parentheses () to control evaluation order.

a = 10
b = 3

print(a + b)    # Addition       -> 13
print(a - b)    # Subtraction    -> 7
print(a * b)    # Multiplication -> 30
print(a / b)    # Division       -> 3.333...
print(a // b)   # Floor Division -> 3
print(a % b)    # Modulus        -> 1
print(a ** b)   # Exponent       -> 1000

"""

doc = Document(
    page_content= text,
    metadata={"source": "test_source", "author": "Devpratap Tomar"},
)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)

chuncks = text_splitter.split_documents([doc])

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

vector_store = Chroma(
    collection_name="python_basics",
    embedding_function=embeddings,
    persist_directory=os.path.join(BASE_DIR, "../embeddings"),
)

vectors = vector_store.add_documents(chuncks)

print(f"Added {len(vectors)} vectors to the Chroma vector store.")
print(f"Vector IDs: {vectors}")
 