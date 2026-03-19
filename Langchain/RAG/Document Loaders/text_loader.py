from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
import os

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1.0)

system_message = """You are a helpful assistant which replies the question asked by the user based on the given text from the text file."""

prompt = ChatPromptTemplate(
    [
    ("system", system_message),
    ("placeholder", "{chat_history}"),
    ("human", """The text is: \n {file_text} \n and the question is "{question}" """)
    ]
)

loader = TextLoader(file_path=os.path.join(BASE_DIR, "../docs", "python_basics.txt"))

docs = loader.load()

parser = StrOutputParser()

chat_history = []

while True:

    user_input = input("👤: ")

    if user_input.lower() == "/exit":
        break

    if user_input.lower() == "/clear":
        chat_history = []
        continue

    chain = prompt | model | parser

    result = chain.invoke(
        {
            "file_text": docs[0].page_content,
            
            "question": user_input,
            "chat_history": chat_history,
        }
    )

    chat_history.append(("human", user_input))
    chat_history.append(("ai", result))

    print(f"🤖: {result}")

#========================================================== Output =====================================================================#

# 👤: what is topic no. 12
# 🤖: Topic no. 12 is **ERROR HANDLING**.
# 👤: what is it   
# 🤖: This document is a **Python Basics Guide**. It covers the core building blocks of the Python programming language, including variables, data types, strings, numbers, lists, tuples, dictionaries, sets, conditionals, loops, functions, classes & objects, error handling, file handling, built-in functions, and modules. It also provides quick reference and tips for beginners.
# 👤: explain error handling with examples mentioned in this file
# 🤖: Based on the provided text, **ERROR HANDLING** (Topic No. 12) is explained as follows:

# **EXPLANATION:**
# Errors (exceptions) occur when something goes wrong at runtime — like dividing by zero or accessing a missing file. Instead of crashing, Python lets you catch and handle these errors gracefully using try/except blocks.

# The structure for error handling includes:
# *   **`try`**: Code that might cause an error.
# *   **`except`**: Runs if a specific error occurs.
# *   **`else`**: Runs only if NO error occurred in `try`.
# *   **`finally`**: Always runs, whether an error happened or not (great for cleanup tasks like closing files).
# *   **`raise`**: Manually trigger an exception with a message.

# Common exceptions mentioned are:
# *   `ZeroDivisionError`: Division by zero.
# *   `ValueError`: Invalid value for a function.
# *   `TypeError`: Wrong type used in an operation.
# *   `FileNotFoundError`: File does not exist.
# *   `KeyError`: Key not found in a dictionary.
# *   `IndexError`: Index out of range in a list.

# **EXAMPLES MENTIONED IN THE FILE:**

# ```python
# try:
#     result = 10 / 0
# except ZeroDivisionError:
#     print("Cannot divide by zero!")
# except ValueError as e:
#     print(f"Value error: {e}")
# else:
#     print("No error occurred.")   # Only runs if try succeeded
# finally:
#     print("This always runs.")    # Cleanup code goes here

# # Raise a custom exception manually
# def check_age(age):
#     if age < 0:
#         raise ValueError("Age cannot be negative.")
#     return age
# ```
# 👤: /exit