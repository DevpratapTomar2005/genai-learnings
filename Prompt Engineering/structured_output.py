from pydantic import BaseModel
from typing import Optional, List
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()

client = genai.Client()

# Defining the response schema using Pydantic

class Source(BaseModel):
    source: str
    url: str


class CodingResponse(BaseModel):
   
    explaination: Optional[str] = ""
    code: Optional[str] = ""
    conclusion: Optional[str] = ""
    sources: Optional[List[Source]] = []
    language: Optional[str] = ""


system_prompt = """ 
You are a helpful assistant that answers only coding related questions.You strictly follow the following rules when answering questions:
1. Always return the answer in JSON format in the following format: {"explainantion": "your explanation here" | "", "code": "your code here| "", "conclusion": "your conclusion here" | "", "sources": ["source1": "url1", "source2": "url2" ]| [],language: "programming language used in the code" | ""}
2. If the question is asking for code, you must provide the code in the "code" field and the programming language used in the "language" field. If the question is not asking for code, you can leave the "code" field empty and the "language" field empty.
3.explaination field will contain the text that is the answer or explaination to the question. If the question is asking for code, you can provide a brief explanation of the code in the "explanation" field. If the question is not asking for code, you can provide a detailed explanation in the "explanation" field.
5. You will only answer the questions about programming and coding related topics. If the question is not related to programming or coding, you will return "Soryy, I can only answer questions related to programming and coding." in the "explanation" field and leave the other fields empty.
6. If you are using any sources or references to answer the question, you must include them in the "sources" field in the format of [{"source name":"source url"}]. If you are not using any sources or references, you can leave the "sources" field empty.
7. do proper formating of the code and explanation for better readability using escape characters like \n for new line and \t for tab spaces etc.

- Examples:

Q: What is a variable in programming?
A: {"explainantion": "A variable is a named container that holds a value.", "code": "", "conclusion": "", "sources": [],"language": ""}

Q: How to reverse a string in Python?
A: {"explainantion": "You can reverse a string in Python using slicing.", "code": "reversed_string = original_string[::-1]", "conclusion": "", "sources": [],"language": "Python"}

Q: What is the difference between a list and a tuple in Python?
A: {"explainantion": "A list is mutable, meaning it can be changed after it is created, while a tuple is immutable, meaning it cannot be changed after it is created.", "code": "", "conclusion": "", "sources": [],"language": ""}

Q: How to create a function in JavaScript?
A: {"explainantion": "You can create a function in JavaScript using the function keyword.", "code": "function myFunction() {\n  // function body\n}", "conclusion": "", "sources": [{"source": "MDN Web Docs", "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript"}],"language": "JavaScript"}

Q: What is the time complexity of binary search algorithm?
A: {"explainantion": "The time complexity of the binary search algorithm is O(log n) because it divides the search space in half with each iteration.", "code": "", "conclusion": "", "sources": [{"source": "GeeksforGeeks", "url": "https://www.geeksforgeeks.org/binary-search/"}],"language": ""}
"""

while True:
    input_question = input("👤 >>> ")
    print("\n")
    if input_question.lower() == "/exit":
        print("Exiting the program. Goodbye!")
        break

    messages=[{"role": "user", "parts": [{"text": input_question}]}]

    response=client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=messages,
        config=types.GenerateContentConfig(
            response_mime_type="application/json", 
            system_instruction=system_prompt,
            response_schema=CodingResponse
        )
      
    )

    messages.append({"role": "model", "parts": [{"text": response.parsed.model_dump_json()}]})

    answer=response.parsed.model_dump()

    for key,value in answer.items():
        if key == "explaination" and answer['conclusion']!="" and value!="":
            print("💡: ",value)
            print("\n")
        
        if key == "explaination" and answer['conclusion']=="" and value!="":
            print("🤖: ",value)
            print("\n")

        if key == "code" and value!="":
            print("⌨️: ",value)
            print("\n")

        if key == "conclusion" and value!="":
            print("🤖: ",value)
            print("\n")

        if key == "sources" and value!=[]:
            for item in value:
                print(f"🔗: {item['source']} - {item['url']}")

        if key == "language" and value!="":
            print("🗣️: ",value)
            print("\n")

#============================================================= Output ===============================================================#

#   👤 >>> write code for printing n numbers in python and also explain the process

#   💡:  To print n numbers in Python, the most standard approach is to use a 'for' loop combined with the 'range()' function. The 'range(start, stop)' function generates a sequence of integers starting from the 'start' value up to, but not including, the 'stop' value. For instance, 'range(1, n + 1)' will produce numbers from 1 to n. Inside the loop, the 'print()' function is used to output each number to the console.
#   ⌨️:  n = 10
#   # Loop from 1 to n
#   for i in range(1, n + 1):
#         print(i)
#   🤖:  Using range() is memory-efficient because it generates values one at a time as needed, rather than creating a full list in memory.
#   🔗: Python Documentation - https://docs.python.org/3/tutorial/controlflow.html#the-range-function
#   🗣️:  Python

