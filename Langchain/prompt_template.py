from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=1.0,
)

system_prompt="""
You are a helpful assistant that is an expert in {topic}. You reply with clear and concise explainations and only answer about this {topic} related questions. if the question is not about {topic} then you reply with "I am sorry but i can only answer questions related to {topic}.

User Question: {question}
"""

prompt_template = PromptTemplate(
    input_variables=["topic", "question"],
    template=system_prompt,
)



while True:

    user_input = input("👤: ")

    if user_input.lower() == "/exit":
        break
    
    prompt = prompt_template.format(topic="coding", question=user_input)

    response = model.invoke(prompt)

    print("🤖: ", response.content[0]["text"])




#=========================================================== Output =====================================================================#

# 👤: What is variable?
# 🤖:  A variable is a named storage location in computer memory used to hold data that can be referenced and manipulated during a program's execution. 

# Think of it as a labeled container: 
# *   **The Name (Identifier):** How you refer to the container (e.g., `userAge`).
# *   **The Value:** The data stored inside the container (e.g., `25`).

# In code, you typically declare a variable and assign it a value like this:
# ```python
# age = 25
# ``` 
# Here, `age` is the variable name, and `25` is the data stored within it.
# 👤: what is 2+2
# 🤖:  I am sorry but i can only answer questions related to coding.
# 👤: /exit