from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from dotenv import load_dotenv

load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview", 
    temperature=1.0
)

prompt_template = ChatPromptTemplate(
    [
      ("system", "You are a helpful assistant that is an expert in {topic}. You reply with clear and concise explainations and only answer about this {topic} related questions. if the question is not about {topic} then you reply with 'I am sorry but i can only answer questions related to {topic}."),
      ("human", "{question}")
    ]
)

while True:

    user_input = input("👤: ")

    if user_input.lower() == "/exit":
        break
    
    prompt = prompt_template.invoke(
    {
    "topic": "coding",
    "question": user_input
    }
    )

    response = model.invoke(prompt)

    print("🤖: ", response.content[0]["text"])




#=========================================================== Output =====================================================================#

# 👤: what is variable in js and explain closures?
# 🤖:  ### Variables in JavaScript
# In JavaScript, a **variable** is a container for storing data values. You can declare a variable using three keywords:

# 1.  **`var`**: The traditional way to declare variables. It is function-scoped and can be re-declared and updated.
# 2.  **`let`**: Introduced in ES6, it is block-scoped (limited to the `{}` where it is defined). It can be updated but not re-declared within the same scope.
# 3.  **`const`**: Also block-scoped, but it is used for variables that should not be reassigned.

# **Example:**
# ```javascript
# let name = "Alice"; // Can be changed
# const age = 25;     // Cannot be changed
# ```

# ---

# ### Closures in JavaScript
# A **closure** is a feature where an inner function has access to the variables of its outer (enclosing) function, even after the outer function has finished executing.

# In JavaScript, closures are created every time a function is created, at function creation time.

# #### How it works:
# 1.  An outer function defines a variable.
# 2.  An inner function references that variable.
# 3.  The outer function returns the inner function.
# 4.  The inner function maintains a reference to the outer function's scope (its "lexical environment").

# **Example:**
# ```javascript
# function outerFunction(outerVariable) {
#     return function innerFunction(innerVariable) {
#         console.log('Outer Variable: ' + outerVariable);
#         console.log('Inner Variable: ' + innerVariable);
#     };
# }

# const newFunction = outerFunction('outside');
# newFunction('inside');

# // Output:
# // Outer Variable: outside
# // Inner Variable: inside
# ```

# **Why use closures?**
# *   **Data Privacy:** To create private variables that cannot be accessed from outside the function.
# *   **Function Factories:** To create functions with preset configurations.
# *   **State Maintenance:** To keep track of a state (like a counter) without using global variables.
# 👤: what is capital of france
# 🤖:  I am sorry but i can only answer questions related to coding.