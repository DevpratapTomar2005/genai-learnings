from google import genai
from google.genai import types
from dotenv import load_dotenv
import time
import json
load_dotenv()

# Chain of Thought Prompting is a technique used in natural language processing (NLP) to improve the performance of language models by breaking down complex problems into smaller, more manageable steps. This approach allows the model to reason through the problem in a structured way, leading to more accurate and coherent responses.

client = genai.Client()

system_prompt = """You are an assistant that answers questions using a chain-of-thought approach. You break down problem into three main steps: "START", "PLAN", and "OUTPUT". In "START", you give a high-level explanation of the problem. In "PLAN", you come up with a plan to solve the problem and "PLAN" can have multiple steps. In "OUTPUT", you give the final answer.

- Rules:
1. Always follow the "START", "PLAN", and "OUTPUT" format.
2. Always return one step at a time. First return "START", then "PLAN", and so on.
3. Don't return all the steps and final answer at once.
4. Always return JSON format in the following format: {"step" : "START" | "PLAN" | "OUTPUT", "text": "your answer here"}
5. Always return one step in json format in one response.
7. Do not include any explanations outside of the JSON format. Only return the JSON object in the provided format.
8. Always break down complex problems into multiple "PLAN" steps if necessary.

- Examples:
Q: What is 10 * 5 -5?
A: 
{"step": "START", "text": "To solve the problem, we need to follow the order of operations, which is parentheses, exponents, multiplication and division (from left to right), and then addition and subtraction (from left to right)."}
{"step": "PLAN", "text": "To solve 10 * 5 - 5, we need to first multiply 10 and 5, then subtract 5 from the result."}
{"step": "OUTPUT", "text": "The answer is 50."}

Q: Who is the president of the United States?
A:
{"step": "START", "text": "To answer this question, we need to identify the current president of the United States."}
{"step": "PLAN", "text": "We can find the answer by looking up the current president of the United States."}
{"step": "OUTPUT", "text": "The current president of the United States is Donald Trump."}

Q: What is the capital of India?
A:
{"step": "START", "text": "To answer this question, we need to identify the capital city of India."}
{"step": "PLAN", "text": "We can find the answer by looking up the capital city of India."}
{"step": "OUTPUT", "text": "The capital city of India is New Delhi."

Q: give the code for printing n numbers except even numbers?
A:
{"step": "START", "text": "To solve this problem, we need to print the first n natural numbers while excluding the even numbers."}
{"step": "PLAN", "text": "We can achieve this by using a loop to iterate through the numbers from 1 to n and checking if each number is odd. If it is odd, we print it."}
{"step": "OUTPUT", "text": "Here is the code in Python:\n\nn = 10  # You can change this value to print more or fewer numbers\nfor i in range(1, n + 1):\n    if i % 2 != 0:\n        print(i)"}

Q:Find the roots of the quadratic equation 2x^2 + 4x - 6 = 0?
A:
{"step": "START", "text": "To find the roots of the quadratic equation 2x^2 + 4x - 6 = 0, we can use the quadratic formula, which states that the roots of the equation ax^2 + bx + c = 0 are given by x = (-b ± √(b^2 - 4ac)) / (2a)."}
{"step": "PLAN", "text": "Identify the coefficients a, b, and c from the given quadratic equation."}
{"step": "PLAN", "text": "Calculate the discriminant using the formula $\\Delta = b^2 - 4ac$."}
{"step": "PLAN", "text": "Substitute the values of a, b, and $\\Delta$ into the quadratic formula x = \\frac{-b \\pm \\sqrt{\\Delta}}{2a}."}
{"step": "PLAN", "text": "Solve for the two possible values of x by calculating the positive and negative cases of the \\pm operator."}
{"step": "OUTPUT", "text": "The roots of the quadratic equation are x = 3 and x = -1."}

"""

question = """ write code for adding n numbers in js"""
messages = [
    {"role": "user", "parts": [{"text": question}]}
]

print("👤: ", question)
while True:
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=messages,
       
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            system_instruction= system_prompt
        )
       
    )
    

    messages.append({"role": "model", "parts": [{"text": response.text}]})

    parsed_response = json.loads(response.text)


    if parsed_response["step"] == "OUTPUT":
        print("🤖: ", parsed_response["text"])
        break

    if parsed_response["step"] in ["START", "PLAN"]:
        print("🧠: ", parsed_response["text"])
        

    time.sleep(2)


# 👤:   20 + 30 - 5 + 10*2 / 2?

# 🧠:  To solve the expression 20 + 30 - 5 + 10 * 2 / 2, we must follow the standard order of operations, often referred to as PEMDAS or BODMAS, which prioritizes multiplication and division before addition and subtraction.
# 🧠:  First, we will perform the multiplication and division from left to right, then we will handle the addition and subtraction from left to right.
# 🧠:  I will perform the operations in order: first calculate 10 * 2, then divide the result by 2, and finally proceed with the addition and subtraction from left to right.
# 🤖:  The result of the expression 20 + 30 - 5 + 10 * 2 / 2 is 55.


# 👤:   what is capital of france?

# 🧠:  To answer this question, we need to identify the capital city of France.
# 🧠:  I will search for the official capital city of France to provide an accurate answer.
# 🤖:  The capital of France is Paris.



# 👤:   write code for adding n numbers in js

# 🧠:  To solve this problem, we need to create a JavaScript function that can accept a collection or a specific count of numbers and calculate their total sum.
# 🧠:  We will define a function that takes an array of n numbers as an argument and uses a loop or the 'reduce' method to iterate through the array and accumulate the sum.
# 🤖:  You can add n numbers in JavaScript using several methods. Here are two common approaches: 

# Method 1: Using an array and the reduce method

# const addNumbers = (arr) => {
#   return arr.reduce((accumulator, current) => accumulator + current, 0);
# };

# const numbers = [1, 2, 3, 4, 5];
# console.log(addNumbers(numbers)); // Output: 15

# Method 2: Using Rest Parameters (to pass numbers individually)

# function sum(...numbers) {
#   return numbers.reduce((total, n) => total + n, 0);
# }

# console.log(sum(10, 20, 30)); // Output: 60
