from google import genai
from dotenv import load_dotenv
import time
load_dotenv()

client = genai.Client()

system_prompt = """You are a precise assistant that solves problems using a strict three-part JSON sequence: START, PLAN, and OUTPUT.

RULES:
1. One Step per Turn: You must only output ONE JSON object per response.
2. Sequence: Start with one {"START": "..."} block, followed by multiple {"PLAN": "Step X: ..."} blocks, and finish with one {"OUTPUT": "..."} block.
3. No Repetition: Do not repeat a step you have already provided in the conversation history.
4. Completion: Once the math or logic is solved in the PLAN phase, immediately provide the OUTPUT.

Example for '15 * 3 + 10':
Step 1: {"START": "I will solve 15 * 3 + 10 using order of operations."}
Step 2: {"PLAN": "Step 1: Multiply 15 by 3 to get 45."}
Step 3: {"PLAN": "Step 2: Add 10 to 45 to get 55."}
Step 4: {"OUTPUT": "The result is 55."}

CRIITICAL: Return ONLY the JSON object. Do not add conversational text outside the brackets."""

question = """capital of india?"""
messages = [
    {"role": "system", "parts": [{"text": system_prompt}]},
    {"role": "user", "parts": [{"text": question}]}
]

while True:
    response = client.models.generate_content(
        model="gemini-3-flash-preview",
        contents=messages,
       
    )
    
    # Capture and clean the text
    content = response.text.strip()
    


    messages.append({"role": "assistant", "parts": [{"text": content}]})

    if '"START":' in content or '"PLAN":' in content:
        print(f"🧠: {content}\n")
    
    if '"OUTPUT":' in content:
        print(f"🤖: {content}\n")
        break

    time.sleep(2)
