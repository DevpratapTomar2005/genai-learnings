from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

# Zero-Shot Prompting: It is a type of prompting in which no examples are provided to the model. The model is expected to generate a response based solely on the input prompt and its pre-existing knowledge.

messages = [
    {"role": "user", "parts": [{"text": "explain AI in simple terms"}]},
] 

response = client.models.generate_content(
     model="gemini-3-flash-preview", contents=messages
)

messages.append({"role": "assistant", "parts": [{"text": response.text}]})

for message in messages:
    role_icon = "👤" if message['role'] == 'user' else "🤖"
    print(f"{role_icon}: {message['parts'][0]['text']}\n")



#Output:
#👤: explain AI in simple terms

#🤖: At its simplest, **Artificial Intelligence (AI)** is a type of computer technology that tries to mimic the way humans think and learn.

# Here is an easy way to understand it using three simple ideas:

# ### 1. The Difference: Old Computers vs. AI
# *   **Old Computers** are like a **calculator**. They only do exactly what they are told. If you don't give them a specific "recipe" (called code), they can’t do anything.
# *   **AI** is more like a **student**. Instead of being given a recipe, it is given thousands of examples. It looks at those examples, finds patterns, and learns how to do things on its own.

# ### 2. How it learns (The "Cat" Example)
# Imagine you want to teach a computer to recognize a cat.
# *   **The old way:** You would have to write thousands of rules: "It has pointy ears," "It has a tail," "It has whiskers." But what if the cat is facing away? The computer gets confused.
# *   **The AI way:** You show the computer 100,000 photos of cats. The AI notices on its own that cats usually have certain shapes and features. Eventually, when you show it a new photo, it says, "I've seen patterns like this before—that's a cat!"

# ### 3. AI in your everyday life
# You are probably already using AI every day without realizing it:
# *   **Netflix or YouTube:** It looks at what you’ve watched in the past and says, "Based on these patterns, I bet you’ll like this show."
# *   **Siri or Alexa:** It uses AI to turn the sound of your voice into text and understand the meaning of your question.
# *   **Email Spam Filters:** It recognizes the "pattern" of a scammy email and hides it before you see it.
# *   **ChatGPT:** It has read almost everything on the internet to learn how humans talk, so it can "predict" the best words to use when answering your questions.

# ### Important Note: Does it "Think"?
# Even though it seems smart, **AI doesn't have a brain, feelings, or a soul.**

# It doesn't "know" what a cat is the way you do. It just knows that according to its data, there is a 99% chance the image is a cat. It is essentially a very, very powerful **pattern-matching machine.**

# **In short: AI is a computer program that learns from experience instead of just following instructions.**

