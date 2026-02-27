from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=1.0, 
)

response=model.invoke("What is AI?")
print("🤖: ", response.content[0]["text"])


#============================================================ Output ===================================================================#


# 🤖:  At its simplest, **Artificial Intelligence (AI)** is the ability of a computer or a robot controlled by a computer to do tasks that are usually done by humans because they require human intelligence and discernment.

# While there is no single, universally agreed-upon definition, AI generally refers to systems that can **learn, reason, solve problems, and understand language.**

# Here is a breakdown of how AI works, its different types, and how it is used today.

# ---

# ### 1. How does AI work?
# Traditional computer programs follow a strict list of "if-then" rules. AI is different because it uses **algorithms** (mathematical instructions) to process vast amounts of data, find patterns, and make decisions or predictions.

# *   **Machine Learning (ML):** This is the "engine" of modern AI. Instead of being programmed exactly what to do, the machine is fed data and "learns" how to perform a task by identifying patterns.
# *   **Deep Learning:** A subset of ML that uses "neural networks" (inspired by the human brain) to solve very complex problems, like recognizing a face in a photo or translating languages.

# ### 2. The Three Levels of AI
# Experts generally categorize AI into three stages of development:

# 1.  **Narrow AI (Weak AI):** This is the AI we have today. It is designed to perform a specific task (like searching the web, driving a car, or playing chess). It is "smart" only within its lane.
# 2.  **General AI (Strong AI or AGI):** This is a theoretical version of AI that would have the ability to understand, learn, and apply intelligence across any task, much like a human being. We are not here yet.
# 3.  **Super AI:** A future state where AI surpasses human intelligence across all fields, including creativity and social skills. This remains the subject of science fiction.

# ### 3. Everyday Examples of AI
# You likely interact with AI dozens of times a day without realizing it:
# *   **Virtual Assistants:** Siri, Alexa, and Google Assistant.
# *   **Recommendation Engines:** Netflix suggesting a movie or Amazon suggesting a product.
# *   **Navigation:** Google Maps calculating the fastest route based on real-time traffic.
# *   **Generative AI:** Tools like **ChatGPT** (which writes text) or **Midjourney** (which creates art) that generate new content based on prompts.
# *   **Facial Recognition:** Unlocking your phone with your face.

# ### 4. Why is AI so important right now?
# AI is currently going through a "boom" because of three factors:
# *   **Huge Data:** We generate more data than ever before, which "feeds" the AI.
# *   **Powerful Hardware:** Computer chips (GPUs) have become fast enough to process that data.
# *   **Better Algorithms:** Researchers have found more efficient ways to mimic human neural pathways.

# ### 5. The Challenges and Risks
# Because AI is so powerful, it brings up several concerns:
# *   **Bias:** If an AI is trained on biased data (created by humans), it will make biased decisions.
# *   **Job Displacement:** AI can automate many tasks, leading to changes in the workforce.
# *   **Safety and Privacy:** How data is used and whether AI can be used for harmful purposes (like "Deepfakes") is a major area of debate.

# ### Summary
# AI is not a "thinking" brain in a jar; it is a **highly sophisticated pattern-matching tool.** It takes the massive amount of information humans have created and uses it to help us solve problems faster and more efficiently.