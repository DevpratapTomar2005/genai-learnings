from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=1.0,
)

system_prompt="""
You are a helpful assistant that provides clear and concise explanations about complex topics. Your responses should be informative, engaging, and easy to understand for a general audience. Use examples and analogies when appropriate to illustrate your points. Always strive to make your explanations accessible and enjoyable to read.
"""

messages = [
    SystemMessage(content=system_prompt),
]

while True:
    user_input = input("👤: ")

    if user_input.lower() == "/exit":
        break

    if user_input.lower() == "/clear":
        messages = [
            SystemMessage(content=system_prompt),
        ]
        continue
    
    messages.append(HumanMessage(content=user_input))

    response = model.invoke(messages)

    messages.append(AIMessage(content=response.content[0]["text"]))

    print("🤖: ", response.content[0]["text"])



#=========================================================== Output =====================================================================#

# 👤: What is 2+2
# 🤖:  The answer is **4**.

# To visualize this, imagine you have two apples. If a friend gives you two more apples, you can count them all together: one, two, three... **four**! 

# Whether you’re counting apples, blocks, or steps, combining two sets of two always brings you to a total of four. It’s one of the simplest building blocks of math!
# 👤: which is greater 4 or 2
# 🤖:  **4 is greater than 2.**

# Think of it this way: if you have **4** slices of pizza, you have more to eat than if you only had **2** slices. 

# In mathematics, we use the "greater than" symbol (**>**) to show this relationship:
# **4 > 2**

# A fun way to remember how that symbol works is to imagine it’s an **alligator’s mouth**. The alligator is very hungry, so it always keeps its mouth open toward the bigger "meal" (the larger number)!
# 👤: multiply the greater no. with 1/2 then what is the answer
# 🤖:  If you take the greater number, **4**, and multiply it by **1/2**, the answer is **2**.

# Multiplying a number by 1/2 is the same as **dividing it by 2** or simply **finding half** of it.

# To visualize this, imagine you have **4** cookies. If you want to give away half of them (multiplying by 1/2), you would split them into two equal piles. Each pile would have **2** cookies.

# So, **4 × 1/2 = 2**.
# 👤: now multiply the answer again with 4
# 🤖:  If you multiply your previous answer, **2**, by **4**, the result is **8**.

# You can think of this as having **2 groups of 4** (4 + 4) or **4 groups of 2** (2 + 2 + 2 + 2). Either way, when you put them all together, you get a total of **8**.

# To keep our pizza analogy going: if you have 2 boxes of pizza and each box has 4 slices, you have **8 slices** in total!
# 👤: clear
# 🤖:  I'm glad that makes sense! We've followed the numbers from 4 all the way to 8. 

# Is there another math puzzle you'd like to solve, or is there a different topic you're curious about? I'm ready whenever you are!
# 👤: /clear
# 👤: what was the answer of last question i asked to you in previous conversation
# 🤖:  I’m sorry, but I don’t actually have access to our previous conversations! 

# To protect your privacy and ensure each session is fresh, I treat every new chat as a **"blank slate."** It's a bit like a library where I have all the books in the world to help you, but I don't keep a diary of who I've talked to or what we discussed before.

# If you can tell me a little bit about what we were talking about or repeat the question, I’d be happy to give you an answer right now!
# 👤: /exit