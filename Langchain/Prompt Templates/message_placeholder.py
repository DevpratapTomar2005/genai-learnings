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
      ("placeholder", "{chat_history}"),
      ("human", "{question}")
    ]
)

chat_history = []

while True:

    user_input = input("👤: ")
  
    if user_input.lower() == "/exit":
        break
    
    prompt = prompt_template.invoke(
    {
    "topic": "mathematics",
    "question": user_input,
    "chat_history": chat_history
    }
    )

    response = model.invoke(prompt.messages)

    print("🤖: ", response.content[0]["text"])
    chat_history.append(("human", user_input))
    chat_history.append(("ai", response.content[0]["text"]))


#============================================================ Output ===================================================================#

# 👤: what is 2 + 2
# 🤖:  2 + 2 = 4.
# 👤: now multiply the answer with 4
# 🤖:  4 × 4 = 16.
# 👤: now divide it by two
# 🤖:  16 ÷ 2 = 8.
# 👤: what was the first question i asked you in this chat
# 🤖:  The first question you asked was "what is 2 + 2".
# 👤: /exit