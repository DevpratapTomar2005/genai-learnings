from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

class Product(BaseModel):
    name: str = Field(..., description="The name of the product")
    price: float = Field(..., description="The price of the product")
    description: str = Field(..., description="A description of the product")
    rating: float = Field(..., description="The average rating of the product")

class ModelOutput(BaseModel):
    verdict: str = Field(..., description="The verdict of the model's analysis")
    products: list[Product] = Field(..., description="A list of products")


template = ChatPromptTemplate(
    [
        ("system", "You are a helpful assistant that provides product recommendations based on user queries."),
        ("placeholder", "{chat_history}"),
        ("human", "{user_input}")
    ]
)


model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1.0)

structured_model = model.with_structured_output(ModelOutput)

chat_history = []



while True:
    user_input = input("👤: ")

    if user_input.lower() == "/exit":
        break

    if user_input.lower() == "/clear":
        chat_history = []
        continue

    prompt_template = template.invoke(
        {
            "chat_history":chat_history,
            "user_input": user_input
        }
    )

    response = structured_model.invoke(prompt_template)

    chat_history.append(("human", user_input))
    chat_history.append(("ai", response.model_dump_json()))

    parsed_response = response.model_dump()

    print("🤖: ", parsed_response["verdict"])
    for product in parsed_response["products"]:
        print(f"Product Name: {product['name']}")
        print(f"Price: {product['price']}")
        print(f"Description: {product['description']}")
        print(f"Rating: {product['rating']} ⭐")

        if parsed_response["products"].index(product) != len(parsed_response["products"]) - 1:
            print("------------------------------------")

    

   



