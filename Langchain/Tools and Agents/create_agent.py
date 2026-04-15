from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from datetime import datetime
from langchain_core.messages import HumanMessage
import requests

load_dotenv()

@tool("search_weather")
def search_weather(city: str):
    """This function actually fetches the weather from the internet."""
    try:
        response = requests.get(f"https://wttr.in/{city.lower()}?format=j1", timeout=20)
        response.raise_for_status()
        data = response.json()
        temp = data["current_condition"][0]["temp_C"]
        condition = data["current_condition"][0]["weatherDesc"][0]["value"]
        return f"The temperature in {city} is {temp}°C and the condition is {condition}."

    except requests.exceptions.ConnectTimeout:
        return f"Error: Connection to weather service timed out. Check your internet connection."
    except requests.exceptions.ConnectionError:
        return f"Error: Could not connect to the internet."
    except requests.exceptions.HTTPError:
        return f"Error: City '{city}' not found."
    except Exception as e:
        print(f"Unexpected error: {e}")
        return f"Error: Something went wrong."

@tool("get_current_time")
def get_current_time():
    """This function returns the current time."""
    
    return f"The current time is {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}."

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

agent = create_agent(llm, [search_weather, get_current_time])

response = agent.invoke({"messages": [HumanMessage(content="What is the weather in New Delhi and what time is it?")]})
print("🤖: ",response['messages'][-1].content[0]['text'])