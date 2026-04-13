from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from dotenv import load_dotenv

load_dotenv()

@tool("get_current_weather")
def get_current_weather(location: str) -> str:
    """Get the current weather in a given location"""
    # For the sake of this example, we'll just return a dummy weather report.
    return {"location": location, "temperature": "20°C", "description": "Sunny"}

@tool("generate_report")
def generate_report(location: str, temperature: str, description: str) -> str:
    """Generate a weather report based on the weather data."""
    return f"The current weather in {location} is {description} with a temperature of {temperature}."

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0.9)

llm_with_tools = llm.bind_tools([get_current_weather, generate_report])

system_message = """You are a helpful assistant that can use tools to get the current weather and generate a report. You have these tools available:
1. get_current_weather: This tool takes a location as input and returns the current weather in that location.
2. generate_report: This tool takes a location, temperature, and description as input and generates a weather report based on that data."""

query = "First get the weather for New York, then generate a report."

messages = [SystemMessage(content=system_message), HumanMessage(content=query)]

while True:
    response = llm_with_tools.invoke(messages)

    messages.append(response)

    if response.tool_calls == [] and response.content:
        print(f"🤖: {response.content[0]['text']}")
        break


    for tool_call in response.tool_calls:
        print(f"🔨Tool call: {tool_call['name']} with args {tool_call['args']}...")
        if tool_call["name"] == "get_current_weather":
            location = tool_call["args"]["location"]
            weather_report = get_current_weather.invoke({"location": location})
            tool_response = ToolMessage(
                        content=weather_report,
                        tool_name="get_current_weather",
                        tool_args={"location": location},
                        tool_call_id=tool_call["id"],
            )

            messages.append(tool_response)
            
        if tool_call["name"] == "generate_report":
            location = tool_call["args"]["location"]
            temperature = tool_call["args"]["temperature"]
            description = tool_call["args"]["description"]
            report = generate_report.invoke({"location": location, "temperature": temperature, "description": description})
            tool_response = ToolMessage(
                        content=report,
                        tool_name="generate_report",
                        tool_args={"location": location, "temperature": temperature, "description": description},
                        tool_call_id=tool_call["id"],
                    )

            messages.append(tool_response)
        print(f"📦Tool calling completed...")


#============================================================ Output ==================================================================#

# 🔨Tool call: get_current_weather with args {'location': 'New York'}...
# 📦Tool calling completed...
# 🔨Tool call: generate_report with args {'temperature': '20°C', 'description': 'Sunny', 'location': 'New York'}...
# 📦Tool calling completed...
# 🤖: The current weather in New York is Sunny with a temperature of 20°C.