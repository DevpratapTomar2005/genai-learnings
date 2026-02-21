import requests
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

SYSTEM_PROMPT="""You are a helpful weather assistant. user will ask the question about weather and you will call the tool to get the weather information. After collecting all tool results, always respond with ONLY a JSON array like this:
[{"city": "London", "condition": "Sunny", "temp": "25°C"}]
No extra text. Just the JSON array."""

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
    

weather_search_tool=types.Tool(function_declarations=[{
    "name": "search_weather",
    "description": "Get the current weather for a city.",
    "parameters": {
        "type": types.Type.OBJECT,
        "properties": {
            "city": {"type": types.Type.STRING}
        },
        "required": ["city"],
    }
}])

while True:
    user_input=input("👤: ")

    if user_input.lower() in ["/exit", "/quit"]:
        print("🤖: Goodbye!")
        break

    messages=[{"role": "user", "parts": [{"text": user_input}]}]

    while True:

        response=client.models.generate_content(
            model="gemini-3-flash-preview",
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                tools=[weather_search_tool]
            )
        )

        parts=response.candidates[0].content.parts

        function_calls=[part for part in parts if part.function_call]

        if function_calls:
            
            messages.append({"role": "model", "parts": parts})
            tool_results=[]
            for part in function_calls:
                
                if part.function_call.name=="search_weather":
                    city=part.function_call.args["city"]
                    print(f"🔧 Calling tool 'search_weather' with city='{city}'...")
                    result=search_weather(city)
                    print(f"📦 Tool 'search_weather' returned: {result}")
                    tool_results.append(types.Part(function_response=types.FunctionResponse(
                        name=part.function_call.name,

                        response={"result": result}
                    )
                    ))
            
            messages.append({"role": "user", "parts": tool_results})

        else:
            print("🤖:", parts[0].text)
            break



#========================================================= OUTPUT ======================================================================#

# 👤: what is weather in delhi, rajasthan and ahmedabad
# 🔧 Calling tool 'search_weather' with city='Delhi'...
# 📦 Tool 'search_weather' returned: The temperature in Delhi is 22°C and the condition is Haze.
# 🔧 Calling tool 'search_weather' with city='Rajasthan'...
# 📦 Tool 'search_weather' returned: The temperature in Rajasthan is 24°C and the condition is Clear.
# 🔧 Calling tool 'search_weather' with city='Ahmedabad'...
# 📦 Tool 'search_weather' returned: The temperature in Ahmedabad is 30°C and the condition is Smoke.
# 🤖: [
#   {"city": "Delhi", "condition": "Haze", "temp": "22°C"},
#   {"city": "Rajasthan", "condition": "Clear", "temp": "24°C"},
#   {"city": "Ahmedabad", "condition": "Smoke", "temp": "30°C"}
# ]
# 👤: /exit
# 🤖: Goodbye!