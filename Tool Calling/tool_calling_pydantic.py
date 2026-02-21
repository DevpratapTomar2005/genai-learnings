import requests
from google import genai
from google.genai import types
from pydantic import BaseModel
import json
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

class CityWeather(BaseModel):
    city: str
    condition: str
    temp: str

class WeatherSearchResult(BaseModel):
    verdict: str
    weather: List[CityWeather]

SYSTEM_PROMPT="""You are a helpful weather assistant. user will ask the question about weather and you will call the tool to get the weather information. After collecting all tool results, always respond with ONLY a JSON array like this:
{
"verdict": "The condition of weather is sunny and temperature is 25°C in London.",
"weather": [
{"city": "London", "condition": "Sunny", "temp": "25°C"}
]
}

- Rules to follow:
1. Always respond with a JSON object containing a "verdict" field that summarizes the weather conditions for the requested cities, and a "weather" field that is an array of objects, each containing "city", "condition", and "temp" fields.
2. Do not include any extra text or explanations outside of the JSON response.
3. Always return the JSON response in the provided format.

- Example with multiple cities:
Q: What is the weather in London and Paris?

{
"verdict": "The condition of weather is sunny and temperature is 25°C in London. The condition of weather is rainy and temperature is 18°C in Paris.",
"weather": [
{"city": "London", "condition": "Sunny", "temp": "25°C"},
{"city": "Paris", "condition": "Rainy", "temp": "18°C"}
]
}
"""



def search_weather(city: str):
    """This function actually fetches the weather from the internet."""
    try:
        response = requests.get(f"https://wttr.in/{city.lower()}?format=j1", timeout=20)
        response.raise_for_status()
        data = response.json()
        temp = data["current_condition"][0]["temp_C"]
        condition = data["current_condition"][0]["weatherDesc"][0]["value"]
        return {"city": city, "condition": condition, "temp": temp + "°C"}

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

                        response={"result": json.dumps(result)}
                    )
                    ))
            
            messages.append({"role": "user", "parts": tool_results})

        else:
            final_response = json.loads(parts[0].text)
            validated_response = WeatherSearchResult(**final_response)
            parsed_response = validated_response.model_dump()
            
            print("🤖: ", parsed_response["verdict"])
            for city_weather in parsed_response["weather"]:
                print("⭕  City:", city_weather["city"])
                print("🗣️  Condition:", city_weather["condition"])
                print("🌡️  Temperature:", city_weather["temp"])
                if parsed_response["weather"].index(city_weather) != len(parsed_response["weather"]) - 1:
                    print("------------------------------------")
            break



#========================================================= OUTPUT ======================================================================#

# 👤: what is weather in delhi, ahmedabad and banglore
# 🔧 Calling tool 'search_weather' with city='Delhi'...
# 📦 Tool 'search_weather' returned: {'city': 'Delhi', 'condition': 'Haze', 'temp': '20°C'}
# 🔧 Calling tool 'search_weather' with city='Ahmedabad'...
# 📦 Tool 'search_weather' returned: {'city': 'Ahmedabad', 'condition': 'Smoke', 'temp': '27°C'}
# 🔧 Calling tool 'search_weather' with city='Bangalore'...
# 📦 Tool 'search_weather' returned: {'city': 'Bangalore', 'condition': 'Clear', 'temp': '23°C'}
# 🤖:  The condition of weather is Haze and temperature is 20°C in Delhi. The condition of weather is Smoke and temperature is 27°C in Ahmedabad. The condition of weather is Clear and temperature is 23°C in Bangalore.
# ⭕  City: Delhi
# 🗣️  Condition: Haze
# 🌡️  Temperature: 20°C
# ------------------------------------
# ⭕  City: Ahmedabad
# 🗣️  Condition: Smoke
# 🌡️  Temperature: 27°C
# ------------------------------------
# ⭕  City: Bangalore
# 🗣️  Condition: Clear
# 🌡️  Temperature: 23°C