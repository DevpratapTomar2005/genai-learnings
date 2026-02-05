from google import genai
from dotenv import load_dotenv

load_dotenv()

client = genai.Client()

prompt = """
        Give me the sentiment of the following text:
        Text: "The product is cool"

        examples:
        Text: "I am so happy and excited about my new job!"
        Sentiment: Positive

        Text: "This is the worst movie I have ever seen."
        Sentiment: Negative

        Text: "I am feeling okay, not too bad but not great either."
        Sentiment: Neutral

        Text: "I can't wait for the weekend! It's going to be amazing."
        Sentiment: Positive

"""

messages = [{"role": "user", "parts":[{"text": prompt}]}]

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents=messages
)

messages.append({"role": "assistant", "parts":[{"text": response.text}]})

for message in messages:
    role_icon = "👤" if message["role"] == "user" else "🤖"
    print(f"{role_icon}: {message['parts'][0]['text']}")