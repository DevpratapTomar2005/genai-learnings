from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel


load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1.0)

template_twitter = PromptTemplate(
    template="""You are a helpful assistant that creates a post for twitter/X on the topic "{topic}". You search the web to fetch the all the latest guidelines of X and according to algorithm of X you create a post that has high chances of going viral on X. no extra text only the post content""",

    input_variables=["topic"]
)


template_linkedin = PromptTemplate(
    template="""You are a helpful assistant that creates a post for LinkedIn on the topic "{topic}". You search the web to fetch the all the latest guidelines of LinkedIn and according to algorithm of LinkedIn you create a post that has high chances of going viral on LinkedIn. no extra text only the post content""",

    input_variables=["topic"]
)

runnable_twitter = template_twitter | model
runnable_linkedin = template_linkedin | model

result = RunnableParallel({"twitter": runnable_twitter, "linkedin": runnable_linkedin}).invoke({"topic": "Generative AI"})

print ("Twitter Post: \n", result["twitter"].content)
print ("\nLinkedIn Post: \n", result["linkedin"].content)

#============================================================== Output =================================================================#

# Twitter Post: 
#  Generative AI just crossed the chasm from "cool tech" to "absolute game-changer." We're talking paradigm shifts in every industry, from design to medicine. Is this humanity's greatest co-pilot or a fast-track to existential questions? 🤔 Drop your boldest prediction! 👇

# #GenerativeAI #AI #FutureIsNow #TechRevolution

# LinkedIn Post:
#  Generative AI: Hype, Reality, or Revolution? 🤔

# It’s no longer a futuristic concept; it’s here, reshaping industries and igniting a global conversation. From transforming content creation to automating complex tasks, Generative AI's impact is undeniable and accelerating daily.

# But beyond the dazzling demos and fear-mongering headlines, where do *you* stand on its immediate future?

# *   **🚀 Innovation & Opportunity:** A game-changer for productivity, creativity, and new business models.
# *   **🚧 Challenges & Ethics:** Bias, IP, job displacement, and the need for robust governance.
# *   **💡 Practical Applications:** How quickly are businesses truly integrating it for tangible ROI?

# This isn't just a tech trend; it's a fundamental shift in how we work, create, and interact.

# **What's your primary sentiment towards Generative AI RIGHT NOW?**

# A. Pure Excitement & Ready to Dive In! 🎉
# B. Cautious Optimism – Huge Potential, but Watch Out. 🧐
# C. Significant Concerns – The Risks Outweigh the Benefits. ⚠️
# D. Still Learning & Observing – The Jury's Still Out. 💡

# Vote in the poll below and share *why* in the comments! Let's spark a real conversation. 👇

# #GenerativeAI #AI #FutureOfWork #Innovation #TechTrends