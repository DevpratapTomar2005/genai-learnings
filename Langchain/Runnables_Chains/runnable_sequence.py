from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

load_dotenv()

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1.0)


class Report(BaseModel):
    topic: str = Field(description="The topic to search and summarize")
    report: str = Field(description="The detailed report on the topic")

class ReportSummary(BaseModel):
    summary: str = Field(description="The summary of the report")

structured_model1 = model.with_structured_output(Report)
structured_model2 = model.with_structured_output(ReportSummary)

template1 = PromptTemplate(
    template="""You are a helpful assistant that search the web for information or to prepare report on topic "{topic}".
    """,
    input_variables=["topic"]
)

template2 = PromptTemplate(
    template="""Based on the report recieved as the input you procide a coincise summary in bulletpoints as the output for the topic "{topic}". \nFormat the text properly by using proper escape sequences for newlines and other formatting characters.\n The report is: \n {report}.
    """,
    input_variables=["topic", "report"]
)

def report_to_summary_input(report: Report) -> dict:
    return {"topic": report.topic, "report": report.report}

runnable_sequence_report = template1 | structured_model1


runnable_sequence_summary = template2 | structured_model2

final_chain = runnable_sequence_report | RunnableLambda(report_to_summary_input) | runnable_sequence_summary

topic = input("👤 Enter the topic you want to search and summarize: ")

result = final_chain.invoke({"topic": topic})

print("🤖: ", result.model_dump()["summary"])



#============================================================== Output =================================================================#

# 👤 Enter the topic you want to search and summarize: ChatGPT VS Claude
# 🤖:  ChatGPT VS Claude: A Summary

# *   **Introduction:** Both ChatGPT (OpenAI) and Claude (Anthropic) are leading LLMs, utilizing transformer architecture but differing in training and focus.

# *   **Underlying Philosophy:**
#     *   **ChatGPT (OpenAI):** Focuses on massive data training and RLHF for broad general knowledge and continuous improvement (GPT-3.5, GPT-4, GPT-4o).
#     *   **Claude (Anthropic):** Employs 'Constitutional AI' with built-in principles for helpfulness, harmlessness, and honesty, prioritizing safety and ethical alignment.

# *   **Key Strengths:**
#     *   **ChatGPT:**
#         *   Broad General Knowledge & Versatility: Excels across topics, creative content, summarization, translation.
#         *   Coding Prowess: Strong for writing, debugging, explaining code.
#         *   Multimodality: Newer versions (GPT-4V, GPT-4o) handle images, audio, video.
#         *   Integration: Widely integrated into applications.
#         *   Speed & Efficiency: GPT-4o is fast and cost-effective.
#     *   **Claude:**
#         *   Long Context Windows: Processes vast amounts of text (e.g., entire books) for complex analysis, summarization, and retrieval.
#         *   Enhanced Safety & Harmlessness: Designed for robust ethical alignment, less prone to harmful output.
#         *   Nuanced & Empathetic Conversations: Maintains coherent, less robotic dialogue.
#         *   Reasoning over Complex Texts: Excels at extracting insights and performing complex reasoning in intricate documents.

# *   **Weaknesses & Limitations:**
#     *   **ChatGPT:**
#         *   Occasional 'Hallucinations': Can generate plausible but incorrect information.
#         *   Verbosity: May be overly verbose.
#         *   Transparency of Safety: Less transparent compared to Constitutional AI.
#     *   **Claude:**
#         *   Overly Cautious: Strong safety focus can lead to conservative responses or refusal to answer.
#         *   General Knowledge: Historically slightly less comprehensive for niche/recent topics, but improving.
#         *   Coding (Historically): Historically slightly behind GPT models for complex coding, though Claude 3 has improved significantly.

# *   **Ideal Use Cases:**
#     *   **ChatGPT:** Content creation, coding assistance, customer support, educational tools, general research, brainstorming, creative writing, multimodal applications.    
#     *   **Claude:** Legal/scientific/financial document review, long-form content generation, highly sensitive conversational AI, ethical AI development, complex data extraction from large datasets.

# *   **Conclusion:** Choice depends on the task: ChatGPT for a versatile generalist with coding/multimodal needs; Claude for vast text processing, deep contextual understanding, and high safety/ethical alignment. Leveraging both can be beneficial.
