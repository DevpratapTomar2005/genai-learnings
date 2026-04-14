from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor,create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_classic import hub

load_dotenv()



search_tool = DuckDuckGoSearchRun()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=1.0)

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(
    llm=llm,
    tools=[search_tool],
    prompt=prompt
)

agent_executor = AgentExecutor(agent=agent, tools=[search_tool], verbose=True)

if __name__ == "__main__":
    query = "Write an article on ReACT loop in ai agents."
    result = agent_executor.invoke({"input": query})
    print(result)

#========================================================== Output ====================================================================#

# Action: duckduckgo_search

# Action Input: ReACT loop AI agents explanationThe AI agent loop is the iterative execution cycle at the core of every agentic AI system. At each iteration, the agent assembles context from available inputs, invokes an LLM to reason and select an action, executes that action, observes the outcome, and feeds the observation back into the next iteration. ReAct Loop and Tool Integration Relevant source files This page details the iterative reasoning-action cycle in AgenticRetrieveOp and how tools are registered, exposed to the LLM, executed, and integrated into the agent's workflow. This covers the core execution loop, tool execution mechanics, parallel processing, and message accumulation across multiple rounds. For the overall agent ... Understand the core concept of AI Agents: simple loops combining LLMs and tools to perform tasks. A practical breakdown with code examples. Build ReAct agents with LangGraph using hardcoded logic and LLM-powered reasoning to create adaptive AI systems. An AI agent works through an iterative loop where it perceives an input, reasons about it, executes an action with external tools and observes the result before deciding the next step. This cycle, known as ReAct, is what separates an agent from a simple language model that generates text. In this article you'll understand the five internal components of an AI agent's architecture, how the ...Action: generate_article

# Action Input: The ReACT loop in AI agents is an iterative execution cycle where an agent perceives an input, reasons about it using a Large Language Model (LLM), selects and executes an action with external tools, observes the outcome, and feeds this observation back into the next iteration. This cycle is what distinguishes an agent from a simple language model. The core components involve assembling context, invoking an LLM for reasoning and action selection, executing the action, observing the result, and integrating this observation for subsequent steps. It often involves tool integration, where tools are registered, exposed to the LLM, executed, and their outputs are integrated into the agent's workflow. This loop allows agents to perform tasks adaptively.TheReACTloopinAIagentsisaniterativeexecutioncyclewhereanagentperceivesaninput,reasonsaboutitusingaLargeLanguageModel(LLM),selectsandexecutesanactionwithexternaltools,observes the outcome, and feeds this observation back into the next iteration. This cycle is what distinguishes an agent from a simple language model. The core components involve assembling context, invoking an LLM for reasoning and action selection, executing the action, observing the result, and integrating this observation for subsequent steps. It often involves tool integration, where tools are registered, exposed to the LLM, executed, and their outputs are integrated into the agent's workflow. This loop allows agents to perform tasks adaptively.

# Final Answer: The ReACT loop in AI agents is an iterative execution cycle where an agent perceives an input, reasons about it using a Large Language Model (LLM), selects and executes an action with external tools, observes the outcome, and feeds this observation back into the next iteration. This cycle is what distinguishes an agent from a simple language model. The core components involve assembling context, invoking an LLM for reasoning and action selection, executing the action, observing the result, and integrating this observation for subsequent steps. It often involves tool integration, where tools are registered, exposed to the LLM, executed, and their outputs are integrated into the agent's workflow. This loop allows agents to perform tasks adaptively.

# > Finished chain.
# {'input': 'Write an article on ReACT loop in ai agents.', 'output': "The ReACT loop in AI agents is an iterative execution cycle where an agent perceives an input, reasons about it using a Large Language Model (LLM), selects and executes an action with external tools, observes the outcome, and feeds this observation back into the next iteration. This cycle is what distinguishes an agent from a simple language model. The core components involve assembling context, invoking an LLM for reasoning and action selection, executing the action, observing the result, and integrating this observation for subsequent steps. It often involves tool integration, where tools are registered, exposed to the LLM, executed, and their outputs are integrated into the agent's workflow. This loop allows agents to perform tasks adaptively."}