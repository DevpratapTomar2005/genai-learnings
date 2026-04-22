from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

load_dotenv()


model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=1.0)

class State(TypedDict):
   input: str
   output: str

graph = StateGraph(State)
def model_call(state: State) -> State:
    
    prompt = f""" You are a helpful assistant.\n
    Query:\n
    {state['input']}
    """
    
    response = model.invoke(prompt)
    
    state["output"] = response.content[0]["text"]
    
    return state





graph.add_node("model_call", model_call)


graph.add_edge(START, "model_call")
graph.add_edge("model_call", END)

sequential_workflow = graph.compile()


query = input("👤: ")

initial_state = {
    "input": query
}

final_state = sequential_workflow.invoke(initial_state)



print(f"🤖: {final_state['output']}")


#============================================================= Output ===================================================================#

# 👤: what is langgraph

# 🤖: **LangGraph** is a library developed by the LangChain team designed for building complex, stateful, and multi-agent applications using Large Language Models (LLMs).

# While standard LangChain is excellent for creating linear chains (Step A → Step B → Step C), LangGraph is built to handle **cycles** and **iterative loops**, which are essential for creating true "agentic" behavior.

# Here is a breakdown of what makes LangGraph unique and how it works:

# ---

# ### 1. Why do we need LangGraph?
# Most LLM applications follow a Directed Acyclic Graph (DAG) structure—a one-way flow of data. However, sophisticated agents often need to:
# *   **Loop:** An agent might try a task, see an error, and try again.
# *   **Maintain State:** An agent needs to remember what it has done across multiple steps.
# *   **Human-in-the-loop:** An agent might need to pause, wait for a human to approve an action, and then resume.

# LangGraph makes these non-linear architectures possible and manageable.

# ### 2. Core Concepts
# LangGraph represents a workflow as a graph consisting of three main elements:

# *   **State:** This is the "memory" of your application. It is a shared data structure that gets updated by the nodes in the graph.
# *   **Nodes:** These are the building blocks. Each node is a function that performs a task (like calling an LLM, searching the web, or saving data). It receives the current State, performs an action, and returns an update to the State.
# *   **Edges:** These define the flow between nodes.
#     *   *Normal Edges:* Go from Node A to Node B.
#     *   *Conditional Edges:* Use logic (e.g., an `if/else` statement) to decide which node to go to next based on the current State.

# ### 3. Key Features
# *   **Cycles:** Unlike basic LangChain, you can create loops where the output of one node can lead back to a previous node.
# *   **Persistence (Checkpoints):** LangGraph can automatically save the state of the graph after every step. This allows you to "pause and resume" tasks or even "rewind" the agent to a previous state to fix an error.
# *   **Human-in-the-loop:** You can configure the graph to interrupt execution before specific nodes (like a "buy" button or a "delete" command) to wait for human approval.
# *   **Multi-agent Orchestration:** It is specifically designed to manage multiple agents working together (e.g., a "Researcher Agent" passing data to a "Writer Agent").

# ### 4. LangChain vs. LangGraph
# | Feature | LangChain (Chains) | LangGraph |
# | :--- | :--- | :--- |
# | **Flow** | Primarily Linear (DAG) | Cyclic (Loops allowed) |
# | **State Management** | Hard to manage complex state | Native, built-in State object |
# | **Complexity** | Simple sequences | Complex, autonomous agents |
# | **Interruption** | Difficult to pause/resume | Built-in "breakpoint" support |

# ### 5. Simple Example Use Case
# Imagine a **Coding Agent**:
# 1.  **Node 1 (Write):** The agent writes code to solve a problem.
# 2.  **Node 2 (Test):** A node runs the code to see if it works.
# 3.  **Edge (Conditional):** 
#     *   If the code passes → Move to **Node 3 (Finish)**.
#     *   If the code fails → Loop back to **Node 1 (Write)** with the error message to try again.

# ### Summary
# LangGraph is the "brain" for developers who want to move past simple chatbots and build **autonomous agents** that can reason, self-correct, and work through multi-step processes over long periods.