"""
This script sets up an arithmetic assistant using LangChain, OpenAI's ChatGPT, and a computation graph.
It defines arithmetic operations as tools (addition, multiplication, and division) and integrates them with an LLM.

1. Define arithmetic functions.
2. Bind functions to a LangChain-powered agent.
3. Create a computation graph to handle arithmetic requests.
4. Implement an interactive CLI for testing.
"""
import os
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from utils import save_graph


from dotenv import load_dotenv
load_dotenv()

def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b

def divide(a: int, b: int) -> float:
    """Divides a by b."""
    if b == 0:
        return "Error: Division by zero"
    return a / b

tools = [add, multiply, divide]

llm = ChatOpenAI(model="gpt-3.5-turbo")
llm_with_tools = llm.bind_tools(tools)

sys_msg = SystemMessage(content="You are a helpful assistant tasked with performing arithmetic on a set of inputs.")

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

graph = builder.compile()

def main():
    print("\n🧮 Arithmetic Assistant 🧮\n")
    print("Computing: 25 * 30 / 24")
    
    user_input = "25 * 30 / 24"
    messages = [HumanMessage(content=user_input)]
    state = {"messages": messages}
    
    # Option 1: Use invoke() for just the final result
    result = graph.invoke(state)
    final_message = result["messages"][-1]
    print("Result:", final_message.content)
    
    save_graph(graph, "./images/basic_agent.png")
    
    
if __name__ == "__main__":
    main()
