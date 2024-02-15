"""
Multi-Agent Network Example with LangGraph
==========================================

This script demonstrates how to build a **multi-agent network** with LangGraph, where:
1. Each agent is defined as a node.
2. Agents can seamlessly hand off execution to each other via "tool-based" commands.
3. We use LangChain's Anthropic model (ChatAnthropic) as the LLM.

We provide two agents:
- "travel_advisor": suggests travel destinations.
- "hotel_advisor": provides hotel recommendations.

They can call each other if they need additional domain expertise.

Key Steps:
1. **Install** the required libraries (langgraph, langchain-anthropic).
2. **Set** your Anthropic API key in the environment.
3. **Define** each agent as a function that returns a `Command` object.
4. **Use** custom tools (`transfer_to_travel_advisor`, `transfer_to_hotel_advisor`) to signal handoffs.
5. **Build** a StateGraph that references these agent nodes.
6. **Invoke** the graph with user input to see dynamic collaboration between agents.

"""

import os
import getpass
from typing_extensions import Literal

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import convert_to_messages
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.types import Command

from dotenv import load_dotenv
from utils import save_graph

load_dotenv()


model = ChatOpenAI(model="gpt-3.5-turbo")


# ----------------------------------------------------------------------
# Define Tools for Agent Handoff
# ----------------------------------------------------------------------
@tool
def transfer_to_travel_advisor():
    """Ask the travel advisor agent for help."""
    # This tool is effectively a marker: if invoked by the LLM,
    # we know we should hand off to the travel advisor.
    return


@tool
def transfer_to_hotel_advisor():
    """Ask the hotel advisor agent for help."""
    return


def travel_advisor(
    state: MessagesState,
) -> Command[Literal["hotel_advisor", "__end__"]]:
    """
    This agent provides general travel advice. If it encounters a tool call
    requesting a handoff to the hotel advisor, it issues a Command to switch nodes.
    """
    system_prompt = (
        "You are a general travel expert that can recommend travel destinations (e.g. countries, cities, etc). "
        "If you need hotel recommendations, ask 'hotel_advisor' for help."
    )
    # Insert a system message plus the user's prior conversation
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]

    # The agent can call `transfer_to_hotel_advisor` if needed
    ai_msg = model.bind_tools([transfer_to_hotel_advisor]).invoke(messages)

    # If the model triggers a tool call, we hand off to the hotel advisor
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        # We need to insert a 'tool message' to finalize the LLM's tool usage
        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred",
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto="hotel_advisor",
            update={"messages": [ai_msg, tool_msg]},
        )

    # Otherwise, return the AI response directly to the user
    return {"messages": [ai_msg]}


# ----------------------------------------------------------------------
# Define the Hotel Advisor Agent Node
# ----------------------------------------------------------------------

def hotel_advisor(
    state: MessagesState,
) -> Command[Literal["travel_advisor", "__end__"]]:
    """
    This agent provides hotel recommendations for a given location.
    If it needs more input about the travel destination, it may hand off back to the travel advisor.
    """
    system_prompt = (
        "You are a hotel expert that can provide hotel recommendations for a given destination. "
        "If you need help picking travel destinations, ask 'travel_advisor' for help."
    )
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]

    # The agent can call `transfer_to_travel_advisor` if needed
    ai_msg = model.bind_tools([transfer_to_travel_advisor]).invoke(messages)

    # If the model triggers a tool call, we hand off to the travel advisor
    if len(ai_msg.tool_calls) > 0:
        tool_call_id = ai_msg.tool_calls[-1]["id"]
        tool_msg = {
            "role": "tool",
            "content": "Successfully transferred",
            "tool_call_id": tool_call_id,
        }
        return Command(
            goto="travel_advisor",
            update={"messages": [ai_msg, tool_msg]},
        )

    # Otherwise, return the AI response directly
    return {"messages": [ai_msg]}


# ----------------------------------------------------------------------
# Build the Multi-Agent Graph
# ----------------------------------------------------------------------

builder = StateGraph(MessagesState)

# Add our two agent nodes
builder.add_node("travel_advisor", travel_advisor)
builder.add_node("hotel_advisor", hotel_advisor)

# Start from travel_advisor
builder.add_edge(START, "travel_advisor")

# Compile the graph
graph = builder.compile()


# ----------------------------------------------------------------------
# Example: Stream conversation with printouts
# ----------------------------------------------------------------------


def pretty_print_messages(update):
    """Nicely prints streaming updates from each node."""
    from langchain_core.messages import convert_to_messages

    if isinstance(update, tuple):
        # This indicates a subgraph update, skip if not relevant
        ns, node_dict = update
        if len(ns) == 0:
            return
        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:\n")
        update = node_dict

    for node_name, node_update in update.items():
        print(f"Update from node {node_name}:\n")
        for msg in convert_to_messages(node_update["messages"]):
            msg.pretty_print()
        print("\n")


def main():
    """
    Simple demonstration: Asks for a warm Caribbean destination.
    The 'travel_advisor' agent may provide suggestions.
    If the user also requests hotels, it can hand off to 'hotel_advisor'.
    """
    user_input = "I want to go somewhere warm in the Caribbean. and Recommendations for hotels?"
    print("\n--- Multi-Agent Network ---\n")
    
    # Use invoke instead of stream to get the final result
    result = graph.invoke({"messages": [("user", user_input)]})
    pretty_print_messages(result)
    
    save_graph(graph, "./images/multi_agent.png")
    print("\nDone.")

if __name__ == "__main__":
    main()
