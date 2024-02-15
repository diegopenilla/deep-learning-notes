"""
LangGraph Few-Shot Learning Node (Minimal)
==========================================

This script defines a single node ("rewriter_node") that demonstrates how to
provide multiple few-shot examples for polite rewrites in an LLM prompt.
"""

import os
import getpass
from typing_extensions import Literal

from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START, END
from langgraph.types import Command
from utils import save_graph

from dotenv import load_dotenv

load_dotenv()


# Initialize the LLM
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)

def rewriter_node(state: MessagesState) -> Command[Literal["__end__"]]:
    """
    A node that uses few-shot examples to guide the LLM in producing
    polite rewrites of user-provided sentences.
    """

    few_shot_examples = [
        """Example 1:
User: "Rewrite this sentence more politely: 'Give me water now!'"
Assistant: "Could you please give me some water?\"""",
        """Example 2:
User: "Rewrite this sentence more politely: 'Move over.'"
Assistant: "Could you please move over?\"""",
        """Example 3:
User: "Rewrite this sentence more politely: 'Stop talking.'"
Assistant: "Could you please stop talking?\"""",
        """Example 4:
User: "Rewrite this sentence more politely: 'Clean this mess up!'"
Assistant: "Could you please help clean up this mess?\"""",
    ]

    # Combine examples into one system prompt
    few_shot_prompt = (
        "You are a helpful rewriting assistant. "
        "Below are examples of rewriting sentences into a more polite form.\n\n"
        + "\n\n".join(few_shot_examples)
    )

    # The final message list
    messages = [
        {"role": "system", "content": few_shot_prompt},
        *state["messages"],
    ]

    # Invoke the LLM
    ai_response = model.invoke(messages)

    # Return the final AI message
    return {"messages": [ai_response]}

# Build the minimal LangGraph pipeline
builder = StateGraph(MessagesState)
builder.add_node("rewriter_node", rewriter_node)
builder.set_entry_point("rewriter_node")
builder.add_edge("rewriter_node", END)

graph = builder.compile()


def main():
    """
    Simple demonstration of polite rewriting using few-shot learning.
    The 'rewriter_node' transforms impolite requests into polite ones.
    """
    user_input = "Give me the bottle of water immediately!"
    print("\n--- Polite Rewriter ---\n")
    print(f"Original input: \"{user_input}\"\n")
    
    # Use invoke instead of stream to get just the final result
    result = graph.invoke({"messages": [("user", user_input)]})
    
    # Extract and print the final message content
    if "messages" in result and result["messages"]:
        final_message = result["messages"][-1]
        print(f"Polite version: \"{final_message.content}\"")
    else:
        print("No response generated.")
    
    print("\nDone.")

    save_graph(graph, "./images/few_shot_learning.png")

if __name__ == "__main__":
    main()
