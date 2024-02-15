from langchain_core.messages import HumanMessage, SystemMessage, RemoveMessage
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from utils import save_graph

load_dotenv()

# We will use this model for both the conversation and the summarization
from langchain_openai import ChatOpenAI
model = ChatOpenAI(model="gpt-4o", temperature=0) 

# State class to store messages and summary
class State(MessagesState):
    summary: str
    
# Define the logic to call the model
def call_model(state: State):
    
    # Get summary if it exists
    summary = state.get("summary", "")

    # If there is summary, then we add it to messages
    if summary:
        
        # Add summary to system message
        system_message = f"Summary of conversation earlier: {summary}"

        # Append summary to any newer messages
        messages = [SystemMessage(content=system_message)] + state["messages"]
    
    else:
        messages = state["messages"]
    
    response = model.invoke(messages)
    return {"messages": response}

# Determine whether to end or summarize the conversation
def should_continue(state: State):
    
    """Return the next node to execute."""
    
    messages = state["messages"]
    
    # If there are more than six messages, then we summarize the conversation
    if len(messages) > 4:
        return "summarize_conversation"
    
    # Otherwise we can just end
    return END

def summarize_conversation(state: State):
    
    # First get the summary if it exists
    summary = state.get("summary", "")

    # Create our summarization prompt 
    if summary:
        
        # If a summary already exists, add it to the prompt
        summary_message = (
            f"This is summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
        
    else:
        # If no summary exists, just create a new one
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)
    
    # Delete all but the 2 most recent messages and add our summary to the state 
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}

# Define a new graph
workflow = StateGraph(State)
workflow.add_node("conversation", call_model)
workflow.add_node(summarize_conversation)

# Set the entrypoint as conversation
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges("conversation", should_continue)
workflow.add_edge("summarize_conversation", END)

# Compile
graph = workflow.compile()

def main():

    initial_state = {
        "messages": [
            HumanMessage(content="Let's discuss the advances in AI over the past decade."),
            HumanMessage(content="How have these developments impacted everyday life?"),
            HumanMessage(content="What are the potential risks associated with AI advancements?"),
            HumanMessage(content="How can we ensure responsible AI development?"),
            HumanMessage(content="What are the key challenges in AI research today?"),
        ],
        # Initially, there's no summary.
        "summary": ""
    }
    
    # Invoke the summarizer graph (which handles conversation and summarization)
    final_state = graph.invoke(initial_state)
    
    print("\n=== Final Summarizer Output ===\n")
    # Try to print the summary if generated, otherwise print the messages.
    if final_state.get("summary"):
        print("Summary:")
        print(final_state["summary"])
    else:
        print("No summary generated. Final messages:")
        for msg in final_state.get("messages", []):
            print(msg.content)
    
    # Optionally, save the graph visualization
    save_graph(graph, "./images/summarizer.png")
    
    # print summary
    print("SUMMARY:")
    print(final_state["summary"])
    
    
    print("\nDone.")
    

if __name__ == "__main__":
    main()

