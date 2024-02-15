import operator
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults

from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from datetime import datetime
from dotenv import load_dotenv
from utils import save_graph

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0) 

sharedKey = Annotated[list, operator.add]
class State(TypedDict):
    question: str
    answer: str
    context: sharedKey

# The shared key context is annotated with operator.add 
# to indicate that the context list should be appended to the existing context list.
def fetch_web_results(state: State):
    """ Retrieve docs from web search """

    # Search
    tavily_search = TavilySearchResults(max_results=3)
    search_docs = tavily_search.invoke(state['question'])

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 

def fetch_wikipedia_results(state: State):
    """ Retrieve docs from Wikipedia """

    # Search
    search_docs = WikipediaLoader(query=state['question'], 
                                  load_max_docs=2).load()

    # Format
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
            for doc in search_docs
        ]
    )

    return {"context": [formatted_search_docs]} 

def generate_response(state):
    """ Node to generate an answer """

    # Get state
    context = state["context"]
    question = state["question"]

    # Template
    answer_template = """Answer the question {question} using this context: {context}"""
    answer_instructions = answer_template.format(question=question, 
                                                       context=context)    
    
    # Answer
    answer = llm.invoke([SystemMessage(content=answer_instructions)]+[HumanMessage(content=f"Answer the question.")])
      
    # Append it to state
    return {"answer": answer}

# Add nodes
builder = StateGraph(State)

# Initialize each node with node_secret 
builder.add_node("fetch_web_results", fetch_web_results)
builder.add_node("fetch_wikipedia_results", fetch_wikipedia_results)
builder.add_node("generate_response", generate_response)

# Flow
builder.add_edge(START, "fetch_wikipedia_results")
builder.add_edge(START, "fetch_web_results")
builder.add_edge("fetch_wikipedia_results", "generate_response")
builder.add_edge("fetch_web_results", "generate_response")
builder.add_edge("generate_response", END)
graph = builder.compile()


save_graph(graph, "./images/parallel_search.png")

if __name__ == "__main__":
    inputs = {
        "question": f"What are the most viral / surprising things that happend today avoid donald trump DONT TALK ABOUT DONALD TRUMP NOTHING TO DO WITH DONALD TRUMP or political {datetime.now().strftime('%Y-%m-%d')}?",
        "context": []
    }
    
    # Run the graph
    result = graph.invoke(inputs)
    
    # Print the answer
    print("\nQuestion:", inputs["question"])
    print("\nAnswer:", result["answer"].content)