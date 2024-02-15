
from langgraph.graph import StateGraph

def save_graph(graph: StateGraph, path: str):
    """Prints a Mermaid.js representation of the computation graph."""
    png_graph = graph.get_graph().draw_mermaid_png()
    with open(path, "wb") as f:
        f.write(png_graph)