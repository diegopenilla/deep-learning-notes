import os
import getpass
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from utils import save_graph
from dotenv import load_dotenv

load_dotenv()


# Initialize the LLM model
model = ChatOpenAI(model="gpt-4o", temperature=0)


@tool
def get_weather(city: Literal["nyc", "sf"]) -> str:
    """Fetches predefined weather information for NYC or SF."""
    if city == "nyc":
        return "It might be cloudy in NYC."
    elif city == "sf":
        return "It's always sunny in SF."
    else:
        raise ValueError("Unknown city")


# List of tools
tools = [get_weather]


class WeatherResponse(BaseModel):
    """Structured response format for weather conditions."""
    conditions: str = Field(description="Weather conditions in the specified city.")


# Create a structured ReAct agent
graph = create_react_agent(
    model,
    tools=tools,
    response_format=WeatherResponse,  # Define structured output
)


def query_weather(city: str) -> WeatherResponse:
    """Queries the ReAct agent for weather information in a given city."""
    inputs = {"messages": [("user", f"What's the weather in {city}?")]}
    response = graph.invoke(inputs)
    return response["structured_response"]


# Example Usage
if __name__ == "__main__":
    city = "nyc"
    weather_info = query_weather(city)
    print(weather_info)

    # Example of using a customized response format
    graph = create_react_agent(
        model,
        tools=tools,
        response_format=("Always return capitalized weather conditions", WeatherResponse),
    )
    weather_info_capitalized = query_weather(city)
    print(weather_info_capitalized)
    
    save_graph(graph, "./images/structured_output_react.json")
