from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
import requests

# Set your OpenAI key
import os
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Tool 1: Get weather
def get_weather(city: str) -> str:
    # Replace with a real weather API if desired
    return f"The weather in {city} is sunny with 25°C."

# Tool 2: Get news
def get_news() -> str:
    return "Top news: AI is revolutionizing everything."

# Tool 3: Tell a joke
def tell_joke() -> str:
    return "Why did the scarecrow win an award? Because he was outstanding in his field."

# LangChain Tools
tools = [
    Tool(
        name="GetWeather",
        func=lambda city: get_weather(city),
        description="Use this to get the weather for a given city. Input should be the city name."
    ),
    Tool(
        name="GetNews",
        func=lambda _: get_news(),
        description="Use this to get the latest news. Input is ignored."
    ),
    Tool(
        name="TellJoke",
        func=lambda _: tell_joke(),
        description="Use this to hear a random joke. Input is ignored."
    )
]

# Initialize LangChain agent
llm = ChatOpenAI(temperature=0, model="gpt-4")  # or "gpt-3.5-turbo" for cheaper runs

agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Example queries
print(agent.run("What's the weather in Tokyo?"))
print(agent.run("Tell me a joke"))
print(agent.run("Give me the latest news"))
