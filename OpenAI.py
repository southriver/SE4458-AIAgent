from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool
from dotenv import load_dotenv
import os
import requests

# Set your OpenAI key
load_dotenv()

# Configure your OpenAI key via environment variable (recommended)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
else:
    raise RuntimeError("Set OPENAI_API_KEY in your environment or .env file.")

@tool
def get_weather(city: str) -> str:
    """Get the weather for a given city."""
    return f"The weather in {city} is sunny with 25C."

@tool
def get_news() -> str:
    """Get the latest news."""
    return "Top news: AI is revolutionizing everything."

@tool
def tell_joke() -> str:
    """Hear a random joke."""
    return "Why did the scarecrow win an award? Because he was outstanding in his field."

llm = ChatOpenAI(temperature=0, model="gpt-4")  # or "gpt-3.5-turbo" for cheaper runs

tools = [get_weather, get_news, tell_joke]

agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="You are a helpful assistant that can use tools when needed.",
)

# Example queries
def run_query(question: str) -> str:
    """Send a user question to the agent and extract the latest reply text."""
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    last_message = result["messages"][-1]
    return getattr(last_message, "content", str(last_message))


print(run_query("What's the weather in Tokyo?"))
print(run_query("Tell me a joke"))
print(run_query("Give me the latest news"))
