from langchain_community.chat_models import ChatOllama
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType

# Define tools (your functions)
def get_weather(city: str) -> str:
    return f"Sunny and 22°C in {city}"

def get_news() -> str:
    return "AI agents are the hottest trend in 2025 tech."

def tell_joke() -> str:
    return "Why did the developer go broke? Because he used up all his cache."

# LangChain tool wrappers
tools = [
    Tool(
        name="GetWeather",
        func=lambda city: get_weather(city),
        description="Use to get the weather for a given city. Input should be the city name."
    ),
    Tool(
        name="GetNews",
        func=lambda _: get_news(),
        description="Use to get the latest news. Input is ignored."
    ),
    Tool(
        name="TellJoke",
        func=lambda _: tell_joke(),
        description="Use to hear a random joke. Input is ignored."
    )
]

# Use local model via Ollama
llm = ChatOllama(model="mistral")  # Or "llama2", "gemma", etc.

# Initialize agent
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Example prompts
print(agent.run("What's the weather in Berlin?"))
print(agent.run("Tell me a joke"))
print(agent.run("Give me some news"))
