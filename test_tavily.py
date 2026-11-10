
import os
from dotenv import load_dotenv
from agno.agent import Agent
from agno.models.groq import Groq
from agno.tools.tavily import TavilyTools

load_dotenv()

if not os.getenv("TAVILY_API_KEY"):
    raise ValueError("TAVILY_API_KEY missing from .env")

agent = Agent(
    model=Groq(id="openai/gpt-oss-20b"),
    tools=[TavilyTools()],
    instructions=[
        "Search using Tavily.",
        "Return actual search results and source URLs.",
        "Never invent information."
    ],
    markdown=True,
)

agent.print_response(
    "Find the latest Apple company news. Include source URLs.",
    stream=True,
)
