
import os
from datetime import date
from dotenv import load_dotenv

from agno.agent import Agent
from agno.team import Team
from agno.team.mode import TeamMode
from agno.models.groq import Groq
from agno.tools.yfinance import YFinanceTools
from agno.tools.tavily import TavilyTools
from agno.db.sqlite import SqliteDb

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY is missing from .env")

# Shared model and database
def get_model():
    return Groq(id="openai/gpt-oss-20b")


db = SqliteDb(db_file="agents.db")


# 1. Web Research Agent
web_search_agent = Agent(
    id="web-search-agent",
    name="WebSearchAgent",
    role="Research recent company news and financial information online",
    model=get_model(),
    tools=[
        TavilyTools(
            enable_search=True,
            enable_extract=True,
        )
    ],
    instructions=[
        "Find recent company news and earnings announcements.",
        "Prefer official investor relations sources.",
        "Return up to 3 relevant results.",
        "Include dates and source URLs.",
        "Never invent information."
    ],
    db=db,
    markdown=True,
)


# 2. Financial Agent
financial_agent = Agent(
    id="financial-agent",
    name="FinancialAgent",
    role="Retrieve stock prices, fundamentals and analyst ratings",
    model=get_model(),
    tools=[
        YFinanceTools(
            enable_stock_price=True,
            enable_analyst_recommendations=True,
            enable_company_info=True,
            enable_stock_fundamentals=True,
            enable_company_news=True,
        )
    ],
    instructions=[
        "Retrieve stock prices and financial fundamentals.",
        "Get analyst recommendations when available.",
        "Show buy, hold and sell ratings when available.",
        "Use concise Markdown tables.",
        "Never invent financial figures.",
        "Identify unavailable data."
    ],
    db=db,
    markdown=True,
)


# 3. Team Leader
multi_agent = Team(
    id="financial-research-team",
    name="FinancialResearchTeam",
    role="Coordinate financial and web research agents and produce a final report",
    model=get_model(),
    members=[
        financial_agent,
        web_search_agent,
    ],
    mode=TeamMode.coordinate,
    instructions=[
        "You are the financial research team leader.",
        "Delegate financial data retrieval to FinancialAgent.",
        "Delegate recent news research to WebSearchAgent.",
        "Combine both agents' findings into one report.",
        "Include source URLs.",
        "Identify missing or conflicting information.",
        "Never invent financial data.",
        "Keep the final report concise."
    ],
    db=db,
    markdown=True,
)


# 4. Test the Team directly
if __name__ == "__main__":
    query = f"""
    Research Apple Inc. (AAPL).

    Date: {date.today()}

    Get:
    1. Stock price
    2. Analyst recommendations
    3. Financial fundamentals
    4. Recent company news
    5. Short analysis
    6. Source URLs

    Report unavailable information clearly.
    """

    multi_agent.print_response(query, stream=False)
