
from dotenv import load_dotenv

from agno.os import AgentOS

from financial_agent import (
    db,
    web_search_agent,
    financial_agent,
    multi_agent,
)

load_dotenv()

# Register agents AND the Team Leader
agent_os = AgentOS(
    name="Financial AI Research",
    description="Multi-agent financial research platform",

    agents=[
        web_search_agent,
        financial_agent,
    ],

    teams=[
        multi_agent,
    ],

    db=db,
)

# FastAPI application
app = agent_os.get_app()


if __name__ == "__main__":
    agent_os.serve(
        app="playground:app",
        reload=True,
    )
