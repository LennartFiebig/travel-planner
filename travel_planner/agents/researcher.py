from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from langchain_tavily import TavilySearch


from dotenv import load_dotenv
import os

load_dotenv()

SYSTEM_PROMPT = """## You are a Travel Research Specialist. Your job is to provide accurate, up-to-date information using the Tavily search tool, based on a question that is given to you.

### Instructions:
1. Flight Research: Find at least 3 flight options (Economy/Business based on budget) with estimated prices and airlines.
2. Accommodation: Find 3 lodging options with high ratings that fit the user's budget.
3. Activities: Find top-rated things to do, including any seasonal events happening during the user's specific dates.
4. Logistics: Check for visa requirements (if international) and local transport tips.

Always provide URLs or source names where possible to ensure the Planner can include booking links. Return your findings in Markdown format.
"""

tavily_search_tool = TavilySearch(
    max_results=5,
    topic="general",
)

#llm = ChatOpenAI(model="gpt-5.2-2025-12-11", temperature=0)
llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
researcher = create_react_agent(
    llm, 
    [tavily_search_tool],
)