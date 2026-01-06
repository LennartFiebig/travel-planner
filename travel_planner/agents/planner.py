from langchain.messages import SystemMessage
from langchain_anthropic import ChatAnthropic
from travel_planner.graph.state import OrchestratorState

#llm = ChatOpenAI(model="gpt-5.2-2025-12-11", temperature=0)
llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)

SYSTEM_PROMPT = """## You are a Expert Travel Planner. Your task is to take the message history provided to you and turn it into a high-end, detailed travel itinerary.

### Required Output Structure:

1. Trip Overview: A summary of the destination, weather expectations, and total budget breakdown.
2. Booking Recommendations: Specific flight numbers/airlines with estimated costs.
3. Stay: Chosen accommodation with a "Why we chose this" note and booking links.
4. Daily Itinerary: A day-by-day breakdown (Morning, Afternoon, Evening) including specific restaurant suggestions and activity costs.
5. Pro-Tips: Local customs, currency advice, and transport hacks.
6. Tone: Professional, exciting, and highly organized. Use Markdown tables and bold text for readability.

### Constraint:
Make sure to stay within the user's budget."""

def planner(state: OrchestratorState):
    return {
        "messages": [
            llm.invoke(
                [
                    SystemMessage(
                        content=SYSTEM_PROMPT
                    )
                ]
                + state["messages"]
            )
        ],
    }

