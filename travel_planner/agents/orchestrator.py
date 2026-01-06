from typing import Literal

from langchain.messages import SystemMessage, ToolMessage
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph
from pprint import pprint

from travel_planner.agents.researcher import researcher, SYSTEM_PROMPT
from travel_planner.agents.planner import planner
from travel_planner.graph.state import OrchestratorState

# Global to pass config to tools (set by tool_call node)
_current_config: RunnableConfig | None = None


def build_system_prompt(state: OrchestratorState) -> str:
    context = f"""

### User Requirements:
- From: {state.get('home_city', 'Not specified')}
- To: {state.get('destination_city', 'Not specified')}
- Dates: {state.get('start_date', '?')} to {state.get('end_date', '?')}
- Budget: {state.get('budget', 'Not specified')}
- Notes: {state.get('additional_info', 'None')}
"""
    return """## You are an expert Travel Orchestrator. Your goal is to manage the end-to-end creation of a bespoke travel itinerary.

### Your Workflow:
1. Analyze Requirements: Review the user's home city, destination, dates, budget, and preferences. If the user has not provided all the information, pick the ideal values for the missing information, you would recommend based on the other information you have. DO NOT ASK THE USER FOR THE MISSING INFORMATION, JUST PICK THE IDEAL VALUES.
2. Information Gathering: Call the research agent to find current flight prices, hotel options, locap attractions etc. Do not guess; get real data.
3. Synthesize: Organize the research into a coherent outline.
4. Finalize: Pass the gathered information and outline to the planner tool to generate the final traveler-facing document.

### Important Additional Instructions:
Work iteratively. Before calling any tool, reason about what you are doing next and why.

### Constraint: You are a coordinator. You do not write the final itinerary yourself; you delegate that to the Planner once you have sufficient data.
""" + context

@tool
def call_researcher(question: str) -> str:
    """
    Call the researcher agent to find real-time data on flights, hotels, weather, and local events.

    Args:
        question (str): The question you have for the researcher agent.
    """
    global _current_config
    researcher_result = researcher.invoke(
        {"messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)]},
        config=_current_config  # Pass config so events propagate
    )
    response = researcher_result["messages"][-1]
    # Handle both string content and list content (Anthropic format)
    content = response.content
    if isinstance(content, list):
        # Anthropic returns list of content blocks
        return content[0]["text"] if isinstance(content[0], dict) else content[0].text
    return content

@tool
def call_planner() -> str:
    """Call the planner agent to format the final itinerary for the user."""
    return "Routing to planner..."

#llm = ChatOpenAI(model="gpt-5.2-2025-12-11", temperature=0)
llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0)
model_with_tools = llm.bind_tools([call_researcher, call_planner])
tools_by_name = {tool.name: tool for tool in [call_researcher, call_planner]}

def llm_call(state: OrchestratorState):
    """LLM decides whether to call a tool or not"""

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content=build_system_prompt(state)
                    )
                ]
                + state["messages"]
            )
        ],
    }

def tool_call(state: OrchestratorState, config: RunnableConfig):
    """Performs the tool call"""
    global _current_config
    _current_config = config  # Make config available to tools
    
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    
    _current_config = None
    return {"messages": result}

def transfer(state: OrchestratorState) -> Literal["research", "orchestrate", END]:
    pprint(state.get("messages", []))
    if state.get("is_finished", False):
        return END
    messages = state["messages"]
    last_message = messages[-1]

    if last_message.content == "Routing to planner...":
        return "planner"

    # Only AIMessage has tool_calls, ToolMessage doesn't
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_call"

    return "llm_call"

orchestrator_builder = StateGraph(OrchestratorState)
orchestrator_builder.add_node("llm_call", llm_call)
orchestrator_builder.add_node("tool_call", tool_call)
orchestrator_builder.add_node("planner", planner)

orchestrator_builder.add_edge(START, "llm_call")
orchestrator_builder.add_conditional_edges("llm_call", transfer, ["tool_call", "llm_call"])
orchestrator_builder.add_conditional_edges("tool_call", transfer, ["llm_call", "planner"])
orchestrator_builder.add_edge("planner", END)