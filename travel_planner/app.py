import asyncio
import streamlit as st
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

from langfuse.langchain import CallbackHandler
from langchain_core.messages import HumanMessage
from travel_planner.agents.orchestrator import orchestrator_builder
from langfuse import get_client

langfuse = get_client()
 
# Verify connection
if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")

st.set_page_config(page_title="Travel Planner", page_icon="✈️", layout="centered")

st.title("✈️ Travel Planner")

# Sidebar inputs
with st.sidebar:
    st.header("Trip Details")
    
    home_city = st.text_input("From", placeholder="Munich")
    destination_city = st.text_input("To", placeholder="Tokyo")
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", value=date.today() + timedelta(days=30))
    with col2:
        end_date = st.date_input("End", value=date.today() + timedelta(days=37))
    
    budget = st.number_input("Budget (USD)", min_value=100, value=3000, step=100)
    additional_info = st.text_area("Preferences (optional)", placeholder="I love local food...")
    
    plan_button = st.button("Plan Trip", type="primary", use_container_width=True)


def extract_content(content: str) -> str:
    """Extract content from nested message string."""
    if content.startswith("content='"):
        start = len("content='")
        for i, char in enumerate(content[start:], start):
            if char == "'" and content[i-1] != "\\":
                return content[start:i].replace("\\n", "\n")
    return content


async def run_planner_async(home_city: str, destination_city: str, start_date: str, 
                end_date: str, budget: float, additional_info: str):
    """Run the agent with streaming updates including sub-agent progress."""
    
    langfuse_handler = CallbackHandler()
    app = orchestrator_builder.compile().with_config(config={"callbacks": [langfuse_handler]})
    
    initial_state = {
        "messages": [HumanMessage(content="Plan my trip")],
        "home_city": home_city,
        "destination_city": destination_city,
        "start_date": start_date,
        "end_date": end_date,
        "budget": budget,
        "additional_info": additional_info
    }
    
    # Unified message stream - maintains true order
    # Each item: {"type": "ai"|"tool_call"|"tool_result", "content": ..., "id": ...}
    message_stream = []
    tool_states = {}  # tool_id -> {"sub_steps": [], "result": None, "done": False}
    current_tool_id = None
    final_plan = None
    plan_streaming = False
    plan_chunks = []
    
    # UI containers
    status_placeholder = st.empty()
    messages_container = st.container()
    planner_status_container = st.container()
    plan_container = st.container()
    
    messages_placeholder = messages_container.empty()
    planner_status_placeholder = planner_status_container.empty()
    plan_preview_placeholder = planner_status_container.empty()
    
    def extract_ai_text(content) -> str:
        """Extract text from AI message content (handles both string and list formats)."""
        if isinstance(content, str):
            return content
        elif isinstance(content, list):
            texts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    texts.append(block.get("text", ""))
                elif hasattr(block, "text"):
                    texts.append(block.text)
            return "\n".join(texts)
        return str(content)
    
    def render_messages():
        """Render all messages in order."""
        with messages_placeholder.container():
            for msg in message_stream:
                if msg["type"] == "ai":
                    with st.chat_message("assistant", avatar="🧠"):
                        st.markdown(msg["content"])
                
                elif msg["type"] == "tool_call":
                    tool_id = msg["id"]
                    tool_state = tool_states.get(tool_id, {})
                    done = tool_state.get("done", False)
                    sub_steps = tool_state.get("sub_steps", [])
                    result = tool_state.get("result")
                    
                    icon = "✅" if done else "⏳"
                    question = msg["content"]
                    short_q = question[:80] + "..." if len(question) > 80 else question
                    
                    with st.expander(f"{icon} 🔍 {short_q}", expanded=not done):
                        if sub_steps:
                            for step in sub_steps:
                                step_icon = "✅" if step.get("done") else "🔄"
                                st.markdown(f"{step_icon} {step['name']}")
                                if step.get("result"):
                                    st.caption(step["result"][:300] + "..." if len(step.get("result", "")) > 300 else step.get("result", ""))
                        elif not done:
                            st.info("Researching...")
                        
                        if done and result:
                            st.divider()
                            st.markdown("**Result:**")
                            st.markdown(result)
    
    # Stream events using astream_events to capture nested runnable events
    async for event in app.astream_events(initial_state, version="v2"):
        event_type = event["event"]
        event_name = event.get("name", "")
        metadata = event.get("metadata", {})
        langgraph_node = metadata.get("langgraph_node", "")
        
        # Track when orchestrator LLM produces output
        if event_type == "on_chat_model_end":
            output = event.get("data", {}).get("output")
            
            # Capture AI messages from the orchestrator's llm_call node
            if langgraph_node == "llm_call" and output:
                # Extract and display the AI's reasoning/text
                if hasattr(output, "content") and output.content:
                    ai_text = extract_ai_text(output.content)
                    if ai_text.strip():
                        message_stream.append({"type": "ai", "content": ai_text})
                        render_messages()
                
                # Track tool calls - add them to the stream
                if hasattr(output, "tool_calls") and output.tool_calls:
                    researcher_calls = [tc for tc in output.tool_calls if tc["name"] == "call_researcher"]
                    if researcher_calls:
                        status_placeholder.info(f"🔍 Researching {len(researcher_calls)} queries...")
                        for tc in researcher_calls:
                            question = tc["args"].get("question", "Planning...")
                            tool_states[tc["id"]] = {"sub_steps": [], "result": None, "done": False}
                            message_stream.append({"type": "tool_call", "id": tc["id"], "content": question})
                        render_messages()
            
            # Check if this is the planner's output
            if langgraph_node == "planner":
                if output and hasattr(output, "content") and output.content:
                    final_plan = output.content
                    status_placeholder.success("✅ Plan ready!")
        
        # Handle tool start events
        elif event_type == "on_tool_start":
            # Track when call_researcher tool starts
            if event_name == "call_researcher":
                run_id = event.get("run_id", "")
                # Find the tool_id for this run
                for tool_id, state in tool_states.items():
                    if not state.get("done") and not state.get("_run_id"):
                        state["_run_id"] = run_id
                        current_tool_id = tool_id
                        break
            
            # Track Tavily search starts (sub-agent tool calls)
            elif "tavily" in event_name.lower():
                if current_tool_id and current_tool_id in tool_states:
                    query = event.get("data", {}).get("input", {})
                    query_str = query.get("query", str(query)) if isinstance(query, dict) else str(query)
                    tool_states[current_tool_id]["sub_steps"].append({
                        "name": f"🔍 {query_str[:70]}",
                        "run_id": event.get("run_id"),
                        "done": False
                    })
                    render_messages()
        
        # Handle tool end events
        elif event_type == "on_tool_end":
            # Track Tavily search completions
            if "tavily" in event_name.lower():
                run_id = event.get("run_id")
                if current_tool_id and current_tool_id in tool_states:
                    for step in tool_states[current_tool_id]["sub_steps"]:
                        if step.get("run_id") == run_id:
                            step["done"] = True
                            output = event.get("data", {}).get("output", "")
                            if output:
                                step["result"] = str(output)[:300]
                            break
                    render_messages()
            
            # Track when call_researcher tool completes
            elif event_name == "call_researcher":
                output = event.get("data", {}).get("output", "")
                if current_tool_id and current_tool_id in tool_states:
                    content = str(output)
                    if hasattr(output, 'content'):
                        content = output.content
                    elif content.startswith("content='"):
                        content = extract_content(content)
                    tool_states[current_tool_id]["result"] = content
                    tool_states[current_tool_id]["done"] = True
                    current_tool_id = None
                    
                    render_messages()
                    done_count = sum(1 for s in tool_states.values() if s.get("done"))
                    total = len(tool_states)
                    status_placeholder.info(f"📊 Research: {done_count}/{total} complete")
            
            # Track when call_planner tool completes (routing to planner)
            elif event_name == "call_planner":
                status_placeholder.info("✨ Creating your personalized travel plan...")
                with planner_status_placeholder.container():
                    st.markdown("### ✍️ Writing Your Itinerary")
                    st.caption("Combining research into a comprehensive travel plan...")
        
        # Handle chat model streaming for live plan preview
        elif event_type == "on_chat_model_stream":
            if langgraph_node == "planner":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    if not plan_streaming:
                        plan_streaming = True
                        plan_chunks = []
                    plan_chunks.append(chunk.content)
                    # Update preview with streamed content
                    with plan_preview_placeholder.container():
                        st.markdown("".join(plan_chunks) + "▌")
        
        # Handle chat model start for planner
        elif event_type == "on_chat_model_start":
            if langgraph_node == "planner":
                status_placeholder.info("✨ Creating your personalized travel plan...")
                with planner_status_placeholder.container():
                    st.markdown("### ✍️ Writing Your Itinerary")
                    with st.spinner("Crafting the perfect travel experience..."):
                        pass
    
    # Clear the streaming preview once done
    if plan_streaming:
        planner_status_placeholder.empty()
        plan_preview_placeholder.empty()
    
    # Show final plan
    if final_plan:
        with plan_container:
            st.divider()
            st.subheader("Your Travel Plan")
            st.markdown(final_plan)
            st.download_button(
                "Download as Markdown",
                final_plan,
                file_name=f"trip_{destination_city.lower().replace(' ', '_')}.md"
            )


def run_planner(home_city: str, destination_city: str, start_date: str, 
                end_date: str, budget: float, additional_info: str):
    """Wrapper to run async planner in Streamlit."""
    asyncio.run(run_planner_async(
        home_city, destination_city, start_date, end_date, budget, additional_info
    ))


# Main logic
if plan_button:
    if not home_city or not destination_city:
        st.error("Please enter both departure and destination cities.")
    elif start_date >= end_date:
        st.error("End date must be after start date.")
    else:
        st.caption(f"{home_city} → {destination_city} · {(end_date - start_date).days} days · ${budget:,}")
        run_planner(
            home_city=home_city,
            destination_city=destination_city,
            start_date=str(start_date),
            end_date=str(end_date),
            budget=float(budget),
            additional_info=additional_info or ""
        )
else:
    st.info("Fill in your trip details in the sidebar and click **Plan Trip** to get started.")
