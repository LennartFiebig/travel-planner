from dotenv import load_dotenv

load_dotenv()

from langfuse.langchain import CallbackHandler
from langchain_core.messages import HumanMessage
from travel_planner.agents.orchestrator import orchestrator_builder

def main():
    langfuse_handler = CallbackHandler()
    app = orchestrator_builder.compile().with_config(config={"callbacks": [langfuse_handler]})
    messages = app.invoke({
        "messages": [HumanMessage(content="Plan my trip")],
        "home_city": "Munich",
        "destination_city": "Tokyo",
        "start_date": "2026-02-01",
        "end_date": "2026-02-03",
        "budget": 2000.0,
        "additional_info": "I love Japanese food"
    })
    for m in messages["messages"]:
        m.pretty_print()

if __name__ == "__main__":
    main()