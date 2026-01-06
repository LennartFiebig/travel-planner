from typing import Optional
from langgraph.graph import MessagesState

class OrchestratorState(MessagesState):
    home_city: Optional[str] = None
    destination_city: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    budget: Optional[float] = None
    additional_info: Optional[str] = None
    
    is_finished: Optional[bool] = None