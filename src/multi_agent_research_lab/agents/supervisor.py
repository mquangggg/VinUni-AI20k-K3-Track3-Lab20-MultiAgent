"""Supervisor / router skeleton."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.state import ResearchState


from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def run(self, state: ResearchState) -> ResearchState:
        """Update `state.route_history` with the next route."""
        settings = get_settings()

        # Guardrail: Check max iterations
        if state.iteration >= settings.max_iterations:
            next_route = "done"
        elif state.research_notes is None:
            next_route = "researcher"
        elif state.analysis_notes is None:
            next_route = "analyst"
        elif state.final_answer is None:
            next_route = "writer"
        else:
            next_route = "done"

        state.record_route(next_route)
        state.add_trace_event(
            "supervisor_routing",
            {"iteration": state.iteration, "decision": next_route},
        )
        return state

