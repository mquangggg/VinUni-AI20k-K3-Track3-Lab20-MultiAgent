"""LangGraph / Multi-agent workflow runner."""

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.critic import CriticAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def __init__(self) -> None:
        self.supervisor = SupervisorAgent()
        self.researcher = ResearcherAgent()
        self.analyst = AnalystAgent()
        self.writer = WriterAgent()
        self.critic = CriticAgent()

    def build(self) -> dict[str, object]:
        """Create the workflow node dictionary."""
        return {
            "supervisor": self.supervisor,
            "researcher": self.researcher,
            "analyst": self.analyst,
            "writer": self.writer,
            "critic": self.critic,
        }

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the workflow state machine."""
        settings = get_settings()

        while state.iteration < settings.max_iterations:
            state = self.supervisor.run(state)
            next_route = state.route_history[-1] if state.route_history else "done"

            if next_route == "done":
                break
            elif next_route == "researcher":
                state = self.researcher.run(state)
            elif next_route == "analyst":
                state = self.analyst.run(state)
            elif next_route == "writer":
                state = self.writer.run(state)
                # Run quality critic after writer completes
                state = self.critic.run(state)
            else:
                break

        return state


