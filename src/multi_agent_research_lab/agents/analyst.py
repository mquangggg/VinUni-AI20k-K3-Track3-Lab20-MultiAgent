from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self.llm_client = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes`."""
        research_notes = state.research_notes or "No research notes available."

        response = self.llm_client.complete(
            system_prompt=(
                "You are an expert analytical agent. Analyze the provided research notes, "
                "extract core themes, evaluate trade-offs/limitations, and identify key takeaways."
            ),
            user_prompt=(
                f"User Request: {state.request.query}\n\n"
                f"Target Audience: {state.request.audience}\n\n"
                f"Research Notes:\n{research_notes}"
            ),
        )

        state.analysis_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.ANALYST,
                content=response.content,
                metadata={"notes_analyzed": True},
            )
        )
        state.add_trace_event(
            "analyst_complete",
            {"analysis_length": len(response.content)},
        )
        return state

