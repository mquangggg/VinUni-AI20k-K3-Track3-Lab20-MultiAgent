from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self.llm_client = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer`."""
        research_notes = state.research_notes or "N/A"
        analysis_notes = state.analysis_notes or "N/A"

        sources_text = "\n".join(
            [f"- [{doc.title}]({doc.url})" for doc in state.sources]
        ) if state.sources else "No explicit URL sources."

        response = self.llm_client.complete(
            system_prompt=(
                "You are an expert technical writer. Synthesize research notes, analysis notes, "
                "and source links into a polished, comprehensive research report tailored to the target audience. "
                "Include a References/Citations section at the end."
            ),
            user_prompt=(
                f"Topic Query: {state.request.query}\n"
                f"Audience: {state.request.audience}\n\n"
                f"--- Research Notes ---\n{research_notes}\n\n"
                f"--- Analysis Notes ---\n{analysis_notes}\n\n"
                f"--- Sources ---\n{sources_text}"
            ),
        )

        state.final_answer = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.WRITER,
                content=response.content,
                metadata={"final_answer_length": len(response.content)},
            )
        )
        state.add_trace_event(
            "writer_complete",
            {"final_answer_length": len(response.content)},
        )
        return state

