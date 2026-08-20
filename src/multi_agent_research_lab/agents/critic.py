"""Critic agent for fact-checking and quality review."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient


class CriticAgent(BaseAgent):
    """Fact-checking, citation verification, and quality audit agent."""

    name = "critic"

    def __init__(self) -> None:
        self.llm_client = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Validate final answer and record critique results."""
        if not state.final_answer:
            state.errors.append("Critic found no final_answer to review.")
            return state

        sources_summary = "\n".join(
            [f"- [{doc.title}]({doc.url}): {doc.snippet[:150]}" for doc in state.sources]
        ) if state.sources else "No explicit sources."

        response = self.llm_client.complete(
            system_prompt=(
                "You are an expert quality and fact-checking critic for AI research reports. "
                "Audit the provided report for: 1) Citation validity, 2) Groundedness, "
                "3) Structural clarity. Provide a brief audit summary and pass/fail score."
            ),
            user_prompt=(
                f"--- Report to Audit ---\n{state.final_answer}\n\n"
                f"--- Reference Sources ---\n{sources_summary}"
            ),
        )

        critique_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.CRITIC,
                content=critique_notes,
                metadata={"audit_completed": True},
            )
        )
        state.add_trace_event(
            "critic_complete",
            {"critique_length": len(critique_notes)},
        )
        return state

