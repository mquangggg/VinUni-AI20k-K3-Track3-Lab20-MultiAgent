from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.schemas import AgentName, AgentResult
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self.search_client = SearchClient()
        self.llm_client = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        sources = self.search_client.search(state.request.query, max_results=state.request.max_sources)
        state.sources.extend(sources)

        sources_text = "\n".join(
            [f"- [{doc.title}]({doc.url}): {doc.snippet}" for doc in sources]
        )

        response = self.llm_client.complete(
            system_prompt=(
                "You are an expert researcher. Synthesize the provided search sources into "
                "clear, structured research notes with key findings and factual bullet points."
            ),
            user_prompt=f"Topic: {state.request.query}\n\nSearch Sources:\n{sources_text}",
        )

        state.research_notes = response.content
        state.agent_results.append(
            AgentResult(
                agent=AgentName.RESEARCHER,
                content=response.content,
                metadata={"sources_count": len(sources)},
            )
        )
        state.add_trace_event(
            "researcher_complete",
            {"sources_found": len(sources), "notes_length": len(response.content)},
        )
        return state

