"""Unit test for SupervisorAgent routing policy."""

from multi_agent_research_lab.agents import SupervisorAgent
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState


def test_supervisor_routing_policy() -> None:
    state = ResearchState(request=ResearchQuery(query="Explain multi-agent systems"))
    
    # 1st step: Should route to researcher
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == "researcher"

    # 2nd step: After research_notes added, should route to analyst
    state.research_notes = "Dummy research notes"
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == "analyst"

    # 3rd step: After analysis_notes added, should route to writer
    state.analysis_notes = "Dummy analysis notes"
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == "writer"

    # 4th step: After final_answer added, should route to done
    state.final_answer = "Dummy final answer"
    state = SupervisorAgent().run(state)
    assert state.route_history[-1] == "done"

