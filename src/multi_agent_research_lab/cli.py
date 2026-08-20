import time
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


def _parse_query(query: str) -> ResearchQuery:
    try:
        return ResearchQuery(query=query)
    except ValidationError as exc:
        console.print(
            Panel.fit(
                f"Invalid query: {exc.errors()[0]['msg']}",
                title="Input Error",
                style="red",
            )
        )
        raise typer.Exit(code=1) from exc


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a single-agent baseline using LLMClient."""

    _init()
    request = _parse_query(query)
    state = ResearchState(request=request)

    llm = LLMClient()
    start_time = time.time()
    
    response = llm.complete(
        system_prompt=(
            "You are a helpful research assistant. Research the user query thoroughly "
            "and provide a concise, high-quality summary."
        ),
        user_prompt=request.query,
    )
    latency = time.time() - start_time
    
    state.final_answer = response.content
    metrics_info = (
        f"\n\n--- Baseline Metrics ---\n"
        f"Latency: {latency:.2f}s | "
        f"Tokens (in/out): {response.input_tokens}/{response.output_tokens} | "
        f"Est. Cost: ${response.cost_usd:.6f}" if response.cost_usd else f"Latency: {latency:.2f}s"
    )
    panel_content = (state.final_answer or "") + "\n" + metrics_info
    console.print(Panel.fit(panel_content, title="Single-Agent Baseline Result"))




@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow skeleton."""

    _init()
    state = ResearchState(request=_parse_query(query))
    workflow = MultiAgentWorkflow()
    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc
    console.print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
