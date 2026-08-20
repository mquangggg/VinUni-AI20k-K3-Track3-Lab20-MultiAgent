from pathlib import Path
import sys
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
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
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


@app.command("benchmark")
def benchmark(
    query: Annotated[
        str,
        typer.Option("--query", "-q", help="Research query for benchmark"),
    ] = "Single-Agent vs Multi-Agent Architectures for Complex Research Tasks",
) -> None:
    """Run benchmark comparing single-agent baseline and multi-agent workflow."""

    _init()
    console.print(f"[bold green]Starting Benchmark for query:[/bold green] {query}\n")

    def run_baseline_agent(q: str) -> ResearchState:
        req = _parse_query(q)
        st = ResearchState(request=req)
        llm = LLMClient()
        resp = llm.complete(
            system_prompt="You are a single-agent research assistant. Summarize key findings thoroughly.",
            user_prompt=q,
        )
        st.final_answer = resp.content
        return st

    def run_multi_agent_workflow(q: str) -> ResearchState:
        req = _parse_query(q)
        st = ResearchState(request=req)
        wf = MultiAgentWorkflow()
        return wf.run(st)

    console.print("Running Single-Agent Baseline...")
    base_state, base_metrics = run_benchmark("Single-Agent Baseline", query, run_baseline_agent)

    console.print("Running Multi-Agent System...")
    multi_state, multi_metrics = run_benchmark("Multi-Agent Workflow", query, run_multi_agent_workflow)

    all_metrics = [base_metrics, multi_metrics]
    report_md = render_markdown_report(all_metrics)

    # Save to reports/benchmark_report.md
    reports_dir = Path.cwd() / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "benchmark_report.md"
    report_path.write_text(report_md, encoding="utf-8")

    console.print(Panel.fit(report_md, title=f"Benchmark Report Saved to {report_path}"))


if __name__ == "__main__":
    app()

