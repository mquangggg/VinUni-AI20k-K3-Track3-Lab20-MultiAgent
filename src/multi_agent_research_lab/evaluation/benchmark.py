"""Benchmark runner for single-agent vs multi-agent."""

from collections.abc import Callable
from time import perf_counter

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(
    run_name: str, query: str, runner: Runner
) -> tuple[ResearchState, BenchmarkMetrics]:
    """Measure latency, cost, citation coverage, and quality score."""

    started = perf_counter()
    error_occurred = False
    try:
        state = runner(query)
    except Exception as exc:
        latency = perf_counter() - started
        state = ResearchState(request=ResearchQuery(query=query), errors=[str(exc)])
        metrics = BenchmarkMetrics(
            run_name=run_name,
            latency_seconds=latency,
            failure_rate=1.0,
            notes=f"Failed with error: {exc}",
        )
        return state, metrics

    latency = perf_counter() - started

    # Calculate metrics
    sources_count = len(state.sources)
    has_final_answer = bool(state.final_answer)
    answer_len = len(state.final_answer) if state.final_answer else 0

    # Citation coverage heuristic
    citation_coverage = min(1.0, sources_count / 3.0) if sources_count > 0 else 0.0

    # Quality score heuristic (scale 0-10)
    quality_score = 0.0
    if has_final_answer:
        quality_score += 5.0
        if answer_len > 300:
            quality_score += 2.0
        if sources_count > 0:
            quality_score += 2.0
        if state.analysis_notes:
            quality_score += 1.0

    # Estimated cost (rough approximation per run)
    cost_usd = 0.0005 if "multi" in run_name.lower() else 0.00015

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=latency,
        estimated_cost_usd=cost_usd,
        quality_score=quality_score,
        citation_coverage=citation_coverage,
        failure_rate=0.0,
        notes="Execution completed successfully",
    )
    return state, metrics

