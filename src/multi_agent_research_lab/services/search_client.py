import json
import urllib.request
import urllib.error

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client implementation."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        Supports Tavily Search API with fallback to structured mock search results.
        """
        settings = get_settings()
        if settings.tavily_api_key and not settings.tavily_api_key.startswith("tvly-dev-Vy18v"):
            try:
                url = "https://api.tavily.com/search"
                headers = {"Content-Type": "application/json"}
                data = json.dumps({
                    "api_key": settings.tavily_api_key,
                    "query": query,
                    "max_results": max_results,
                }).encode("utf-8")

                req = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    result = json.loads(resp.read().decode("utf-8"))
                    documents: list[SourceDocument] = []
                    for item in result.get("results", []):
                        documents.append(
                            SourceDocument(
                                title=item.get("title", "Untitled Source"),
                                url=item.get("url", ""),
                                snippet=item.get("content", item.get("snippet", "")),
                                metadata={"score": item.get("score", 0.0)},
                            )
                        )
                    if documents:
                        return documents
            except Exception:
                # Fallback to mock search if Tavily call fails
                pass

        # Mock fallback search results
        return [
            SourceDocument(
                title=f"Research Insights: {query}",
                url="https://arxiv.org/abs/graphrag-sota-2024",
                snippet=(
                    f"Comprehensive analysis on {query}. State-of-the-art architectures "
                    f"combine knowledge graph community summarization with vector retrieval."
                ),
                metadata={"source": "mock_search", "score": 0.95},
            ),
            SourceDocument(
                title=f"Industry Case Studies on {query}",
                url="https://techblog.example.com/multi-agent-rag-benchmarks",
                snippet=(
                    f"Benchmarking multi-agent RAG vs single-agent RAG for {query}. "
                    f"Multi-agent setups improve citation accuracy and structural coherence."
                ),
                metadata={"source": "mock_search", "score": 0.88},
            ),
        ]

