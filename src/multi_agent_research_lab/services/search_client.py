import json
import os
from pathlib import Path
import urllib.error
import urllib.request

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client implementation with offline corpus support."""

    def _find_corpus_dir(self) -> Path | None:
        """Locate the ai_agent_offline_research_corpus_v2 directory."""
        candidates = [
            Path.cwd() / "ai_agent_offline_research_corpus_v2",
            Path.cwd().parent / "ai_agent_offline_research_corpus_v2",
            Path("c:/Users/Nitro Tiger/OneDrive/Dokumen/VInUni_AI/Lab/Lab_20/ai_agent_offline_research_corpus_v2"),
        ]
        for candidate in candidates:
            if candidate.exists() and (candidate / "topics").exists():
                return candidate
        return None

    def search_offline_corpus(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search through offline research corpus v2 topics."""
        corpus_dir = self._find_corpus_dir()
        if not corpus_dir:
            return []

        topics_dir = corpus_dir / "topics"
        query_words = set(query.lower().split())

        scored_results: list[tuple[float, SourceDocument]] = []

        for json_file in topics_dir.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                kb = data.get("knowledge_base", {})
                topic_info = data.get("topic", {})
                topic_name = topic_info.get("name", "")

                # Score articles
                for article in kb.get("knowledge_articles", []):
                    art_id = article.get("article_id", "A0")
                    title = article.get("title", "")
                    content = article.get("content", "")
                    text = f"{title} {content}".lower()
                    overlap = sum(1 for w in query_words if w in text)
                    if overlap > 0:
                        scored_results.append((
                            overlap + 2.0,
                            SourceDocument(
                                title=f"[{art_id}] {title} ({topic_name})",
                                url=f"corpus://articles/{art_id}",
                                snippet=content[:400] + ("..." if len(content) > 400 else ""),
                                metadata={"article_id": art_id, "topic": topic_name, "type": "knowledge_article"},
                            )
                        ))

                # Score source documents
                for doc in kb.get("source_documents", []):
                    doc_id = doc.get("source_id") or doc.get("document_id") or "S0"
                    title = doc.get("title", "")
                    snippet = doc.get("snippet") or doc.get("summary") or doc.get("content") or ""
                    text = f"{title} {snippet}".lower()
                    overlap = sum(1 for w in query_words if w in text)
                    if overlap > 0:
                        scored_results.append((
                            overlap + 1.5,
                            SourceDocument(
                                title=f"[{doc_id}] {title}",
                                url=doc.get("url") or f"corpus://sources/{doc_id}",
                                snippet=snippet[:400] + ("..." if len(snippet) > 400 else ""),
                                metadata={"source_id": doc_id, "is_synthetic": doc.get("is_synthetic", False)},
                            )
                        ))

                # Score atomic facts
                for fact in kb.get("fact_bank", []):
                    fact_id = fact.get("fact_id", "F0")
                    statement = fact.get("fact_statement") or fact.get("statement") or ""
                    text = statement.lower()
                    overlap = sum(1 for w in query_words if w in text)
                    if overlap > 0:
                        scored_results.append((
                            overlap + 1.0,
                            SourceDocument(
                                title=f"[{fact_id}] Atomic Fact",
                                url=f"corpus://facts/{fact_id}",
                                snippet=statement,
                                metadata={"fact_id": fact_id, "evidence_id": fact.get("evidence_id")},
                            )
                        ))
            except Exception:
                continue

        # Sort by score descending
        scored_results.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored_results[:max_results]]

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        Tries offline corpus first, then Tavily Search API, with fallback to mock search results.
        """
        # 1. Try offline corpus search first
        corpus_docs = self.search_offline_corpus(query, max_results=max_results)
        if corpus_docs:
            return corpus_docs

        # 2. Try Tavily Search API if key provided
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
                pass

        # 3. Mock fallback search results
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


