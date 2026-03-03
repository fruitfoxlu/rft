#!/usr/bin/env python3
"""MCP stdio server providing brave_search and perplexity_ask tools.

Runs as a subprocess managed by Claude Code. Subagents get these
as first-class callable tools.

Environment variables:
    BRAVE_API_KEY:      Brave Search API subscription token
    PERPLEXITY_API_KEY: Perplexity AI API key
    PERPLEXITY_MODEL:   Model name (default: sonar-pro)
    PERPLEXITY_TIMEOUT: Request timeout in seconds (default: 180)
    PERPLEXITY_SEARCH_CONTEXT_SIZE: low|medium|high (default: high)
    PERPLEXITY_SEARCH_MODE: web|academic|sec (default: web)
"""

import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("search")

# ---------------------------------------------------------------------------
# Brave Search
# ---------------------------------------------------------------------------

@mcp.tool()
def brave_search(query: str, count: int = 5) -> str:
    """Search the web using Brave Search API. Returns titles, snippets, and URLs.

    Args:
        query: Search query string.
        count: Number of results (1-20, default 5).
    """
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        return "Error: BRAVE_API_KEY not set."

    query = query.strip()
    if not query:
        return "Error: query is required."

    count = max(1, min(count, 20))
    params = urlencode({"q": query, "count": count})
    url = f"https://api.search.brave.com/res/v1/web/search?{params}"

    req = Request(url)
    req.add_header("Accept", "application/json")
    req.add_header("Accept-Encoding", "identity")
    req.add_header("X-Subscription-Token", api_key)

    try:
        with urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        return f"Brave API error {e.code}: {body}"
    except URLError as e:
        return f"Connection error: {e.reason}"

    results = data.get("web", {}).get("results", [])
    if not results:
        return f"No results for: {query}"

    lines = [f"Brave Search results for: {query}\n"]
    for i, r in enumerate(results[:count], 1):
        title = r.get("title", "")
        url_ = r.get("url", "")
        snippet = r.get("description", "")
        lines.append(f"{i}. **{title}**")
        lines.append(f"   URL: {url_}")
        if snippet:
            lines.append(f"   {snippet}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Perplexity Ask
# ---------------------------------------------------------------------------

@mcp.tool()
def perplexity_ask(
    question: str,
    search_mode: str = "",
    search_domain_filter: str = "",
    search_recency_filter: str = "",
) -> str:
    """Ask Perplexity AI and get a synthesized answer with citations.

    Args:
        question: The question to ask.
        search_mode: web, academic, or sec (default: web).
        search_domain_filter: Comma-separated domains to restrict search.
        search_recency_filter: day, week, month, or year.
    """
    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        return "Error: PERPLEXITY_API_KEY not set."

    question = question.strip()
    if not question:
        return "Error: question is required."

    model = os.environ.get("PERPLEXITY_MODEL", "sonar-pro")
    try:
        timeout = int(os.environ.get("PERPLEXITY_TIMEOUT", "180"))
    except ValueError:
        timeout = 180

    raw_ctx = os.environ.get("PERPLEXITY_SEARCH_CONTEXT_SIZE", "high").strip().lower()
    ctx_size = raw_ctx if raw_ctx in {"low", "medium", "high"} else "high"
    default_mode = os.environ.get("PERPLEXITY_SEARCH_MODE", "web").strip().lower()

    payload = {
        "model": model,
        "safe_search": False,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a cybersecurity research assistant. "
                    "Provide precise, technical answers with specific "
                    "tool commands, code snippets, and references. "
                    "Focus on CTF techniques and security tools."
                ),
            },
            {"role": "user", "content": question},
        ],
        "web_search_options": {
            "search_context_size": ctx_size,
        },
    }

    mode = (search_mode.strip().lower() or default_mode)
    if mode in {"web", "academic", "sec"}:
        payload["search_mode"] = mode

    if search_domain_filter:
        domains = [d.strip() for d in search_domain_filter.split(",") if d.strip()]
        if domains:
            payload["search_domain_filter"] = domains[:20]

    recency = search_recency_filter.strip().lower()
    if recency in {"day", "week", "month", "year"}:
        payload["search_recency_filter"] = recency

    req = Request(
        "https://api.perplexity.ai/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
    )
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")

    try:
        with urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        return f"Perplexity API error {e.code}: {body}"
    except URLError as e:
        return f"Connection error: {e.reason}"

    choices = data.get("choices", [])
    if not choices:
        return "Perplexity returned no response."

    answer = choices[0].get("message", {}).get("content", "")
    if not answer:
        return "Perplexity returned empty answer."

    citations = data.get("citations", [])
    lines = [f"Perplexity answer for: {question}\n", answer]
    if citations:
        lines.append("\n\n**Sources:**")
        for i, cite in enumerate(citations, 1):
            if isinstance(cite, str):
                lines.append(f"{i}. {cite}")
            elif isinstance(cite, dict):
                lines.append(f"{i}. {cite.get('url', cite)}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
