#!/usr/bin/env python3
"""Perplexity Ask CLI - get AI-synthesized answers with citations."""

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API_URL = "https://api.perplexity.ai/chat/completions"


def main():
    parser = argparse.ArgumentParser(description="Ask Perplexity AI")
    parser.add_argument("question", help="Question to ask")
    parser.add_argument("--mode", choices=["web", "academic", "sec"],
                        default=os.environ.get("PERPLEXITY_SEARCH_MODE", "web"))
    parser.add_argument("--domains", default="",
                        help="Comma-separated domain filter")
    parser.add_argument("--recency", choices=["day", "week", "month", "year"],
                        default=None)
    args = parser.parse_args()

    api_key = os.environ.get("PERPLEXITY_API_KEY")
    if not api_key:
        print("Error: PERPLEXITY_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    question = args.question.strip()
    if not question:
        print("Error: question is required.", file=sys.stderr)
        sys.exit(1)

    model = os.environ.get("PERPLEXITY_MODEL", "sonar-pro")
    try:
        timeout = int(os.environ.get("PERPLEXITY_TIMEOUT", "180"))
    except ValueError:
        timeout = 180

    raw_ctx = os.environ.get("PERPLEXITY_SEARCH_CONTEXT_SIZE", "high").strip().lower()
    ctx_size = raw_ctx if raw_ctx in {"low", "medium", "high"} else "high"

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

    if args.mode:
        payload["search_mode"] = args.mode

    if args.domains:
        domains = [d.strip() for d in args.domains.split(",") if d.strip()]
        if domains:
            payload["search_domain_filter"] = domains[:20]

    if args.recency:
        payload["search_recency_filter"] = args.recency

    req = Request(
        API_URL,
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
        print(f"Perplexity API error {e.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except URLError as e:
        print(f"Connection error: {e.reason}", file=sys.stderr)
        sys.exit(1)

    choices = data.get("choices", [])
    if not choices:
        print("Perplexity returned no response.")
        return

    answer = choices[0].get("message", {}).get("content", "")
    if not answer:
        print("Perplexity returned empty answer.")
        return

    print(f"Perplexity answer for: {question}\n")
    print(answer)

    citations = data.get("citations", [])
    if citations:
        print("\n\n**Sources:**")
        for i, cite in enumerate(citations, 1):
            if isinstance(cite, str):
                print(f"{i}. {cite}")
            elif isinstance(cite, dict):
                print(f"{i}. {cite.get('url', cite)}")


if __name__ == "__main__":
    main()
