#!/usr/bin/env python3
"""Brave Search CLI - search the web via Brave Search API."""

import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

API_URL = "https://api.search.brave.com/res/v1/web/search"
TIMEOUT = 15


def main():
    api_key = os.environ.get("BRAVE_API_KEY")
    if not api_key:
        print("Error: BRAVE_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("Usage: search.py QUERY [COUNT]", file=sys.stderr)
        sys.exit(1)

    query = sys.argv[1].strip()
    count = 5
    if len(sys.argv) >= 3:
        try:
            count = max(1, min(int(sys.argv[2]), 20))
        except ValueError:
            pass

    params = urlencode({"q": query, "count": count})
    url = f"{API_URL}?{params}"

    req = Request(url)
    req.add_header("Accept", "application/json")
    req.add_header("Accept-Encoding", "identity")
    req.add_header("X-Subscription-Token", api_key)

    try:
        with urlopen(req, timeout=TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        print(f"Brave API error {e.code}: {body}", file=sys.stderr)
        sys.exit(1)
    except URLError as e:
        print(f"Connection error: {e.reason}", file=sys.stderr)
        sys.exit(1)

    results = data.get("web", {}).get("results", [])
    if not results:
        print(f"No results for: {query}")
        return

    print(f"Brave Search results for: {query}\n")
    for i, r in enumerate(results[:count], 1):
        title = r.get("title", "")
        url_ = r.get("url", "")
        snippet = r.get("description", "")
        print(f"{i}. **{title}**")
        print(f"   URL: {url_}")
        if snippet:
            print(f"   {snippet}")
        print()


if __name__ == "__main__":
    main()
