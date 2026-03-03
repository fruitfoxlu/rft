---
name: brave-search
description: Search the web using Brave Search API. Returns titles, snippets, and URLs.
allowed-tools: Bash(python3 *)
---

Search the web using Brave Search API. Use this for broad discovery of techniques, tools, CVEs, and documentation.

Run the search script with the user's query:

```bash
python3 .claude/skills/brave-search/scripts/search.py "QUERY" [COUNT]
```

- Replace QUERY with the actual search query
- COUNT is optional (1-20, default 5)
- Requires BRAVE_API_KEY environment variable

After getting results, you can use WebFetch/Read to drill into specific pages.
