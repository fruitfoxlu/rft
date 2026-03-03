---
name: perplexity-ask
description: Ask Perplexity AI a question and get a synthesized answer with citations.
allowed-tools: Bash(python3 *)
---

Ask Perplexity AI a question and get a synthesized answer with citations.

Run the ask script with the user's question:

```bash
python3 .claude/skills/perplexity-ask/scripts/ask.py "QUESTION" [OPTIONS]
```

Options:
- `--mode web|academic|sec` (default: web, or PERPLEXITY_SEARCH_MODE env)
- `--domains example.com,other.com` (comma-separated domain filter)
- `--recency day|week|month|year` (time filter)

Requires PERPLEXITY_API_KEY environment variable.
