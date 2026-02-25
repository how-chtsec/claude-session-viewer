# Claude Session Viewer

A web-based dashboard for browsing and analyzing [Claude Code](https://docs.anthropic.com/en/docs/claude-code) session logs. View conversations, tool calls, token usage, costs, team activity, and monitor active sessions in real time.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-3.0+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

### Session Browser
- Full conversation timeline with user messages, assistant responses, and extended thinking
- Tool call details with input/output and execution duration
- Subagent activity with dedicated detail pages
- Code syntax highlighting via Highlight.js

### Token Usage & Cost Analytics
- Global and per-project token usage breakdowns (input, output, cache read, cache creation)
- Daily token usage and cost trends
- Per-model cost calculation aligned with [ccusage](https://github.com/ryoppippi/ccusage)
- Dynamic pricing fetched from [LiteLLM](https://github.com/BerriAI/litellm) with local caching and offline fallback

### Live Monitoring
- Real-time session streaming via Server-Sent Events (SSE)
- Active session detection (sessions modified within the last 2 minutes)
- Multi-pane monitor dashboard for watching multiple sessions simultaneously
- Embeddable view for integration into other tools

### Team & Task Tracking
- Browse teams with member lists and task summaries
- Task dependency graph visualization
- Per-member activity timelines and task assignments
- Agent-to-team-member name mapping

### Project Organization
- Sessions grouped by date with smart labels (Today, Yesterday, weekday names)
- Per-project usage comparison
- Subagent count badges on session listings

## Quick Start

### Prerequisites

- Python 3.10+
- Claude Code session logs at `~/.claude/projects/`

### Installation

```bash
git clone https://github.com/how-chtsec/claude-session-viewer.git
cd claude-session-viewer
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

## Project Structure

```
claude-session-viewer/
├── app.py                  # Flask routes and template filters
├── requirements.txt
├── services/
│   ├── data.py             # Data aggregation and statistics
│   ├── parser.py           # JSONL session log parser
│   └── pricing.py          # Dynamic pricing from LiteLLM
└── templates/
    ├── base.html            # Base template (Tailwind CSS dark theme)
    ├── index.html           # Dashboard
    ├── project.html         # Project session list
    ├── session.html         # Session detail timeline
    ├── subagent_detail.html # Subagent detail page
    ├── usage.html           # Global usage stats
    ├── usage_projects.html  # Cross-project comparison
    ├── live.html            # Active sessions list
    ├── live_session.html    # Real-time session view
    ├── live_embed.html      # Embeddable live view
    ├── monitor.html         # Multi-pane monitor
    ├── teams.html           # Team listing
    ├── team_detail.html     # Team detail with task graph
    └── member_detail.html   # Team member detail
```

## How It Works

Claude Code stores session logs as JSONL files under `~/.claude/projects/`. This viewer parses those files and presents them through a Flask web interface.

**Data flow:**

1. **Parser** (`services/parser.py`) reads JSONL files, extracts messages, tool calls, subagents, and token usage
2. **Data layer** (`services/data.py`) aggregates sessions, computes statistics, resolves team/agent relationships
3. **Pricing** (`services/pricing.py`) fetches model pricing from LiteLLM's database, caches locally for 24 hours, falls back to hardcoded rates when offline
4. **Routes** (`app.py`) serve HTML pages and JSON APIs
5. **Templates** render with Tailwind CSS in a dark theme optimized for readability

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/session/<project>/<session_id>` | Session data as JSON |
| `GET /api/session/<project>/<session_id>/stream` | SSE stream for live updates |
| `GET /api/session/<project>/<session_id>/updates?from_line=N` | Poll for new entries |
| `GET /api/active` | List currently active sessions |
| `GET /api/teams` | List all teams |
| `GET /api/team/<name>` | Team detail |
| `GET /api/team/<name>/member/<member>` | Member detail |

## Cost Calculation

Token costs are calculated per JSONL entry using model-specific rates. The algorithm deduplicates streaming entries by `messageId:requestId` and groups daily stats by each entry's local-timezone timestamp, matching [ccusage](https://github.com/ryoppippi/ccusage)'s methodology.

Pricing is fetched dynamically from [LiteLLM's model pricing database](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json) on startup. A local cache (`.cache/litellm_pricing.json`) is maintained for offline use, with hardcoded fallback rates as a last resort.

## Security Notice

This application reads and displays the contents of your local Claude Code session logs (`~/.claude/`). Please be aware of the following:

- **Session logs may contain sensitive data** — file contents, environment variables, API keys, passwords, or other secrets that appeared in your Claude Code conversations could be exposed through this viewer.
- **Do not expose this application to the public internet.** It is designed for local use only. Running it on `0.0.0.0` makes it accessible to other devices on your network.
- **No authentication is built in.** Anyone who can reach the server can browse all session data.
- **Review your logs before sharing screenshots or recordings** of this viewer, as they may inadvertently reveal sensitive information.

## Disclaimer

This project was **entirely written by Claude** (Anthropic's AI assistant) through Claude Code. All code — including the Flask application, JSONL parser, pricing engine, and HTML templates — was generated by AI with human direction and review.

This is an unofficial community tool and is **not affiliated with, endorsed by, or supported by Anthropic**. It relies on undocumented Claude Code session log formats that may change without notice. Use at your own risk.

## License

MIT
