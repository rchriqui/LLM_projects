# Summer Camp Registration Assistant


## Overview

Build a conversational AI agent that helps parents register their children for summer camps. The agent should handle natural language requests, interact with a mock database through tools, manage multi-turn conversations, and handle edge cases gracefully.

You are provided with a mock database (`mock_db.json`) and minimal scaffolding. Your job is to design and implement a working agent that delivers a great conversational experience.

---

## Setup

```bash
# 1. Add your OpenAI API key to .env
# 2. Install dependencies
uv sync
# 3. Run the agent (launches debug UI in browser)
uv run python agent.py
```

**Available model:** `gpt-4o-mini`

**Important:** The `.env` file is gitignored. Do **not** commit or push your API key to GitHub.

---

## Tasks

### Task 1: Implement Tools (Data Layer)

Implement tool functions that let the agent interact with `mock_db.json`. Start from the stubs in `tool_schemas.py` — you may restructure, rename, add parameters, or add helper functions as you see fit.

**Required capabilities:**

| Operation | Description |
|-----------|-------------|
| Read camps | Return camp details (availability, schedule, age range, price, status) |
| Read kids | Return children info (name, age, parent contact) |
| Read registrations | Return registration records with status |
| Register a child | Create a new registration with proper validation |
| Cancel a registration | Cancel an existing registration and update enrollment |
| Update registration status | Change status (e.g., pending → confirmed) |

Your tools are the foundation of the agent's conversational experience. Beyond basic operations, they should ensure data integrity — for example, a child shouldn't be registered for a camp outside their age range, or one that's already full. Look at the data carefully and consider what else could go wrong.

### Task 2: Build the Agent (Conversation Layer)

Implement the conversational agent in `agent.py`. The `CampAssistant` class must support the `chat()` interface:

```python
class CampAssistant:
    def chat(self, user_message: str) -> str:
        ...
```

**Requirements:**
- The agent should use the tools from Task 1 to fulfill user requests
- **System prompt design** — craft a system prompt that guides the agent's behavior. Think carefully about what belongs in the system prompt vs. tool descriptions vs. neither
- **Multi-turn conversation** — the agent must maintain context across turns. A user might say "Register my kid for a camp," then provide the name in the next message, then confirm in a third
- **Confirmation before writes** — the agent must confirm with the user before creating, cancelling, or modifying registrations
- **Graceful handling** — when something goes wrong (invalid request, ambiguous input, missing info), the agent should explain the issue clearly and suggest next steps

The data in `mock_db.json` is intentionally realistic — not every request will be straightforward. Make sure your agent handles the messy cases well.

Use the Gradio debug UI (`uv run python agent.py`) to interactively test your agent during development.

---

## Bonus (Optional)

- **Waitlist system** — when a camp is full, offer to add the child to a waitlist. If a cancellation opens a spot, the agent should be aware of waitlisted children. You may modify the DB schema.

Have another interesting feature in mind? Go for it — we'd love to see what you come up with.

---

## Files

| File | Purpose |
|------|---------|
| `mock_db.json` | The database — camps, kids, registrations |
| `tool_schemas.py` | Tool function stubs (starting point) |
| `agent.py` | Agent implementation + Gradio debug UI |
| `pyproject.toml` | Project dependencies |
| `.env` | OpenAI API key (provided) |

---

## Submission

Submit your solution as a **git repository** (zip or link). Please do not push your API key to git!

We should be able to run your agent with:
```bash
uv sync
uv run python agent.py
```

Good Luck!