"""LLM client wrapper and prompt assembly for Dazi.

This module owns:
- LLM client creation (create_llm) and lazy initialization (_get_llm)
- System prompt section constants and assembly
- Memory/skills content injection into prompts
- The prompt builder singleton
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from dazi.config import OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
from dazi._singletons import (
    memory_store,
    skill_registry,
    settings_manager,
    proactive_manager,
)
from dazi.prompt_builder import STATIC_SECTIONS, PromptSection, SystemPromptBuilder


# ─────────────────────────────────────────────────────────
# LLM CLIENT FACTORY
# ─────────────────────────────────────────────────────────


def create_llm(
    model: str | None = None,
    temperature: float = 0.0,
    streaming: bool = True,
    base_url: str | None = None,
    api_key: str | None = None,
) -> ChatOpenAI:
    """Create a ChatOpenAI instance with project defaults.

    Args:
        model: Override model name. Defaults to OPENAI_MODEL env var.
        temperature: Model temperature.
        streaming: Enable streaming responses.
        base_url: Override API base URL. Defaults to OPENAI_BASE_URL env var.
        api_key: Override API key. Defaults to OPENAI_API_KEY env var.

    Returns:
        Configured ChatOpenAI instance.
    """
    kwargs: dict = {
        "model": model or OPENAI_MODEL,
        "temperature": temperature,
        "streaming": streaming,
    }
    if api_key:
        kwargs["api_key"] = api_key
    elif OPENAI_API_KEY:
        kwargs["api_key"] = OPENAI_API_KEY
    effective_base_url = base_url or OPENAI_BASE_URL
    if effective_base_url:
        kwargs["base_url"] = effective_base_url

    return ChatOpenAI(**kwargs)


_base_llm = None


def _get_llm():
    """Lazy LLM initialization — avoid crashing at import time if no API key."""
    global _base_llm
    if _base_llm is None:
        _base_llm = create_llm(
            model=settings_manager.get_model_name(),
            api_key=settings_manager.get_api_key(),
            base_url=settings_manager.get_api_base_url(),
        )
    return _base_llm


def _get_model_name() -> str:
    """Get the current model name for token counting."""
    return settings_manager.get_model_name()


# ─────────────────────────────────────────────────────────
# PROMPT SECTION CONSTANTS
# ─────────────────────────────────────────────────────────

TASK_MANAGEMENT_SECTION = """\
## Task Management
When given a complex goal, break it into small, concrete tasks:
1. Create tasks with clear subjects and descriptions
2. Set dependencies (blockedBy) for ordering requirements
3. Work in dependency order: mark in_progress when starting, completed when done
4. Use addBlocks to indicate which tasks depend on this one
Status lifecycle: pending -> in_progress -> completed
Use status='deleted' to remove a task entirely."""

BACKGROUND_TASKS_SECTION = """\
## Background Tasks
For long-running commands (builds, tests, downloads), use run_background to execute
them non-blocking. The system will notify you when background tasks complete.
Use check_background to monitor progress at any time.
Use cancel_background to stop a running task (requires user approval).
The agent stays responsive while background tasks run — you can answer other questions."""

MCP_TOOLS_SECTION = """\
## MCP Tools
You have access to tools from MCP (Model Context Protocol) servers. These tools are
prefixed with "mcp__<server>__<tool>". Use list_mcp_servers to see connected servers
and their available tools. MCP tools are external — handle errors gracefully and
report connection issues to the user. Use /mcp to manage server connections."""

SKILLS_GUIDANCE = """\
## Skills
You have access to skills that provide specialized instructions for common tasks.
Use the `skill` tool to invoke a skill by name. Users can also invoke skills via /<skill-name> in the REPL."""

TEAM_MANAGEMENT_SECTION = """\
## Team Management
You can create and manage agent teams for collaborative work:
1. Use create_team to create a new team with a name and description
2. Use list_teams to see all existing teams and their member counts
3. Use show_team to see team details including member status
4. Use delete_team to remove a team (all members must be completed first)
Teams share a task board. When a team is active, task operations go to that team's board.
Users can also manage teams via REPL: /teams, /team create <name>, /team <name>, /team delete <name>."""

PROTOCOLS_SECTION = """\
## Team Protocols and Messaging
When working in an active team, you can communicate with teammates:
1. Use send_message to send DMs (to: "agent-name") or broadcasts (to: "*")
2. Use check_inbox to read messages from other agents
3. Use request_permission to ask the team leader for tool approval
4. Always check your inbox for new messages and instructions when on a team
5. Respond to shutdown_request messages with a shutdown_response
6. When you complete your work, the system sends an idle_notification to teammates

Protocol message types: text, shutdown_request, shutdown_response,
permission_request, permission_response, plan_approval_request,
plan_approval_response, idle_notification"""

PROACTIVE_SECTION = """\
## Autonomous Work
You are in proactive mode. You will receive <tick> prompts that keep you alive
between turns -- treat them as "you're awake, what now?" The time in each <tick>
is the user's current local time.

### Pacing
Use the sleep tool to control how long you wait between actions. Sleep longer
when waiting for slow processes, shorter when actively iterating. Each wake-up
costs an API call, but the prompt cache expires after 5 minutes of inactivity
-- balance accordingly.

**If you have nothing useful to do on a tick, you MUST call sleep.** Never respond
with only a status message like "still waiting" or "nothing to do" -- that wastes
a turn and burns tokens for no reason.

### First Wake-Up
On your very first tick after proactive mode is activated (or resumed), greet the
user briefly and ask what they'd like to work on. Do not start exploring the
codebase or making changes unprompted -- wait for direction.

### Subsequent Wake-Ups
Look for useful work. A good colleague faced with ambiguity doesn't just stop --
they investigate, reduce risk, and build understanding. Ask yourself: what don't
I know yet? What could go wrong?

Do not spam the user. If you already asked something and they haven't responded,
do not ask again. Do not narrate what you're about to do -- just do it.

### Staying Responsive
When the user is actively engaging with you, check for and respond to their
messages frequently. Treat real-time conversations like pairing -- keep the
feedback loop tight. If the user sends a message, prioritize responding over
continuing background work.

### Bias Toward Action
Act on your best judgment rather than asking for confirmation. Read files, search
code, explore the project, run tests, check types, run linters -- all without
asking. Make code changes. If you're unsure between two reasonable approaches,
pick one and go. You can always course-correct.

### Be Concise
Keep your text output brief and high-level. The user does not need a play-by-play.
Focus text output on decisions that need the user's input, high-level status
updates at natural milestones, and errors or blockers that change the plan."""

AUTONOMOUS_SECTION = """\
## Autonomous Teams
You can spawn autonomous teammates that self-organize around a shared task board.

### How It Works
1. Create a team with /team create <name>
2. Break work into small, independent tasks
3. Spawn autonomous teammates with the spawn_agent tool
4. Teammates scan the task board, claim available work, execute it, and report back
5. Faster agents naturally pick up more tasks — no central dispatching needed

### Task Board
- Use TaskCreate to add tasks with clear subjects and descriptions
- Set dependencies with addBlocks/addBlockedBy when order matters
- Tasks must be PENDING and unblocked to be claimed
- Claimed tasks become IN_PROGRESS, then COMPLETED on success

### Monitoring
- Use /tasks to see the current task board status
- Idle teammates send idle_notification when no work is available
- Use /autonomous to see which teammates are active
- Use /shutdown <agent> to gracefully stop a teammate

### Best Practices
- Break large goals into small, concrete tasks
- Keep tasks independent when possible (faster completion)
- Set dependencies only when strictly necessary
- Monitor for idle teammates — they may need more tasks"""

WORKTREE_SECTION = """\
## Worktree Isolation
You can create git worktrees for filesystem isolation between agents.

### Why Worktrees
When multiple agents edit the same files in the same directory, they create merge
conflicts. Git worktrees solve this by giving each agent its own working directory
on a separate branch.

### Creating Worktrees
- Use create_worktree to create an isolated working directory for an agent
- Each worktree is at .dazi/worktrees/<name> on branch agent-<name>
- Edits in one worktree don't affect another

### Finishing Worktrees
- Use finish_worktree with action='keep' to preserve the branch for manual merge
- Use finish_worktree with action='remove' to clean up entirely
- Safety: refuses to remove worktrees with uncommitted changes unless forced

### REPL Commands
- /worktree — list active worktrees
- /worktree create <name> — create a new worktree
- /worktree finish <name> — finish a worktree (prompts for keep/remove)
- /worktree finish <name> --keep — keep the branch
- /worktree finish <name> --remove — remove the worktree"""


# ─────────────────────────────────────────────────────────
# PROMPT BUILDER SINGLETON
# ─────────────────────────────────────────────────────────


def _build_prompt_sections(include_proactive: bool = False) -> str:
    """Build the DOING_TASKS custom section with or without proactive content."""
    sections = (
        original_doing_tasks
        + "\n\n"
        + TASK_MANAGEMENT_SECTION
        + "\n\n"
        + BACKGROUND_TASKS_SECTION
        + "\n\n"
        + MCP_TOOLS_SECTION
        + "\n\n"
        + SKILLS_GUIDANCE
        + "\n\n"
        + TEAM_MANAGEMENT_SECTION
        + "\n\n"
        + PROTOCOLS_SECTION
        + "\n\n"
        + AUTONOMOUS_SECTION
        + "\n\n"
        + WORKTREE_SECTION
    )
    if include_proactive:
        sections += "\n\n" + PROACTIVE_SECTION
    return sections


def _update_proactive_prompt() -> None:
    """Add or remove proactive section from system prompt based on state.

    Called before each graph invocation to keep the prompt in sync.
    """
    include = proactive_manager.is_proactive_active()
    prompt_builder.set_custom_section(
        PromptSection.DOING_TASKS,
        _build_prompt_sections(include_proactive=include),
    )


prompt_builder = SystemPromptBuilder()
# Add task management guidance to the DOING_TASKS section
original_doing_tasks = STATIC_SECTIONS.get(PromptSection.DOING_TASKS, "")
prompt_builder.set_custom_section(
    PromptSection.DOING_TASKS,
    _build_prompt_sections(include_proactive=False),
)


# ─────────────────────────────────────────────────────────
# PROMPT CONTENT HELPERS
# ─────────────────────────────────────────────────────────


def get_memory_content(user_query: str, limit: int = 5) -> str:
    """Find relevant memories for the current query."""
    memories = memory_store.find_relevant(user_query, limit=limit)
    if not memories:
        return ""

    parts = []
    for mem in memories:
        desc = mem.description or mem.content[:100]
        parts.append(f"**[{mem.category.value}]** {desc}")

    return "\n".join(parts)


def get_skills_content() -> str:
    """Build skills listing from the registry for prompt injection."""
    skills = skill_registry.list_user_invocable()
    if not skills:
        return ""
    lines = ["Available skills (use the `skill` tool or /<name> to invoke):"]
    for s in skills:
        hint = f" {s.argument_hint}" if s.argument_hint else ""
        lines.append(f"- {s.name}{hint}: {s.description}")
    return "\n".join(lines)
