"""Dazi — coding assistant REPL."""

import os
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console
from rich.columns import Columns
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from dazi import __version__
from dazi.background import (
    BackgroundTask,
    BackgroundTaskManager,
    BackgroundTaskStatus,
)
from dazi.dazimd import (
    DaziMdFile,
    discover_dazimd_files,
    merge_dazimd_content,
)
from dazi.compact import manual_compact
from dazi.hooks import HookEvent, HookResult
from dazi.llm import (
    _get_llm,
    _get_model_name,
    _update_proactive_prompt,
    prompt_builder,
)
from dazi.memory import MemoryEntry
from dazi.permissions import (
    PermissionBehavior,
    PermissionRule,
    parse_rule,
)
from dazi.proactive import format_tick, ProactiveSource
from dazi.coordinator import AutonomousTeammate, AutonomousConfig
from dazi.worktree import Worktree, WorktreeManager, WorktreeConfig
from dazi.task_store import Task, TaskStatus, TaskStore
from dazi.tokenizer import (
    count_messages_tokens,
    get_context_window,
    get_token_warning_state,
)
from dazi._singletons import (
    BACKGROUND_DIR,
    MEMORY_DIR,
    PLAN_FILE,
    TASKS_DIR,
    TEAMS_DIR,
    background_manager,
    cost_tracker,
    mailbox,
    memory_store,
    mcp_manager,
    permission_bridge,
    proactive_manager,
    autonomous_teammate,
    settings_manager,
    skill_registry,
    task_store,
    team_manager,
    teammate_runner,
    worktree_manager,
)
from dazi.registry import (
    EXECUTE_MODE_META,
    EXECUTE_MODE_TOOLS,
    PLAN_MODE_META,
    PLAN_MODE_TOOLS,
)
from dazi.mailbox import send_idle_notification
from dazi.skills import skill_tool
from dazi.graph import (
    PLAN_MODE,
    EXECUTE_MODE,
    AgentState,
    connect_mcp_servers,
    hook_registry,
    permission_rules,
    rebuild_tool_lists,
    run_graph_turn,
)
from langgraph.errors import GraphInterrupt
import dazi.graph as _graph_mod
from dazi.config import DATA_DIR


# ─────────────────────────────────────────────────────────
# DAZI.md LOADING
# ─────────────────────────────────────────────────────────


def load_dazimd() -> list[DaziMdFile]:
    """Discover and load DAZI.md files at startup."""
    project_root = Path.cwd()
    files = discover_dazimd_files(project_root=project_root, cwd=project_root)

    if files:
        merged = merge_dazimd_content(files)
        prompt_builder.set_dazimd_content(merged)
        console.print(f"[dim]Loaded {len(files)} DAZI.md file(s):[/dim]")
        for f in files:
            console.print(f"  [dim]{f.path} (priority: {f.priority})[/dim]")
    else:
        console.print("[dim]No DAZI.md files found.[/dim]")

    return files


async def cleanup_on_exit(*, say_goodbye: bool = False) -> None:
    """Shutdown all subsystems on REPL exit.

    Consolidates the cleanup sequence shared by /quit and the finally block.
    """
    proactive_manager.deactivate()

    for handle in autonomous_teammate.list_handles():
        await autonomous_teammate.shutdown(handle.team_name, handle.name)

    active_worktrees = worktree_manager.list_all()
    for wt in active_worktrees:
        try:
            worktree_manager.remove(wt.id, force=True)
        except Exception:
            pass
    if active_worktrees:
        console.print(f"[dim]Cleaned up {len(active_worktrees)} worktree(s).[/dim]")

    if active_team_name:
        count = await teammate_runner.shutdown_all(active_team_name)
        if count:
            console.print(
                f"[dim]Shut down {count} remaining teammate(s) for team '{active_team_name}'.[/dim]"
            )

    active = background_manager.list_active()
    if active:
        console.print(
            f"[dim]Cancelling {len(active)} remaining background task(s)...[/dim]"
        )
        for task in active:
            await background_manager.cancel(task.id)

    await mcp_manager.disconnect_all()
    cost_tracker.save()

    if say_goodbye:
        console.print("[dim]Goodbye![/dim]")


# ─────────────────────────────────────────────────────────
# MCP REPL HELPERS
# ─────────────────────────────────────────────────────────


def show_mcp_servers_table() -> None:
    """Show all MCP servers in a Rich table."""
    servers = mcp_manager.list_servers()
    if not servers:
        console.print("[dim]No MCP servers configured.[/dim]")
        console.print(
            '[dim]Add servers via settings.json: {"mcpServers": {"name": {"command": "...", "args": [...]}}}[/dim]'
        )
        return

    table = Table(title="MCP Servers")
    table.add_column("Name", style="cyan")
    table.add_column("Status")
    table.add_column("Tools", justify="right")
    table.add_column("Resources", justify="right")
    table.add_column("Command", style="dim")

    for s in servers:
        status = s["status"]
        if status == "connected":
            status_str = f"[green]{status}[/green]"
        elif status == "error":
            status_str = f"[red]{status}[/red]"
        else:
            status_str = f"[dim]{status}[/dim]"

        table.add_row(
            s["name"],
            status_str,
            str(s["tool_count"]),
            str(s["resource_count"]),
            s["command"],
        )

    console.print(table)
    console.print(
        "[dim]Commands: /mcp <name> for details, /mcp connect <name>, /mcp disconnect <name>[/dim]"
    )


def show_mcp_server_detail(server_name: str) -> None:
    """Show detailed info for a specific MCP server."""
    conn = mcp_manager.get_server(server_name)
    if conn is None:
        console.print(f"[red]Server '{server_name}' not found.[/red]")
        return

    from dazi.mcp_client import MCPServerStatus

    # Server info panel
    config_text = (
        f"Command: {conn.config.command}\n"
        f"Args: {' '.join(conn.config.args) if conn.config.args else '(none)'}\n"
        f"Status: {conn.status.value}"
    )
    if conn.error:
        config_text += f"\nError: {conn.error}"
    console.print(
        Panel(config_text, title=f"[cyan]{server_name}[/cyan]", border_style="blue")
    )

    # Tools
    if conn.tools:
        tool_table = Table(title=f"Tools ({len(conn.tools)})", show_lines=False)
        tool_table.add_column("Qualified Name", style="cyan")
        tool_table.add_column("Original", style="dim")
        tool_table.add_column("Read-Only")
        tool_table.add_column("Description", max_width=60)
        for t in conn.tools:
            ro = "[green]yes[/green]" if t.is_read_only else "[dim]no[/dim]"
            desc = (
                t.description[:60] + "..." if len(t.description) > 60 else t.description
            )
            tool_table.add_row(t.qualified_name, t.name, ro, desc)
        console.print(tool_table)
    elif conn.status == MCPServerStatus.CONNECTED:
        console.print("[dim]No tools discovered.[/dim]")

    # Resources
    if conn.resources:
        res_table = Table(title=f"Resources ({len(conn.resources)})", show_lines=False)
        res_table.add_column("URI", style="cyan")
        res_table.add_column("Name")
        res_table.add_column("MIME Type", style="dim")
        for r in conn.resources:
            res_table.add_row(str(r.uri), r.name, r.mime_type)
        console.print(res_table)


# ─────────────────────────────────────────────────────────
# SKILL REPL HELPERS
# ─────────────────────────────────────────────────────────


def show_skills_table() -> None:
    """Show all registered skills in a Rich table."""
    skills = skill_registry.list_all()
    if not skills:
        console.print("[dim]No skills loaded.[/dim]")
        return

    table = Table(title=f"Skills ({len(skills)})")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Source", style="dim")
    table.add_column("Invocable")

    for s in skills:
        source = (
            "bundled"
            if s.is_bundled
            else str(s.source_path.parent.name) if s.source_path else "unknown"
        )
        invocable = "[green]yes[/green]" if s.user_invocable else "[dim]no[/dim]"
        desc = s.description[:50] + "..." if len(s.description) > 50 else s.description
        table.add_row(f"/{s.name}", desc, source, invocable)

    console.print(table)


def show_skill_detail(skill_name: str) -> None:
    """Show detailed information about a specific skill."""
    skill = skill_registry.get(skill_name)
    if skill is None:
        console.print(f"[red]Skill '{skill_name}' not found.[/red]")
        return

    lines = [
        f"[bold]/{skill.name}[/bold]",
        f"Description: {skill.description}",
        f"Version: {skill.version}",
        f"User-invocable: {'yes' if skill.user_invocable else 'no'}",
    ]
    if skill.argument_hint:
        lines.append(f"Argument hint: {skill.argument_hint}")
    if skill.arguments:
        lines.append(f"Arguments: {', '.join(skill.arguments)}")
    if skill.when_to_use:
        lines.append(f"When to use: {skill.when_to_use}")
    if skill.allowed_tools:
        lines.append(f"Allowed tools: {', '.join(skill.allowed_tools)}")
    if skill.model:
        lines.append(f"Model override: {skill.model}")
    if skill.effort:
        lines.append(f"Effort: {skill.effort}")
    source = (
        "bundled"
        if skill.is_bundled
        else str(skill.source_path) if skill.source_path else "unknown"
    )
    lines.append(f"Source: {source}")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"Skill: {skill.name}",
            border_style="cyan",
        )
    )

    # Show the prompt template
    console.print(
        Panel(
            Markdown(skill.prompt),
            title="Prompt Template",
            border_style="blue",
        )
    )


# ─────────────────────────────────────────────────────────
# TEAM REPL HELPERS
# ─────────────────────────────────────────────────────────

# Module-level team state (session-level, not per-conversation-turn)
active_team_name: str | None = None
current_agent_name: str | None = None
team_task_store: TaskStore | None = None


def _require_team() -> bool:
    """Return True if an active team and agent identity are set."""
    if not active_team_name:
        console.print(
            "[red]No active team. Use /team <name> to activate a team first.[/red]"
        )
        return False
    if not current_agent_name:
        console.print("[red]Agent identity not set.[/red]")
        return False
    return True


def show_teams_table() -> None:
    """Show all teams in a Rich table."""
    teams = team_manager.list_teams()
    if not teams:
        console.print(
            "[dim]No teams exist. Use /team create <name> to create one.[/dim]"
        )
        return

    table = Table(title=f"Teams ({len(teams)})")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Members", justify="right")
    table.add_column("Active", justify="right")
    table.add_column("Created", style="dim")
    table.add_column("Status", style="dim")

    for t in teams:
        active = sum(1 for m in t.members if m.status == "active")
        idle = sum(1 for m in t.members if m.status == "idle")
        completed = sum(1 for m in t.members if m.status == "completed")
        status = f"{active}A/{idle}I/{completed}C"
        created = t.created_at[:10] if t.created_at else "?"
        desc = t.description[:40] + "..." if len(t.description) > 40 else t.description
        table.add_row(t.name, desc, str(len(t.members)), str(active), created, status)

    console.print(table)
    console.print(
        "[dim]Commands: /team <name> to activate, /team delete <name> to remove[/dim]"
    )


def show_team_detail(team_name: str) -> None:
    """Show detailed information about a specific team."""
    team = team_manager.get_team(team_name)
    if team is None:
        console.print(f"[red]Team '{team_name}' not found.[/red]")
        return

    lines = [
        f"[bold]{team.name}[/bold]",
        f"Description: {team.description or '(no description)'}",
        f"Created: {team.created_at}",
        f"Members ({len(team.members)}):",
    ]

    if team.members:
        status_icons = {
            "active": "[green]+[/green] active",
            "idle": "[yellow]=[/yellow] idle",
            "completed": "[dim]✓[/dim] completed",
        }
        for m in team.members:
            icon = status_icons.get(m.status, f"[?]{m.status}[/?]")
            lines.append(
                f"  {icon}  [bold]{m.name}[/bold] ({m.agent_id}) — {m.agent_type}"
            )
    else:
        lines.append("  (no members)")

    lines.append(f"\nConfig: {team_manager._config_path(team_name)}")
    lines.append(f"Tasks:  {team_manager._task_dir(team_name)}")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"Team: {team.name}",
            border_style="cyan",
        )
    )


def activate_team(name: str) -> None:
    """Switch to a team context, creating a team-scoped task store."""
    global active_team_name, current_agent_name, team_task_store
    import dazi._singletons as _singletons

    team = team_manager.get_team(name)
    if team is None:
        console.print(f"[red]Team '{name}' not found.[/red]")
        return

    active_team_name = name
    current_agent_name = TEAM_LEAD_NAME  # REPL user is always the leader

    # Sync with tools module
    _singletons.active_team_name = name
    _singletons.current_agent_name = TEAM_LEAD_NAME

    # Ensure inbox directory exists
    mailbox._ensure_inbox_dir(name)

    task_dir = team_manager._task_dir(name)
    team_task_store = TaskStore(task_dir, list_id="default")
    console.print(f"[green]Switched to team: {name} (as team-lead)[/green]")
    console.print(f"[dim]Task board: {task_dir}[/dim]")
    console.print(
        f"[dim]Inbox: {team_manager.teams_dir / team_manager._sanitize_name(name) / 'inboxes' / f'{TEAM_LEAD_NAME}.json'}[/dim]"
    )
    if team.members:
        console.print(f"[dim]Members: {', '.join(m.name for m in team.members)}[/dim]")


def deactivate_team() -> None:
    """Deactivate the current team, returning to the default task store."""
    global active_team_name, current_agent_name, team_task_store
    import dazi._singletons as _singletons

    if active_team_name:
        console.print(f"[dim]Left team: {active_team_name}[/dim]")
    active_team_name = None
    current_agent_name = None
    team_task_store = None

    # Sync with tools module
    _singletons.active_team_name = None
    _singletons.current_agent_name = None


# ─────────────────────────────────────────────────────────
# INBOX REPL HELPERS
# ─────────────────────────────────────────────────────────


async def show_inbox(agent_name: str | None = None) -> None:
    """Show inbox messages for an agent."""
    if not active_team_name:
        console.print(
            "[red]No active team. Use /team <name> to activate a team first.[/red]"
        )
        return

    target = agent_name or current_agent_name
    if not target:
        console.print("[red]Agent identity not set.[/red]")
        return

    messages = await mailbox.receive(
        team_name=active_team_name,
        agent_name=target,
        unread_only=True,
        limit=20,
    )

    if not messages:
        label = f"{agent_name}'s" if agent_name else "your"
        console.print(f"[dim]No unread messages in {label} inbox.[/dim]")
        return

    # Mark as read
    msg_ids = [m.id for m in messages]
    await mailbox.mark_read(
        team_name=active_team_name,
        agent_name=target,
        message_ids=msg_ids,
    )

    title = f"Inbox: {target}" + (" (peek)" if agent_name else "")
    table = Table(title=title)
    table.add_column("From", style="cyan")
    table.add_column("Type", style="bold")
    table.add_column("Summary", max_width=50)
    table.add_column("Time", style="dim")

    for msg in messages:
        type_tag = msg.msg_type if msg.msg_type != "text" else ""
        time_short = msg.timestamp[:19] if msg.timestamp else "?"
        summary = msg.summary or msg.text[:50]
        table.add_row(msg.from_agent, type_tag, summary, time_short)

    console.print(table)
    console.print(f"[dim]{len(messages)} message(s) marked as read.[/dim]")


async def send_repl_message(agent_name: str, text: str) -> None:
    """Send a message to a specific teammate via REPL."""
    if not _require_team():
        return

    if agent_name == current_agent_name:
        console.print(
            f"[red]Cannot send a message to yourself ({current_agent_name}).[/red]"
        )
        return

    from dazi.protocols import create_text_message

    msg = create_text_message(
        from_agent=current_agent_name,
        to_agent=agent_name,
        text=text,
    )

    recipients = await mailbox.send(
        team_name=active_team_name,
        message=msg,
    )

    if recipients:
        console.print(f"[green]Message sent to {agent_name}.[/green]")
    else:
        console.print(f"[red]Could not deliver message to '{agent_name}'.[/red]")


async def broadcast_repl_message(text: str) -> None:
    """Broadcast a message to all teammates via REPL."""
    if not _require_team():
        return

    team = team_manager.get_team(active_team_name)
    if not team or not team.members:
        console.print(f"[red]No team members found for '{active_team_name}'.[/red]")
        return

    from dazi.protocols import create_text_message

    msg = create_text_message(
        from_agent=current_agent_name,
        to_agent="*",
        text=text,
    )

    team_members = [m.name for m in team.members]
    recipients = await mailbox.send(
        team_name=active_team_name,
        message=msg,
        team_members=team_members,
    )

    console.print(
        f"[green]Broadcast sent to {len(recipients)} teammate(s): {', '.join(recipients)}[/green]"
    )


async def send_shutdown_request(agent_name: str) -> None:
    """Send a shutdown request to a specific teammate via REPL."""
    if not _require_team():
        return

    from dazi.protocols import create_shutdown_request

    msg = create_shutdown_request(
        from_agent=current_agent_name,
        to_agent=agent_name,
    )

    recipients = await mailbox.send(
        team_name=active_team_name,
        message=msg,
    )

    if recipients:
        console.print(f"[yellow]Shutdown request sent to {agent_name}.[/yellow]")
    else:
        console.print(f"[red]Could not send shutdown request to '{agent_name}'.[/red]")


# ─────────────────────────────────────────────────────────
# REPL
# ─────────────────────────────────────────────────────────
# Commands:
#   /plan    — enter plan mode
#   /go      — exit plan mode
#   /show    — display plan file
#   /tools   — list tools for current mode
#   /rules   — list permission rules
#   /allow   — add allow rule
#   /deny    — add deny rule
#   /hooks   — list registered hooks
#   /hook    — add a demo hook
#   /remember <content> — store a memory
#   /forget <id>        — delete a memory
#   /memories           — list all memories
#   /dazimd           — show loaded DAZI.md files
#   /reindex            — rebuild memory index
#   /compact            — manual full compact
#   /tokens             — show token count and context window
#   /tasks              — show task board
#   /task <id>          — show single task details
#   /bg                 — list background tasks
#   /bg <task_id>       — show background task detail
#   /cost               — show session cost summary
#   /cost last          — show previous session cost
#   /settings           — show settings with source annotations
#   /reload             — reload settings + reconnect MCP servers + reload skills
#   /mcp                — list MCP servers
#   /mcp <name>         — show MCP server details
#   /mcp connect <name> — connect MCP server
#   /mcp disconnect <name> — disconnect MCP server
#   /skills             — list available skills
#   /skill <name>       — show skill details
#   /<skill-name> [args]— invoke a skill
#   /teams              — list all teams
#   /team <name>        — activate a team
#   /team create <name> — create a new team
#   /team delete <name> — delete a team
#   /inbox              — check your inbox
#   /inbox <agent>      — check an agent's inbox
#   /send <agent> <msg> — send a DM to a teammate
#   /broadcast <msg>    — broadcast to all teammates
#   /team delete <name> — delete a team
#   /clear   — reset conversation
#   /quit    — exit

console = Console()
dazimd_files: list[DaziMdFile] = []


def get_mode_badge(mode: str) -> list[tuple[str, str]]:
    if mode == PLAN_MODE:
        return [("bold fg:yellow", "PLAN")]
    return [("bold fg:green", "EXECUTE")]


def list_rules_table() -> None:
    table = Table(title="Permission Rules")
    table.add_column("Behavior", style="bold")
    table.add_column("Tool", style="cyan")
    table.add_column("Pattern", style="dim")
    table.add_column("Source", style="dim")

    all_rules = _get_effective_rules()
    for rule in all_rules:
        behavior_color = {
            PermissionBehavior.ALLOW: "[green]ALLOW[/green]",
            PermissionBehavior.DENY: "[red]DENY[/red]",
            PermissionBehavior.ASK: "[yellow]ASK[/yellow]",
        }
        table.add_row(
            behavior_color.get(rule.behavior, str(rule.behavior)),
            rule.tool_name or "*",
            rule.pattern or "*",
            rule.source,
        )
    console.print(table)


def list_memories_table() -> None:
    entries = memory_store.list_all()
    if not entries:
        console.print("[dim]No memories stored.[/dim]")
        return

    table = Table(title=f"Memories ({len(entries)} entries)")
    table.add_column("ID", style="cyan")
    table.add_column("Category")
    table.add_column("Description")
    table.add_column("Created", style="dim")

    for entry in entries:
        desc = entry.description or entry.content[:60]
        if len(desc) > 60:
            desc = desc[:57] + "..."
        table.add_row(entry.id, entry.category.value, desc, entry.created_at[:10])
    console.print(table)


def show_dazimd_files() -> None:
    if not dazimd_files:
        console.print("[dim]No DAZI.md files loaded.[/dim]")
        return

    table = Table(title="Loaded DAZI.md Files")
    table.add_column("Path", style="cyan")
    table.add_column("Priority")
    table.add_column("Size", style="dim")

    for f in dazimd_files:
        table.add_row(str(f.path), str(f.priority), f"{len(f.content)} chars")
    console.print(table)

    merged = merge_dazimd_content(dazimd_files)
    if merged:
        console.print(
            Panel(
                Markdown(merged),
                title="Merged DAZI.md Content",
                border_style="blue",
            )
        )


def show_token_info(messages: list) -> None:
    """Display token usage information."""
    model = _get_model_name()
    token_count = count_messages_tokens(messages, model) if messages else 0
    context_window = get_context_window(model)
    threshold = context_window - 13_000
    warning_state = get_token_warning_state(messages, model) if messages else "ok"

    pct = (token_count / context_window * 100) if context_window > 0 else 0

    state_colors = {
        "ok": "green",
        "warning": "yellow",
        "compact": "red",
        "error": "bold red",
    }
    state_color = state_colors.get(warning_state, "white")

    table = Table(title="Token Usage")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Model", model)
    table.add_row("Context Window", f"{context_window:,} tokens")
    table.add_row("Compact Threshold", f"{threshold:,} tokens")
    table.add_row("Current Usage", f"{token_count:,} tokens ({pct:.1f}%)")
    table.add_row("Warning State", f"[{state_color}]{warning_state}[/{state_color}]")
    table.add_row("Message Count", str(len(messages)))
    table.add_row("Compact Failures", f"{_graph_mod.consecutive_compact_failures}/3")

    console.print(table)


def list_tasks_table() -> None:
    """Display all tasks in a formatted table (team-scoped or default)."""
    store = team_task_store if active_team_name else task_store
    tasks = store.list_all()
    if not tasks:
        console.print("[dim]No tasks on the board.[/dim]")
        return

    table = Table(title=f"Task Board ({len(tasks)} tasks)")
    table.add_column("ID", style="cyan", justify="right")
    table.add_column("Status")
    table.add_column("Subject")
    table.add_column("Owner", style="dim")
    table.add_column("Blocked By", style="dim")

    status_colors = {
        "pending": "[white]pending[/white]",
        "in_progress": "[yellow]in_progress[/yellow]",
        "completed": "[green]completed[/green]",
    }

    for task in tasks:
        active_blockers = store.get_active_blockers(task.id)
        table.add_row(
            str(task.id),
            status_colors.get(task.status.value, task.status.value),
            task.subject,
            task.owner or "-",
            ", ".join(str(b) for b in active_blockers) if active_blockers else "-",
        )
    console.print(table)


def show_task_detail(task_id: int) -> None:
    """Display full details of a single task (team-scoped or default)."""
    store = team_task_store if active_team_name else task_store
    task = store.get(task_id)
    if task is None:
        console.print(f"[red]Task #{task_id} not found.[/red]")
        return

    lines = [
        f"[bold]#{task.id}[/bold] [{task.status.value}] [bold]{task.subject}[/bold]",
        "",
        task.description,
    ]
    if task.active_form:
        lines.append(f"\n[dim]Active form: {task.active_form}[/dim]")
    if task.owner:
        lines.append(f"[dim]Owner: {task.owner}[/dim]")
    if task.blocks:
        lines.append(f"[dim]Blocks: {task.blocks}[/dim]")
    if task.blocked_by:
        lines.append(f"[dim]Blocked by: {task.blocked_by}[/dim]")
    if task.metadata:
        lines.append(f"[dim]Metadata: {task.metadata}[/dim]")
    lines.append(f"[dim]Created: {task.created_at}[/dim]")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"Task #{task.id}",
            border_style="cyan",
        )
    )


# ─────────────────────────────────────────────────────────
# REPL HELPERS — Background tasks
# ─────────────────────────────────────────────────────────


def show_background_tasks_table() -> None:
    """Display all background tasks in a Rich table. command."""
    tasks = background_manager.list_all()
    if not tasks:
        console.print("[dim]No background tasks.[/dim]")
        return

    table = Table(title=f"Background Tasks ({len(tasks)})")
    table.add_column("ID", style="cyan")
    table.add_column("Status")
    table.add_column("Command", max_width=50)
    table.add_column("PID", style="dim", justify="right")
    table.add_column("Duration", style="dim", justify="right")
    table.add_column("Exit", style="dim", justify="right")

    status_colors = {
        "pending": "[white]pending[/white]",
        "running": "[yellow]running[/yellow]",
        "completed": "[green]completed[/green]",
        "failed": "[red]failed[/red]",
        "killed": "[bold red]killed[/bold red]",
    }

    for task in tasks:
        cmd = task.command[:50] + ("..." if len(task.command) > 50 else "")
        pid_str = str(task.pid) if task.pid else "-"
        dur_str = f"{task.duration_seconds:.1f}s" if task.duration_seconds else "-"
        exit_str = str(task.exit_code) if task.exit_code is not None else "-"
        table.add_row(
            task.id,
            status_colors.get(task.status.value, task.status.value),
            cmd,
            pid_str,
            dur_str,
            exit_str,
        )
    console.print(table)


def show_background_task_detail(task_id: str) -> None:
    """Show full details of a background task including output. command."""
    task = background_manager.check_sync(task_id)
    if task is None:
        console.print(f"[red]Background task '{task_id}' not found.[/red]")
        return

    lines = [
        f"[bold]{task.id}[/bold] [{task.status.value}]",
        f"[bold]Command:[/bold] {task.command}",
    ]
    if task.description:
        lines.append(f"[bold]Description:[/bold] {task.description}")
    if task.pid:
        lines.append(f"[dim]PID: {task.pid}[/dim]")
    if task.duration_seconds:
        lines.append(f"[dim]Duration: {task.duration_seconds:.1f}s[/dim]")
    if task.exit_code is not None:
        lines.append(f"[dim]Exit code: {task.exit_code}[/dim]")
    if task.error:
        lines.append(f"[red]Error: {task.error}[/red]")

    # Show output
    output = background_manager.get_output_tail(task_id, lines=30)
    if output:
        lines.append(f"\n[bold]Output (last 30 lines):[/bold]")
        lines.append("```")
        lines.append(output)
        lines.append("```")

    status_style = {
        BackgroundTaskStatus.RUNNING: "yellow",
        BackgroundTaskStatus.COMPLETED: "green",
        BackgroundTaskStatus.FAILED: "red",
        BackgroundTaskStatus.KILLED: "bold red",
        BackgroundTaskStatus.PENDING: "white",
    }
    border = status_style.get(task.status, "blue")

    console.print(
        Panel(
            "\n".join(lines),
            title=f"Background Task: {task.id}",
            border_style=border,
        )
    )


def add_demo_hook() -> None:
    async def logging_hook(
        tool_name: str = "", tool_args: dict = None, **kwargs
    ) -> HookResult:
        args_display = str(tool_args or {})[:100]
        console.print(f"  [dim][hook] pre_tool_use: {tool_name}({args_display})[/dim]")
        return HookResult()

    hook_registry.register(HookEvent.PRE_TOOL_USE, logging_hook, priority=100)
    console.print("[green]Registered logging hook (priority=100).[/green]")


# ─────────────────────────────────────────────────────────
# REPL LOOP
# ─────────────────────────────────────────────────────────


async def run_repl() -> None:
    import asyncio
    from prompt_toolkit import PromptSession
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.history import FileHistory

    # Load DAZI.md at startup
    global dazimd_files
    dazimd_files = load_dazimd()

    # Ensure directories exist
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)

    memory_count = len(memory_store.list_all())
    task_count = len(task_store.list_all())
    bg_count = len(background_manager.list_active())

    # Connect to MCP servers at startup
    await connect_mcp_servers()
    mcp_tool_count = len(mcp_manager.get_tools())

    # Load skills at startup
    skill_count = skill_registry.load_skills()

    # Count existing teams
    team_count = len(team_manager.list_teams())

    # Check for proactive env var activation
    import os as _os

    if _os.getenv("DAZI_PROACTIVE", "").lower() in ("1", "true", "on"):
        proactive_manager.activate(source=ProactiveSource.ENV)
        _update_proactive_prompt()
        console.print("[dim]Proactive mode activated via DAZI_PROACTIVE env var.[/dim]")

    def print_ascii_banner():
        """Print DAZI ASCII art banner with Rich two-column layout."""
        from datetime import datetime
        from rich.columns import Columns
        from rich.align import Align

        # Correct blocky block-style DAZI
        block_lines = [
            # "==== Develop Autonomously ========",
            "██████╗    ████╗  ███████╗ ██████╗",
            "██╔══██╗  ██╔═██╗ ╚══███╔╝ ╚═██╔═╝",
            "██║  ██║  ██║ ██║   ███╔╝    ██║",
            "██║  ██║ ███████║  ███╔╝     ██║",
            "██████╔╝ ██╔══██║ ███████╗ ██████╗",
            "╚════╝   ╚═╝  ╚═╝ ╚══════╝ ╚═════╝",
            # "============= Zero Interruption===",
        ]

        # Create DAZI logo as aligned text
        logo_text = "\n".join(block_lines)
        logo_panel = Panel(
            Align.center(logo_text, vertical="middle"),
            style="bold cyan",
            padding=(0, 1),
        )

        # Time-based greeting
        now = datetime.now()
        hour = now.hour
        if hour < 6:
            greeting = "Late night coding"
        elif hour < 12:
            greeting = "Good morning"
        elif hour < 17:
            greeting = "Good afternoon"
        elif hour < 21:
            greeting = "Good evening"
        else:
            greeting = "Late night coding"

        # Greeting content
        messages = [
            f"[bold magenta]{greeting}! 💡[/bold magenta]",
            "",
            f"[dim]{now:%A, %B %d, %Y — %H:%M}[/dim]",
            "",
            "🔮 DAZI is ready, and Happy Coding! ✨",
            # "",
            "[italic cyan]DAZI - Develop Autonomously, Zero Interruption.[/italic cyan]",
        ]
        greeting_panel = Panel(
            "\n".join(messages),
            title="[bold cyan]  DAZI  ",
            subtitle=f"[dim]v{__version__}[/dim]",
            border_style="bright_cyan",
            padding=(0, 2),
        )

        # Two column layout
        console.print()
        console.print(Columns([logo_panel, greeting_panel], expand=True, equal=False))
        console.print()

    def print_welcome_message():

        console.print(
            Panel(
                "Commands: /plan, /go, /show, /tools, /rules, /allow, /deny,\n"
                "          /hooks, /hook, /remember, /forget, /memories,\n"
                "          /dazimd, /reindex, /compact, /tokens,\n"
                "          /tasks, /task <id>, /bg, /bg <task_id>,\n"
                "          /cost, /cost last, /settings, /reload,\n"
                "          /mcp, /mcp <name>, /mcp connect/disconnect,\n"
                "          /skills, /skill <name>, /<skill-name> [args],\n"
                "          /teams, /team <name>, /team create, /team delete,\n"
                "          /inbox, /send <agent> <msg>, /broadcast <msg>, /shutdown <agent>,\n"
                "          /proactive, /proactive on, /proactive off, /autonomous,\n"
                "          /worktree, /worktree create <name>, /worktree finish <name> --keep/--remove,\n"
                "          /clear, /quit",
                border_style="blue",
            )
        )

        model = _get_model_name()
        settings_rules = settings_manager.get_permission_rules()
        _cur_task_store = team_task_store if active_team_name else task_store
        console.print(
            f"\n[dim]Model: {model} | Context: {get_context_window(model):,} tokens | "
            f"Settings rules: {len(settings_rules)} | "
            f"Memories: {len(memory_store.list_all())} | Tasks: {len(_cur_task_store.list_all())} | "
            f"Background: {len(background_manager.list_active())} active | Skills: {skill_count} | "
            f"Teams: {team_count} | "
            f"DAZI.md: {len(dazimd_files)} file(s)[/dim]"
        )

    print_ascii_banner()
    print_welcome_message()

    # Create .dazi directory if it doesn't exist
    (DATA_DIR / ".dazi" / "chat").mkdir(parents=True, exist_ok=True)
    session = PromptSession(
        history=FileHistory(DATA_DIR / ".dazi" / "chat" / "history")
    )
    state: dict = {"mode": EXECUTE_MODE, "messages": []}

    try:
        while True:
            try:
                mode_badge = get_mode_badge(state["mode"])
                rule_count = len(permission_rules)
                mem_count = len(memory_store.list_all())
                active_store = team_task_store if active_team_name else task_store
                tsk_count = len(active_store.list_all())
                bg_active = len(background_manager.list_active())

                # Token count for status bar
                model = _get_model_name()
                msgs = state.get("messages", [])
                display_msgs = [m for m in msgs if not isinstance(m, SystemMessage)]
                token_count = (
                    count_messages_tokens(display_msgs, model) if display_msgs else 0
                )
                context_window = get_context_window(model)
                token_pct = (
                    (token_count / context_window * 100) if context_window > 0 else 0
                )

                warning_state = (
                    get_token_warning_state(display_msgs, model)
                    if display_msgs
                    else "ok"
                )
                token_color = {
                    "ok": "green",
                    "warning": "yellow",
                    "compact": "red",
                }.get(warning_state, "white")

                info_parts = [f"{token_pct:.0f}%"]
                if proactive_manager.is_proactive_active():
                    badge = (
                        "PAUSED"
                        if proactive_manager.is_proactive_paused()
                        else "ACTIVE"
                    )
                    info_parts.insert(0, f"[PROACTIVE:{badge}]")
                autonomous_handles = autonomous_teammate.list_handles()
                if autonomous_handles:
                    active_count = len(
                        [
                            h
                            for h in autonomous_handles
                            if h.status.value in ("active", "idle", "spawning")
                        ]
                    )
                    info_parts.insert(0, f"[AUTONOMOUS:{active_count}]")
                active_worktrees = worktree_manager.list_all()
                if active_worktrees:
                    info_parts.insert(0, f"[WT:{len(active_worktrees)}]")
                if rule_count:
                    info_parts.append(f"{rule_count} rules")
                if mem_count:
                    info_parts.append(f"{mem_count} mem")
                if tsk_count:
                    info_parts.append(f"{tsk_count} tasks")
                if bg_active:
                    info_parts.append(f"{bg_active} bg")
                mcp_tools_count = len(mcp_manager.get_tools())
                if mcp_tools_count:
                    info_parts.append(f"{mcp_tools_count} mcp")
                if active_team_name:
                    info_parts.append(f"{active_team_name}")
                cost_str = cost_tracker.format_cost()
                info_parts.append(cost_str)
                info_text = "[" + " | ".join(info_parts) + "]"
                user_input = await session.prompt_async(
                    FormattedText(
                        [
                            ("", "\n"),
                            *mode_badge,
                            (f"fg:{token_color}", f" {info_text} "),
                            ("bold fg:green", "You: "),
                        ]
                    ),
                )
                if not user_input.strip():
                    continue

                cmd = user_input.strip()

                # ── /quit ──
                if cmd == "/quit":
                    await cleanup_on_exit(say_goodbye=True)
                    break

                # ── /cost — show session cost summary ──
                if cmd == "/cost" or cmd.startswith("/cost "):
                    if cmd == "/cost last":
                        console.print(
                            Panel(
                                cost_tracker.format_last_session(),
                                title="Previous Session",
                                border_style="blue",
                            )
                        )
                    else:
                        console.print(
                            Panel(
                                cost_tracker.format_summary(),
                                title="Session Cost",
                                border_style="green",
                            )
                        )
                    continue

                # ── /settings — show settings table with source annotations ──
                if cmd == "/settings":
                    s = settings_manager.settings
                    sm = settings_manager.source_map
                    table = Table(title="Settings")
                    table.add_column("Field", style="bold")
                    table.add_column("Value")
                    table.add_column("Source", style="dim")
                    for f_name, f_value in [
                        ("model", s.model),
                        ("api_base_url", str(s.api_base_url)),
                        ("default_mode", s.default_mode),
                        ("auto_compact", str(s.auto_compact)),
                        ("auto_memory", str(s.auto_memory)),
                        ("max_concurrent_tools", str(s.max_concurrent_tools)),
                        ("allow_rules", f"{len(s.allow_rules)} rule(s)"),
                        ("deny_rules", f"{len(s.deny_rules)} rule(s)"),
                        ("env", f"{len(s.env)} var(s)"),
                    ]:
                        source = sm.get(f_name, "default")
                        source_color = {"user": "yellow", "project": "green"}.get(
                            source, "dim"
                        )
                        table.add_row(
                            f_name,
                            str(f_value),
                            f"[{source_color}]{source}[/{source_color}]",
                        )
                    console.print(table)
                    console.print(f"\n[dim]User:   {settings_manager.user_path}[/dim]")
                    console.print(
                        f"[dim]Project: {settings_manager.project_path}[/dim]"
                    )
                    continue

                # ── /reload — reload settings + skills + MCP  ──
                if cmd == "/reload":
                    settings_manager.reload()
                    console.print("[green]Settings reloaded.[/green]")

                    # Reload skills (discover new/changed skills from disk)
                    new_skill_count = skill_registry.reload()
                    console.print(
                        f"[green]Skills reloaded: {new_skill_count} skill(s).[/green]"
                    )

                    # Reconnect MCP servers with new config
                    await mcp_manager.disconnect_all()
                    await connect_mcp_servers()
                    continue

                # ── /mcp — MCP server management ──
                if cmd == "/mcp":
                    show_mcp_servers_table()
                    continue

                if cmd.startswith("/mcp "):
                    mcp_arg = cmd[5:].strip()
                    if mcp_arg.startswith("connect "):
                        server_name = mcp_arg[8:].strip()
                        if server_name:
                            from dazi.mcp_client import (
                                MCPServerConfig,
                                MCPServerStatus,
                            )

                            conn = mcp_manager.get_server(server_name)
                            if conn and conn.status == MCPServerStatus.CONNECTED:
                                console.print(
                                    f"[yellow]{server_name} is already connected.[/yellow]"
                                )
                            else:
                                console.print(
                                    f"[dim]Connecting to {server_name}...[/dim]"
                                )
                                success = await mcp_manager.connect_server(server_name)
                                if success:
                                    c = mcp_manager.get_server(server_name)
                                    console.print(
                                        f"[green]+ {server_name} ({len(c.tools)} tools)[/green]"
                                    )
                                    # Rebuild tool lists
                                    rebuild_tool_lists()
                                else:
                                    err = mcp_manager.get_server(server_name)
                                    console.print(
                                        f"[red]Failed: {err.error if err else 'unknown error'}[/red]"
                                    )
                        else:
                            console.print(
                                "[red]Usage: /mcp connect <server_name>[/red]"
                            )
                    elif mcp_arg.startswith("disconnect "):
                        server_name = mcp_arg[10:].strip()
                        if server_name:
                            await mcp_manager.disconnect_server(server_name)
                            console.print(f"[dim]Disconnected {server_name}.[/dim]")
                            rebuild_tool_lists()
                        else:
                            console.print(
                                "[red]Usage: /mcp disconnect <server_name>[/red]"
                            )
                    else:
                        # Show specific server details
                        show_mcp_server_detail(mcp_arg)
                    continue

                # ── /clear ──
                if cmd == "/clear":
                    state["messages"] = []
                    _graph_mod.consecutive_compact_failures = 0
                    _os.system("clear" if _os.name != "nt" else "cls")
                    print_welcome_message()
                    console.print(
                        "[dim]Cleared. Memories, tasks, and background tasks persist.[/dim]"
                    )
                    continue

                # ── /plan ──
                if cmd == "/plan":
                    if state["mode"] == PLAN_MODE:
                        console.print("[yellow]Already in plan mode.[/yellow]")
                        continue
                    state["mode"] = PLAN_MODE
                    console.print(
                        Panel(
                            "[bold yellow]PLAN MODE[/bold yellow]\n"
                            "Read-only tools + plan_writer + memory tools + task tools + background tools enabled.\n"
                            "Type /go to exit plan mode.",
                            border_style="yellow",
                        )
                    )
                    continue

                # ── /go ──
                if cmd == "/go":
                    if state["mode"] == EXECUTE_MODE:
                        console.print("[yellow]Not in plan mode.[/yellow]")
                        continue
                    state["mode"] = EXECUTE_MODE
                    if PLAN_FILE.exists():
                        plan_content = PLAN_FILE.read_text(encoding="utf-8")
                        console.print(
                            Panel(
                                Markdown(plan_content),
                                title="[bold green]Plan[/bold green]",
                                border_style="green",
                            )
                        )
                    console.print(
                        "[bold green]EXECUTE MODE[/bold green] -- all tools enabled."
                    )
                    continue

                # ── /show ──
                if cmd == "/show":
                    if not PLAN_FILE.exists():
                        console.print("[dim]No plan file found.[/dim]")
                        continue
                    console.print(
                        Panel(
                            Markdown(PLAN_FILE.read_text(encoding="utf-8")),
                            title="Plan File",
                            border_style="blue",
                        )
                    )
                    continue

                # ── /tools ──
                if cmd == "/tools":
                    mode = state["mode"]
                    meta_dict = (
                        PLAN_MODE_META if mode == PLAN_MODE else EXECUTE_MODE_META
                    )
                    tools_list = (
                        PLAN_MODE_TOOLS if mode == PLAN_MODE else EXECUTE_MODE_TOOLS
                    )
                    safety_tags = {
                        "safe": "[green]safe[/green]",
                        "write": "[yellow]write[/yellow]",
                        "destructive": "[red]destructive[/red]",
                    }
                    console.print(f"\n[bold]Tools ({mode} mode):[/bold]")
                    for tool in tools_list:
                        meta = meta_dict.get(tool.name)
                        if meta:
                            tag = safety_tags.get(meta.safety.value, meta.safety.value)
                            concurrent = (
                                "[dim]parallel[/dim]"
                                if meta.is_concurrency_safe
                                else "[dim]serial[/dim]"
                            )
                            console.print(
                                f"  * {meta.name} -- {meta.description} ({tag}, {concurrent})"
                            )
                    continue

                # ── /rules ──
                if cmd == "/rules":
                    list_rules_table()
                    continue

                # ── /allow <rule> ──
                if cmd.startswith("/allow "):
                    rule_str = cmd[7:].strip()
                    try:
                        rule = parse_rule(f"allow {rule_str}", "cli")
                        permission_rules.append(rule)
                        console.print(
                            f"[green]Added rule: ALLOW {rule.tool_name or '*'} {rule.pattern or ''}[/green]"
                        )
                    except ValueError as e:
                        console.print(f"[red]Invalid rule: {e}[/red]")
                    continue

                # ── /deny <rule> ──
                if cmd.startswith("/deny "):
                    rule_str = cmd[6:].strip()
                    try:
                        rule = parse_rule(f"deny {rule_str}", "cli")
                        permission_rules.append(rule)
                        console.print(
                            f"[red]Added rule: DENY {rule.tool_name or '*'} {rule.pattern or ''}[/red]"
                        )
                    except ValueError as e:
                        console.print(f"[red]Invalid rule: {e}[/red]")
                    continue

                # ── /hooks ──
                if cmd == "/hooks":
                    hooks = hook_registry.list_hooks()
                    if not hooks:
                        console.print("[dim]No hooks registered.[/dim]")
                    else:
                        for event, priorities in hooks.items():
                            console.print(
                                f"  {event}: {len(priorities)} handler(s), priorities: {priorities}"
                            )
                    continue

                # ── /hook (demo) ──
                if cmd == "/hook":
                    add_demo_hook()
                    continue

                # ── /remember <content> ──
                if cmd.startswith("/remember "):
                    content = cmd[10:].strip()
                    if not content:
                        console.print("[red]Usage: /remember <content>[/red]")
                        continue
                    from dazi.memory import MemoryCategory

                    memory_entry = MemoryEntry(
                        content=content, category=MemoryCategory("user")
                    )
                    memory_store.write(memory_entry)
                    console.print(f"[green]Remembered: {memory_entry.id}[/green]")
                    continue

                # ── /forget <id> ──
                if cmd.startswith("/forget "):
                    mem_id = cmd[8:].strip()
                    if not mem_id:
                        console.print("[red]Usage: /forget <memory-id>[/red]")
                        continue
                    entries = memory_store.list_all()
                    matches = [e for e in entries if e.id.startswith(mem_id)]
                    if len(matches) == 1:
                        memory_store.delete(matches[0].id)
                        console.print(f"[green]Forgotten: {matches[0].id}[/green]")
                    elif len(matches) > 1:
                        console.print(
                            f"[yellow]Multiple matches: {[e.id for e in matches]}. Be more specific.[/yellow]"
                        )
                    else:
                        if memory_store.delete(mem_id):
                            console.print(f"[green]Forgotten: {mem_id}[/green]")
                        else:
                            console.print(f"[red]Memory not found: {mem_id}[/red]")
                    continue

                # ── /memories ──
                if cmd == "/memories":
                    list_memories_table()
                    continue

                # ── /dazimd ──
                if cmd == "/dazimd":
                    show_dazimd_files()
                    continue

                # ── /reindex ──
                if cmd == "/reindex":
                    memory_store.rebuild_index()
                    console.print("[green]Memory index rebuilt.[/green]")
                    continue

                # ── /compact ──
                if cmd == "/compact":
                    msgs = state.get("messages", [])
                    if len(msgs) < 2:
                        console.print("[dim]Not enough messages to compact.[/dim]")
                        continue

                    model = _get_model_name()
                    tokens_before = count_messages_tokens(msgs, model)
                    console.print(
                        f"[dim]Compressing... ({tokens_before:,} tokens)[/dim]"
                    )

                    result = await manual_compact(msgs, _get_llm(), model=model)

                    if result.method != "none":
                        saved = result.tokens_before - result.tokens_after
                        console.print(
                            f"[green]Compacted ({result.method}):[/green] "
                            f"{result.tokens_before:,} -> {result.tokens_after:,} tokens "
                            f"(saved {saved:,})"
                        )
                        state["messages"] = result.messages
                    else:
                        console.print(
                            f"[dim]{result.summary or 'No compaction needed.'}[/dim]"
                        )
                    continue

                # ── /tokens ──
                if cmd == "/tokens":
                    show_token_info(state.get("messages", []))
                    continue

                # ── /tasks — show task board ──
                if cmd == "/tasks":
                    list_tasks_table()
                    continue

                # ── /task <id> — show single task ──
                if cmd.startswith("/task "):
                    task_id_str = cmd[6:].strip()
                    try:
                        task_id = int(task_id_str)
                        show_task_detail(task_id)
                    except ValueError:
                        console.print("[red]Usage: /task <id>[/red]")
                    continue

                # ── /bg — list background tasks ──
                if cmd == "/bg":
                    show_background_tasks_table()
                    continue

                # ── /bg <task_id> — show background task detail ──
                if cmd.startswith("/bg "):
                    task_id = cmd[4:].strip()
                    show_background_task_detail(task_id)
                    continue

                # ── /skills — list available skills ──
                if cmd == "/skills":
                    show_skills_table()
                    continue

                # ── /skill <name> — show skill details ──
                if cmd.startswith("/skill "):
                    skill_name = cmd[7:].strip()
                    show_skill_detail(skill_name)
                    continue

                # ── /teams — list all teams ──
                if cmd == "/teams":
                    show_teams_table()
                    continue

                # ── /team <name> — team management ──
                if cmd.startswith("/team "):
                    team_arg = cmd[6:].strip()
                    if team_arg.startswith("create "):
                        team_name = team_arg[7:].strip()
                        if not team_name:
                            console.print("[red]Usage: /team create <name>[/red]")
                            continue
                        try:
                            team = team_manager.create_team(team_name)
                            sanitized = team_manager._sanitize_name(team_name)
                            console.print(f"[green]Team created: {team.name}[/green]")
                            console.print(
                                f"[dim]Config: {team_manager._config_path(team_name)}[/dim]"
                            )
                            console.print(
                                f"[dim]Tasks:  {team_manager._task_dir(team_name)}[/dim]"
                            )
                        except Exception as e:
                            console.print(f"[red]Error: {e}[/red]")
                        continue
                    elif team_arg == "leave":
                        if active_team_name:
                            deactivate_team()
                        else:
                            console.print("[dim]No active team to leave.[/dim]")
                        continue
                    elif team_arg.startswith("delete "):
                        team_name = team_arg[7:].strip()
                        if not team_name:
                            console.print("[red]Usage: /team delete <name>[/red]")
                            continue
                        if active_team_name == team_name:
                            deactivate_team()
                        try:
                            if team_manager.delete_team(team_name):
                                console.print(
                                    f"[green]Team deleted: {team_name}[/green]"
                                )
                            else:
                                console.print(
                                    f"[red]Team '{team_name}' not found.[/red]"
                                )
                        except Exception as e:
                            console.print(f"[red]Error: {e}[/red]")
                        continue
                    else:
                        # /team <name> — activate team
                        if team_arg:
                            activate_team(team_arg)
                        else:
                            # /team with no arg — show teams
                            show_teams_table()
                        continue

                # ── /proactive — proactive mode management ──
                if cmd == "/proactive" or cmd.startswith("/proactive "):
                    proactive_arg = cmd[10:].strip() if len(cmd) > 10 else ""
                    if proactive_arg == "on":
                        proactive_manager.activate(source=ProactiveSource.COMMAND)
                        _update_proactive_prompt()
                        console.print(
                            Panel(
                                "[bold green]PROACTIVE MODE ON[/bold green]\n"
                                "The agent will wake up periodically via tick prompts.\n"
                                "Use /proactive off to stop. Ctrl+C pauses ticks.",
                                border_style="green",
                            )
                        )
                    elif proactive_arg == "off":
                        proactive_manager.deactivate()
                        _update_proactive_prompt()
                        console.print("[dim]Proactive mode off.[/dim]")
                    else:
                        # Show status
                        state_desc = proactive_manager.state.value
                        source_desc = (
                            proactive_manager.source.value
                            if proactive_manager.source
                            else "none"
                        )
                        count = proactive_manager.activation_count
                        if state_desc == "inactive":
                            console.print(
                                f"[dim]Proactive: {state_desc} | Use /proactive on to activate[/dim]"
                            )
                        else:
                            first = "yes" if proactive_manager.is_first_tick else "no"
                            last_tick = proactive_manager.last_tick_time or "never"
                            console.print(
                                f"[bold]Proactive:[/bold] {state_desc} | Source: {source_desc} | "
                                f"Activations: {count} | First tick: {first} | Last tick: {last_tick}"
                            )
                    continue

                # ── /autonomous — autonomous team status ──
                if cmd == "/autonomous" or cmd.startswith("/autonomous "):
                    autonomous_handles = autonomous_teammate.list_handles()
                    if not autonomous_handles:
                        console.print("[dim]No autonomous teammates running.[/dim]")
                    else:
                        table = Table(title="Autonomous Teammates", show_lines=True)
                        table.add_column("Name", style="cyan")
                        table.add_column("Team", style="green")
                        table.add_column("Status", style="yellow")
                        table.add_column("Tasks Claimed", style="magenta")
                        for h in autonomous_handles:
                            claimed = autonomous_teammate._tasks_claimed.get(h.name, 0)
                            table.add_row(
                                h.name, h.team_name, h.status.value, str(claimed)
                            )
                        console.print(table)
                    continue

                # ── /worktree — worktree management ──
                if cmd == "/worktree" or cmd.startswith("/worktree "):
                    wt_arg = cmd[10:].strip() if len(cmd) > 10 else ""
                    if wt_arg.startswith("create "):
                        wt_name = wt_arg[7:].strip()
                        if not wt_name:
                            console.print("[red]Usage: /worktree create <name>[/red]")
                            continue
                        try:
                            wt = worktree_manager.create(wt_name)
                            console.print(
                                f"[green]Created worktree:[/green] {wt.path} on branch {wt.branch}"
                            )
                            console.print(
                                f"[dim]Use /worktree finish {wt_name} when done.[/dim]"
                            )
                        except (ValueError, RuntimeError) as e:
                            console.print(f"[red]Error: {e}[/red]")
                    elif wt_arg.startswith("finish "):
                        finish_parts = wt_arg[7:].strip().split()
                        if not finish_parts:
                            console.print(
                                "[red]Usage: /worktree finish <name> [--keep|--remove][/red]"
                            )
                            continue
                        finish_name = finish_parts[0]
                        action_flag = (
                            finish_parts[1] if len(finish_parts) > 1 else "--keep"
                        )
                        slug = worktree_manager.sanitize_agent_name(finish_name)
                        wt = worktree_manager.get(slug)
                        if wt is None:
                            console.print(
                                f"[red]No worktree found for '{finish_name}'.[/red]"
                            )
                            continue
                        if action_flag in ("--keep", "--remove"):
                            action = action_flag[2:]  # strip --
                        else:
                            action = "keep"
                        if action == "keep":
                            branch = worktree_manager.keep(slug)
                            console.print(
                                f"[green]Kept worktree:[/green] branch '{branch}' preserved at {wt.path}"
                            )
                        elif action == "remove":
                            if worktree_manager.has_uncommitted_changes(slug):
                                console.print(
                                    "[yellow]Worktree has uncommitted changes. Use --remove with --force or keep instead.[/yellow]"
                                )
                                removed = worktree_manager.remove(slug, force=False)
                            else:
                                removed = worktree_manager.remove(slug, force=True)
                            if removed:
                                console.print(
                                    f"[green]Removed worktree:[/green] {finish_name}"
                                )
                            else:
                                console.print(f"[red]Failed to remove worktree.[/red]")
                    else:
                        # List worktrees
                        active_worktrees = worktree_manager.list_all()
                        if not active_worktrees:
                            console.print("[dim]No active worktrees.[/dim]")
                        else:
                            table = Table(
                                title=f"Worktrees ({len(active_worktrees)})",
                                show_lines=True,
                            )
                            table.add_column("Name", style="cyan")
                            table.add_column("Branch", style="green")
                            table.add_column("Path", style="dim")
                            table.add_column("Dirty")
                            for wt in active_worktrees:
                                dirty = (
                                    "[yellow]yes[/yellow]"
                                    if worktree_manager.has_uncommitted_changes(wt.id)
                                    else "[green]no[/green]"
                                )
                                table.add_row(
                                    wt.agent_name, wt.branch, str(wt.path), dirty
                                )
                            console.print(table)
                    continue

                # ── /inbox — check inbox ──
                if cmd == "/inbox" or cmd.startswith("/inbox "):
                    inbox_agent = cmd[7:].strip() if cmd.startswith("/inbox ") else None
                    await show_inbox(inbox_agent if inbox_agent else None)
                    continue

                # ── /send <agent> <msg> — send DM ──
                if cmd.startswith("/send "):
                    parts = cmd[6:].strip().split(None, 1)
                    if len(parts) < 2:
                        console.print("[red]Usage: /send <agent-name> <message>[/red]")
                    else:
                        await send_repl_message(parts[0], parts[1])
                    continue

                # ── /broadcast <msg> — broadcast ──
                if cmd.startswith("/broadcast "):
                    msg_text = cmd[11:].strip()
                    if not msg_text:
                        console.print("[red]Usage: /broadcast <message>[/red]")
                    else:
                        await broadcast_repl_message(msg_text)
                    continue

                # ── /shutdown <agent> — send shutdown request ──
                if cmd.startswith("/shutdown "):
                    agent = cmd[10:].strip()
                    if not agent:
                        console.print("[red]Usage: /shutdown <agent-name>[/red]")
                    else:
                        await send_shutdown_request(agent)
                    continue

                # ── Slash command expansion ──
                # Check if the input matches a user-invocable skill.
                # This happens AFTER all built-in commands so /plan, /quit etc. take priority.
                if cmd.startswith("/"):
                    parts = cmd.split(None, 1)
                    skill_name = parts[0][1:]  # strip leading "/"
                    skill_args = parts[1] if len(parts) > 1 else ""

                    if skill_registry.has_skill(skill_name):
                        skill = skill_registry.get(skill_name)
                        if not skill.user_invocable:
                            console.print(
                                f"[red]Skill '{skill_name}' is not user-invocable.[/red]"
                            )
                            continue

                        expanded = skill_registry.expand_skill(skill_name, skill_args)
                        console.print(f"[dim]Expanded skill: /{skill_name}[/dim]")

                        # Inject expanded prompt as HumanMessage
                        messages = state.get("messages", [])
                        messages = [
                            m for m in messages if not isinstance(m, SystemMessage)
                        ]
                        messages.append(HumanMessage(content=expanded))
                        state["messages"] = messages

                        result = await run_graph_turn(
                            messages=messages,
                            state=state,
                            session=session,
                            status_label=f"Thinking... (skill: {skill_name})",
                        )
                        continue

                # ── Regular input: send to graph ──
                messages = state.get("messages", [])
                messages = [m for m in messages if not isinstance(m, SystemMessage)]
                messages.append(HumanMessage(content=user_input))
                state["messages"] = messages

                # Resume proactive on user input (user is engaging)
                if proactive_manager.is_proactive_active():
                    proactive_manager.resume()

                # Update prompt for proactive state
                _update_proactive_prompt()

                result = await run_graph_turn(
                    messages=messages,
                    state=state,
                    session=session,
                    status_label=f"Thinking... ({state['mode']} mode)",
                )

                # ── Proactive tick injection ──
                # After processing user input, check if proactive tick should fire.
                # The tick loop continues until proactive is paused or agent sleeps.
                while proactive_manager.should_generate_tick():
                    await asyncio.sleep(
                        0
                    )  # Yield to event loop (setTimeout(0) equivalent)
                    proactive_manager.mark_tick_sent()

                    tick_content = format_tick()
                    tick_msg = HumanMessage(
                        content=tick_content,
                        additional_kwargs={"is_meta": True, "is_tick": True},
                    )

                    _update_proactive_prompt()
                    tick_messages = state.get("messages", [])
                    tick_messages = [
                        m for m in tick_messages if not isinstance(m, SystemMessage)
                    ]
                    tick_messages.append(tick_msg)

                    tick_result = await run_graph_turn(
                        messages=tick_messages,
                        state=state,
                        session=session,
                        status_label="Thinking... (tick)",
                        label_suffix=" (tick)",
                    )

            except GraphInterrupt:
                console.print(
                    "\n[yellow]Interrupt escaped graph — this should not happen.[/yellow]"
                )
            except KeyboardInterrupt:
                if proactive_manager.is_proactive_active():
                    proactive_manager.pause()
                    console.print(
                        "\n[dim]Proactive mode paused. Submit input to resume. Use /quit to exit.[/dim]"
                    )
                else:
                    console.print("\n[dim]Use /quit to exit.[/dim]")
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")

    finally:
        await cleanup_on_exit()


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_repl())
