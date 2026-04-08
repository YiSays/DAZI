"""Dazi — coding assistant REPL."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from rich.console import Console

from dazi import __version__
from dazi.dazimd import DaziMdFile
from dazi.graph import (
    EXECUTE_MODE,
    PLAN_MODE,
    connect_mcp_servers,
    hook_registry,
    permission_rules,
    run_graph_turn,
)
from dazi.llm import _get_model_name, _update_proactive_prompt
from dazi.proactive import format_tick, ProactiveSource
from dazi.lifecycle import load_dazimd, cleanup_on_exit
from dazi.repl_display import get_mode_badge, print_ascii_banner, print_welcome_message
from dazi.repl_commands import handle_command
from dazi.tokenizer import (
    count_messages_tokens,
    get_context_window,
    get_token_warning_state,
)
from dazi._singletons import (
    BACKGROUND_DIR,
    MEMORY_DIR,
    TASKS_DIR,
    background_manager,
    cost_tracker,
    memory_store,
    mcp_manager,
    proactive_manager,
    autonomous_teammate,
    settings_manager,
    skill_registry,
    task_store,
    team_manager,
    worktree_manager,
)
import dazi.graph as _graph_mod
import dazi.repl_teams as _teams
from dazi.config import DATA_DIR
from langgraph.errors import GraphInterrupt


console = Console()
dazimd_files: list[DaziMdFile] = []


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
    dazimd_files = load_dazimd(console=console)

    # Ensure directories exist
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    BACKGROUND_DIR.mkdir(parents=True, exist_ok=True)

    # Connect to MCP servers at startup
    await connect_mcp_servers()

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

    def print_welcome():
        print_welcome_message(console, skill_count=skill_count, team_count=team_count, dazimd_files=dazimd_files)

    print_ascii_banner(console, version=__version__)
    print_welcome()

    # Create .dazi directory if it doesn't exist
    (DATA_DIR / ".dazi" / "chat").mkdir(parents=True, exist_ok=True)
    session = PromptSession(
        history=FileHistory(DATA_DIR / ".dazi" / "chat" / "history")
    )
    state: dict = {"mode": EXECUTE_MODE, "messages": []}

    try:
        while True:
            try:
                # ── Build status bar ──
                mode_badge = get_mode_badge(state["mode"])
                rule_count = len(permission_rules)
                mem_count = len(memory_store.list_all())
                active_store = _teams.team_task_store if _teams.active_team_name else task_store
                tsk_count = len(active_store.list_all())
                bg_active = len(background_manager.list_active())

                model = _get_model_name()
                msgs = state.get("messages", [])
                display_msgs = [m for m in msgs if not isinstance(m, SystemMessage)]
                token_count = count_messages_tokens(display_msgs, model) if display_msgs else 0
                context_window = get_context_window(model)
                token_pct = (token_count / context_window * 100) if context_window > 0 else 0

                warning_state = get_token_warning_state(display_msgs, model) if display_msgs else "ok"
                token_color = {"ok": "green", "warning": "yellow", "compact": "red"}.get(warning_state, "white")

                info_parts = [f"{token_pct:.0f}%"]
                if proactive_manager.is_proactive_active():
                    badge = "PAUSED" if proactive_manager.is_proactive_paused() else "ACTIVE"
                    info_parts.insert(0, f"[PROACTIVE:{badge}]")
                autonomous_handles = autonomous_teammate.list_handles()
                if autonomous_handles:
                    active_count = len([h for h in autonomous_handles if h.status.value in ("active", "idle", "spawning")])
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
                if _teams.active_team_name:
                    info_parts.append(f"{_teams.active_team_name}")
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

                # ── Try built-in slash commands ──
                result = await handle_command(
                    cmd,
                    state=state,
                    session=session,
                    console=console,
                    dazimd_files=dazimd_files,
                    print_welcome_fn=print_welcome,
                )
                if result == "break":
                    break
                if result == "continue":
                    continue

                # ── Regular input: send to graph ──
                messages = state.get("messages", [])
                messages = [m for m in messages if not isinstance(m, SystemMessage)]
                messages.append(HumanMessage(content=user_input))
                state["messages"] = messages

                # Resume proactive on user input
                if proactive_manager.is_proactive_active():
                    proactive_manager.resume()
                _update_proactive_prompt()

                await run_graph_turn(
                    messages=messages,
                    state=state,
                    session=session,
                    status_label=f"Thinking... ({state['mode']} mode)",
                )

                # ── Proactive tick injection ──
                while proactive_manager.should_generate_tick():
                    await asyncio.sleep(0)
                    proactive_manager.mark_tick_sent()

                    tick_content = format_tick()
                    tick_msg = HumanMessage(
                        content=tick_content,
                        additional_kwargs={"is_meta": True, "is_tick": True},
                    )

                    _update_proactive_prompt()
                    tick_messages = state.get("messages", [])
                    tick_messages = [m for m in tick_messages if not isinstance(m, SystemMessage)]
                    tick_messages.append(tick_msg)

                    await run_graph_turn(
                        messages=tick_messages,
                        state=state,
                        session=session,
                        status_label="Thinking... (tick)",
                        label_suffix=" (tick)",
                    )

            except GraphInterrupt:
                console.print("\n[yellow]Interrupt escaped graph — this should not happen.[/yellow]")
            except KeyboardInterrupt:
                if proactive_manager.is_proactive_active():
                    proactive_manager.pause()
                    console.print("\n[dim]Proactive mode paused. Submit input to resume. Use /quit to exit.[/dim]")
                else:
                    console.print("\n[dim]Use /quit to exit.[/dim]")
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/red]")

    finally:
        await cleanup_on_exit(console=console, active_team_name=_teams.active_team_name)


if __name__ == "__main__":
    import asyncio

    asyncio.run(run_repl())
