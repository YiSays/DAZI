"""LangGraph agent pipeline for Dazi.

This module owns the agent graph: node functions, routing, graph construction,
streaming display, and the main execution wrapper (run_graph_turn).
"""

from __future__ import annotations

import uuid
from typing import Annotated

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text
from typing_extensions import TypedDict

from dazi._singletons import (
    PLAN_FILE,
    background_manager,
    cost_tracker,
    mcp_manager,
    settings_manager,
)
from dazi.background import BackgroundTask, BackgroundTaskStatus
from dazi.compact import auto_compact
from dazi.concurrent import execute_tools_concurrent
from dazi.hooks import HookEvent, HookRegistry
from dazi.llm import (
    _get_llm,
    _get_model_name,
    get_memory_content,
    get_skills_content,
    prompt_builder,
)
from dazi.mcp_client import (
    list_mcp_resources_tool,
    list_mcp_servers_tool,
    read_mcp_resource_tool,
)
from dazi.permissions import (
    PermissionBehavior,
    PermissionMode,
    PermissionResult,
    PermissionRule,
    check_permission,
    derive_permission_pattern,
    prompt_permission_decisions,
)
from dazi.registry import (
    EXECUTE_MODE_META,
    EXECUTE_MODE_TOOLS,
    PLAN_MODE_META,
    PLAN_MODE_TOOLS,
)
from dazi.tokenizer import (
    count_messages_tokens,
    get_context_window,
    should_auto_compact,
)

console = Console()


# ─────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    mode: str  # "plan" or "execute"
    permission_rules: list[str]  # serialized rules for state
    allowed_tool_ids: list[str]  # tool IDs that passed permission check
    consecutive_compact_failures: int  # circuit breaker counter


# ─────────────────────────────────────────────────────────
# MODE CONSTANTS
# ─────────────────────────────────────────────────────────

PLAN_MODE = "plan"
EXECUTE_MODE = "execute"

# ─────────────────────────────────────────────────────────
# MODULE-LEVEL STATE (shared with main.py REPL)
# ─────────────────────────────────────────────────────────

permission_rules: list[PermissionRule] = []
hook_registry = HookRegistry()
consecutive_compact_failures: int = 0


# ─────────────────────────────────────────────────────────
# PERMISSION HELPERS
# ─────────────────────────────────────────────────────────


def _get_effective_rules() -> list[PermissionRule]:
    """Combine settings rules with user-added CLI rules."""
    settings_rules = settings_manager.get_permission_rules()
    return settings_rules + permission_rules


# ─────────────────────────────────────────────────────────
# TOOL LIST ASSEMBLY (with dynamic MCP tools)
# ─────────────────────────────────────────────────────────


def _build_full_tool_lists() -> tuple[list, list]:
    """Build tool lists including dynamic MCP tools.

    Returns (execute_tools, plan_tools).

    MCP server tools are added dynamically. Read-only MCP tools go to plan mode,
    all MCP tools go to execute mode. MCP management tools go to both modes.
    """
    # MCP management tools (available in both modes)
    mcp_mgmt_tools = [
        list_mcp_servers_tool,
        list_mcp_resources_tool,
        read_mcp_resource_tool,
    ]

    # Dynamic MCP server tools
    mcp_langchain_tools = mcp_manager.build_langchain_tools()

    # Filter MCP tools by safety for plan mode (read-only only)
    safe_mcp = [t for t in mcp_langchain_tools if t.metadata.get("mcp_is_read_only", False)]
    write_mcp = [t for t in mcp_langchain_tools if not t.metadata.get("mcp_is_read_only", False)]

    execute_tools = list(EXECUTE_MODE_TOOLS) + mcp_mgmt_tools + safe_mcp + write_mcp
    plan_tools = list(PLAN_MODE_TOOLS) + mcp_mgmt_tools + safe_mcp

    return execute_tools, plan_tools


# Build initial tool lists (will be rebuilt after MCP connections)
EXECUTE_TOOLS_FULL, PLAN_TOOLS_FULL = _build_full_tool_lists()


def rebuild_tool_lists() -> None:
    """Rebuild EXECUTE_TOOLS_FULL and PLAN_TOOLS_FULL after MCP changes."""
    global EXECUTE_TOOLS_FULL, PLAN_TOOLS_FULL
    EXECUTE_TOOLS_FULL, PLAN_TOOLS_FULL = _build_full_tool_lists()


# ─────────────────────────────────────────────────────────
# MCP SERVER CONNECTION
# ─────────────────────────────────────────────────────────


async def connect_mcp_servers() -> None:
    """Connect to MCP servers from settings at startup."""
    global EXECUTE_TOOLS_FULL, PLAN_TOOLS_FULL

    mcp_servers = settings_manager.get_mcp_servers()
    if not mcp_servers:
        console.print("[dim]No MCP servers configured in settings.[/dim]")
        return

    from dazi.mcp_client import MCPServerConfig

    for name, config_dict in mcp_servers.items():
        try:
            config = MCPServerConfig.from_dict(name, config_dict)
            mcp_manager.add_server(config)
        except Exception as e:
            console.print(f"[yellow]Warning: Invalid MCP server config '{name}': {e}[/yellow]")

    console.print(f"[dim]Connecting to {len(mcp_servers)} MCP server(s)...[/dim]")
    results = await mcp_manager.connect_all()

    for name, success in results.items():
        if success:
            conn = mcp_manager.get_server(name)
            tool_count = len(conn.tools) if conn else 0
            console.print(f"  [green]+[/green] {name} ({tool_count} tools)")
        else:
            error = mcp_manager.get_server(name)
            err_msg = error.error if error else "unknown"
            console.print(f"  [red]![/red] {name}: {err_msg}")

    # Rebuild tool lists with newly discovered MCP tools
    EXECUTE_TOOLS_FULL, PLAN_TOOLS_FULL = _build_full_tool_lists()
    total_mcp = len(mcp_manager.get_tools())
    console.print(
        f"[dim]MCP ready: "
        f"{len(results) - sum(not v for v in results.values())} "
        f"connected, {total_mcp} tools[/dim]"
    )


# ─────────────────────────────────────────────────────────
# GRAPH NODE: check_compact
# ─────────────────────────────────────────────────────────


async def check_compact(state: AgentState) -> dict:
    """Check if compaction is needed and run it."""
    global consecutive_compact_failures
    messages = state.get("messages", [])
    model = _get_model_name()

    if len(messages) < 4:
        return {"messages": []}

    if not should_auto_compact(messages, model, consecutive_compact_failures):
        return {"messages": []}

    token_count = count_messages_tokens(messages, model)
    threshold = get_context_window(model) - 13_000

    console.print(
        f"[yellow]Auto-compact triggered:[/yellow] "
        f"{token_count:,} tokens (threshold: {threshold:,})"
    )

    result = await auto_compact(
        messages,
        _get_llm(),
        model=model,
        consecutive_failures=consecutive_compact_failures,
    )

    if result.method != "none":
        saved = result.tokens_before - result.tokens_after
        console.print(
            f"[green]Compacted ({result.method}):[/green] "
            f"{result.tokens_before:,} -> {result.tokens_after:,} tokens "
            f"(saved {saved:,})"
        )
        if result.tool_results_cleared:
            console.print(f"  [dim]Cleared {result.tool_results_cleared} old tool results[/dim]")
        if result.rounds_removed:
            console.print(
                f"  [dim]Summarized {result.rounds_removed} old conversation rounds[/dim]"
            )

        consecutive_compact_failures = 0
        return {"messages": result.messages}
    else:
        if result.summary and "failed" in result.summary.lower():
            consecutive_compact_failures += 1
            console.print(f"[red]Compact failed:[/red] {result.summary}")
            console.print(f"  [dim]Circuit breaker: {consecutive_compact_failures}/3[/dim]")

    return {"messages": []}


# ─────────────────────────────────────────────────────────
# GRAPH NODE: call_llm
# ─────────────────────────────────────────────────────────


async def call_llm(state: AgentState) -> dict:
    """Call the LLM with dynamically built system prompt."""
    messages = state["messages"]
    mode = state.get("mode", EXECUTE_MODE)

    user_query = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break

    memory_content = get_memory_content(user_query)
    skills_content = get_skills_content()
    rules = _get_effective_rules()
    has_plan = mode == EXECUTE_MODE and PLAN_FILE.exists()

    sys_prompt = prompt_builder.build(
        mode=mode,
        user_query=user_query,
        memory_content=memory_content,
        skills_content=skills_content,
        rule_count=len(rules),
        has_plan=has_plan,
    )

    if not messages or not isinstance(messages[0], SystemMessage):
        messages = [SystemMessage(content=sys_prompt)] + list(messages)
    else:
        messages = [SystemMessage(content=sys_prompt)] + [
            m for m in messages if not isinstance(m, SystemMessage)
        ]

    tools = PLAN_TOOLS_FULL if mode == PLAN_MODE else EXECUTE_TOOLS_FULL
    llm = _get_llm().bind_tools(tools)

    # Stream LLM response and merge chunks into a single AIMessage
    chunks: list = []
    async for chunk in llm.astream(messages):
        chunks.append(chunk)

    if not chunks:
        return {"messages": []}

    response = chunks[0]
    for chunk in chunks[1:]:
        response = response + chunk

    # Extract token usage for cost tracking
    usage = response.response_metadata.get("token_usage", {}) or {}
    input_tokens = usage.get("prompt_tokens", 0)
    output_tokens = usage.get("completion_tokens", 0)
    if input_tokens or output_tokens:
        cost_tracker.record_usage(
            model=settings_manager.get_model_name(),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    return {"messages": [response]}


# ─────────────────────────────────────────────────────────
# GRAPH NODE: check_permissions
# ─────────────────────────────────────────────────────────


async def check_permissions(state: AgentState) -> dict:
    """Check permissions for all tool calls from the last AI message."""
    messages = state["messages"]
    last_message = messages[-1]

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": []}

    mode = state.get("mode", EXECUTE_MODE)
    rules = _get_effective_rules()
    perm_mode = PermissionMode.PLAN if mode == PLAN_MODE else PermissionMode.DEFAULT
    tool_meta = PLAN_MODE_META if mode == PLAN_MODE else EXECUTE_MODE_META

    allowed_ids: list[str] = []
    denied_messages: list[ToolMessage] = []
    ask_tools: list[dict] = []

    for tc in last_message.tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_id = tc["id"]

        hook_result = await hook_registry.fire(
            HookEvent.PRE_TOOL_USE,
            tool_name=tool_name,
            tool_args=tool_args,
        )

        if hook_result.should_block:
            denied_messages.append(
                ToolMessage(
                    content=f"BLOCKED by hook: {hook_result.block_reason}",
                    tool_call_id=tool_id,
                )
            )
            continue

        effective_args = hook_result.modified_input if hook_result.modified_input else tool_args

        if hook_result.permission_override:
            perm_result = PermissionResult(
                behavior=hook_result.permission_override,
                reason="Permission overridden by hook",
            )
        else:
            meta = tool_meta.get(tool_name)
            if meta is None:
                # MCP tools are not in mode meta dicts — derive safety from MCP metadata
                mcp_tool_info = mcp_manager.get_tool(tool_name)
                if mcp_tool_info:
                    safety = "safe" if mcp_tool_info.is_read_only else "write"
                else:
                    safety = "destructive"
            else:
                safety = meta.safety.value
            perm_result = check_permission(tool_name, effective_args, rules, perm_mode, safety)

        if perm_result.behavior == PermissionBehavior.DENY:
            denied_messages.append(
                ToolMessage(
                    content=(
                        f"DENIED: {perm_result.reason}. "
                        f"The tool '{tool_name}' was blocked by permission rules."
                    ),
                    tool_call_id=tool_id,
                )
            )
        elif perm_result.behavior == PermissionBehavior.ASK:
            ask_tools.append(
                {
                    "tool_call_id": tool_id,
                    "tool_name": tool_name,
                    "tool_args": effective_args,
                    "reason": perm_result.reason,
                }
            )
        else:
            tc["args"] = effective_args
            allowed_ids.append(tool_id)

    if ask_tools:
        decisions = interrupt({"ask_tools": ask_tools})
        for ask in ask_tools:
            tid = ask["tool_call_id"]
            raw = decisions.get(tid, {"action": "deny"})
            # Support legacy string format ("allow"/"deny") and new dict format
            if isinstance(raw, str):
                decision = {"action": raw}
            else:
                decision = raw
            action = decision.get("action", "deny")

            if action == "allow":
                pattern = derive_permission_pattern(ask["tool_name"], ask["tool_args"])
                permission_rules.append(
                    PermissionRule(
                        behavior=PermissionBehavior.ALLOW,
                        tool_name=ask["tool_name"],
                        pattern=pattern,
                        source="cli",
                    )
                )
                for tc in last_message.tool_calls:
                    if tc["id"] == tid:
                        tc["args"] = ask["tool_args"]
                allowed_ids.append(tid)
            elif action == "skip":
                # Skip execution, inject user's message as tool result.
                # No rule added — future similar calls will still ASK.
                user_msg = decision.get("message", "Skipped by user.")
                denied_messages.append(
                    ToolMessage(
                        content=f"SKIPPED by user: {ask['tool_name']} — {user_msg}",
                        tool_call_id=tid,
                    )
                )
            else:
                permission_rules.append(
                    PermissionRule(
                        behavior=PermissionBehavior.DENY,
                        tool_name=ask["tool_name"],
                        source="cli",
                    )
                )
                denied_messages.append(
                    ToolMessage(
                        content=f"DENIED by user: {ask['tool_name']} was not approved.",
                        tool_call_id=tid,
                    )
                )

    if allowed_ids:
        return {"messages": denied_messages, "allowed_tool_ids": allowed_ids}
    elif denied_messages:
        return {"messages": denied_messages, "allowed_tool_ids": []}
    else:
        return {"messages": [], "allowed_tool_ids": []}


# ─────────────────────────────────────────────────────────
# GRAPH NODE: execute_tools
# ─────────────────────────────────────────────────────────


async def execute_tools(state: AgentState) -> dict:
    """Execute allowed tool calls, then run POST_TOOL_USE hooks."""
    messages = state["messages"]
    last_message = messages[-1]
    allowed_ids = state.get("allowed_tool_ids", [])

    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        return {"messages": []}

    mode = state.get("mode", EXECUTE_MODE)
    # Use _FULL lists (includes dynamic MCP tools + skill tool)
    available_tools = PLAN_TOOLS_FULL if mode == PLAN_MODE else EXECUTE_TOOLS_FULL
    tool_meta = PLAN_MODE_META if mode == PLAN_MODE else EXECUTE_MODE_META

    allowed_calls = [tc for tc in last_message.tool_calls if tc["id"] in allowed_ids]

    if not allowed_calls:
        return {"messages": []}

    tool_messages = await execute_tools_concurrent(
        allowed_calls,
        available_tools,
        tool_meta,
        max_concurrent=5,
    )

    for i, tc in enumerate(allowed_calls):
        if i < len(tool_messages):
            hook_result = await hook_registry.fire(
                HookEvent.POST_TOOL_USE,
                tool_name=tc["name"],
                tool_args=tc["args"],
                tool_output=tool_messages[i].content,
            )
            if hook_result.modified_output is not None:
                tool_messages[i].content = hook_result.modified_output

    return {"messages": tool_messages}


# ─────────────────────────────────────────────────────────
# ROUTING
# ─────────────────────────────────────────────────────────


def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "check_permissions"
    return END


def has_allowed_tools(state: AgentState) -> str:
    allowed_ids = state.get("allowed_tool_ids", [])
    if allowed_ids:
        return "execute_tools"
    return "call_llm"


# ─────────────────────────────────────────────────────────
# GRAPH CONSTRUCTION
# ─────────────────────────────────────────────────────────

graph = StateGraph(AgentState)
graph.add_node("check_compact", check_compact)
graph.add_node("call_llm", call_llm)
graph.add_node("check_permissions", check_permissions)
graph.add_node("execute_tools", execute_tools)

graph.add_edge(START, "check_compact")
graph.add_edge("check_compact", "call_llm")
graph.add_conditional_edges(
    "call_llm",
    should_continue,
    {
        "check_permissions": "check_permissions",
        END: END,
    },
)
graph.add_conditional_edges(
    "check_permissions",
    has_allowed_tools,
    {
        "execute_tools": "execute_tools",
        "call_llm": "call_llm",
    },
)
graph.add_edge("execute_tools", "check_compact")

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# ─────────────────────────────────────────────────────────
# SPINNER MANAGER
# ─────────────────────────────────────────────────────────


class SpinnerManager:
    """Wraps a Rich Live spinner with start/stop/update_label control.

    Designed to persist across multiple stream-and-resume cycles within a
    single graph turn, only pausing for prompt_toolkit interactions that
    bypass Rich's Live display.
    """

    def __init__(self, initial_label: str = "Thinking...") -> None:
        self._spinner = Spinner("dots", Text(initial_label, style="bold cyan"))
        self._live = Live(
            self._spinner,
            console=console,
            refresh_per_second=8,
            transient=True,
        )
        self._started = False

    def start(self) -> None:
        if not self._started:
            self._live.start()
            self._started = True

    def stop(self) -> None:
        if self._started:
            self._live.stop()
            self._started = False

    def update_label(self, label: str) -> None:
        self._spinner = Spinner("dots", Text(label, style="bold cyan"))
        if self._started:
            self._live.update(self._spinner)


# ─────────────────────────────────────────────────────────
# STREAMING DISPLAY
# ─────────────────────────────────────────────────────────


def _print_tool_call_compact(tool_call: dict) -> None:
    """Print a tool call in compact single-line format."""
    name = tool_call["name"]
    args = str(tool_call["args"])
    if len(args) > 120:
        args = args[:120] + "..."
    console.print(f"  [dim]\u2502[/dim] [bold cyan]{name}[/bold cyan]([dim]{args}[/dim])")


def _print_tool_result_compact(content: str, *, is_error: bool = False) -> None:
    """Print a tool result in compact single-line format."""
    first_line = content.split("\n")[0]
    if len(first_line) > 120:
        first_line = first_line[:120] + "..."
    if is_error:
        console.print(f"  [red]\u2514 {first_line}[/red]")
    else:
        console.print(f"  [dim]\u2514 {first_line}[/dim]")


async def _stream_and_display(
    stream,
    *,
    spinner: SpinnerManager,
) -> None:
    """Consume astream_events and print live output to console.

    Pure event renderer — does NOT manage the spinner lifecycle.
    The caller (run_graph_turn) starts/stops the spinner.

    Events handled:
      - on_chat_model_stream: accumulate text for Markdown rendering
      - on_chat_model_end: render text or tool calls
      - on_tool_end: render compact tool results
    """
    from dazi.repl_display import render_dazi_panel

    accumulated_text = ""

    async for event in stream:
        kind = event["event"]

        if kind == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if isinstance(chunk.content, str) and chunk.content:
                spinner.update_label("Thinking...")
                accumulated_text += chunk.content

        elif kind == "on_chat_model_end":
            output = event["data"]["output"]
            has_tool_calls = bool(output.tool_calls)
            # Only show DAZI panel for pure text responses (no tool calls)
            if accumulated_text.strip() and not has_tool_calls:
                console.print()
                render_dazi_panel(accumulated_text, console)
                accumulated_text = ""
            # Print tool calls in compact format
            for tc in output.tool_calls:
                _print_tool_call_compact(tc)
            if has_tool_calls:
                spinner.update_label("Executing tools...")

        elif kind == "on_tool_end":
            tool_output = event["data"]["output"]
            raw = tool_output.content if hasattr(tool_output, "content") else str(tool_output)
            content = str(raw)[:500]
            if len(str(raw)) > 500:
                content += "\n... (truncated)"
            is_error = (
                content.startswith("DENIED")
                or content.startswith("BLOCKED")
                or content.startswith("REQUIRES")
            )
            _print_tool_result_compact(content, is_error=is_error)

    # Handle edge case: text accumulated but stream ended without on_chat_model_end
    if accumulated_text.strip():
        console.print()
        render_dazi_panel(accumulated_text, console)


# ─────────────────────────────────────────────────────────
# GRAPH EXECUTION WRAPPER
# ─────────────────────────────────────────────────────────


async def run_graph_turn(
    *,
    messages: list,
    state: dict,
    session,
    status_label: str = "Thinking...",
    label_suffix: str = "",
) -> dict:
    """Invoke the agent graph with streaming, handle interrupts, and display results.

    A single spinner wraps the entire turn (LLM rounds, tool execution, and
    permission interrupts).  The spinner only pauses briefly for prompt_toolkit
    permission prompts, which bypass Rich's Live display.

    Args:
        messages: The message list to send (already filtered for SystemMessage).
        state: The mutable REPL state dict (mode, etc.).
        status_label: Initial text for the spinner.
        label_suffix: Suffix for panel titles, e.g. " (tick)".

    Returns:
        The result state dict.
    """
    input_state = {
        "messages": messages,
        "mode": state["mode"],
        "allowed_tool_ids": [],
        "consecutive_compact_failures": consecutive_compact_failures,
    }
    thread_id = str(uuid.uuid4())[:8]
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}

    spinner = SpinnerManager(initial_label=status_label)
    spinner.start()

    try:
        # Initial stream
        try:
            stream = app.astream_events(input_state, config, version="v2")
            await _stream_and_display(stream, spinner=spinner)
        except GraphInterrupt:
            pass  # state saved by checkpointer, handle below

        # Handle interrupts — must pause spinner for prompt_toolkit interaction
        graph_state = app.get_state(config)
        while graph_state.next:
            interrupt_value = None
            for task in graph_state.tasks:
                if task.interrupts:
                    interrupt_value = task.interrupts[0].value
                    break
            if not interrupt_value or "ask_tools" not in interrupt_value:
                break
            ask_tools = interrupt_value["ask_tools"]

            # Pause spinner — prompt_toolkit writes directly to terminal
            spinner.stop()
            decisions = await prompt_permission_decisions(ask_tools, session, console)
            spinner.start()

            try:
                stream = app.astream_events(Command(resume=decisions), config, version="v2")
                await _stream_and_display(stream, spinner=spinner)
            except GraphInterrupt:
                pass
            graph_state = app.get_state(config)
    finally:
        spinner.stop()

    # Get final state (spinner stopped, safe to print)
    result = app.get_state(config).values

    # Update REPL state
    state["messages"] = result["messages"]

    # Background task notifications
    completed = background_manager.collect_completed()
    if completed:
        notification_msgs = display_background_notifications(completed)
        if notification_msgs:
            state["messages"].extend(notification_msgs)

    return result


# ─────────────────────────────────────────────────────────
# BACKGROUND TASK NOTIFICATIONS
# ─────────────────────────────────────────────────────────


def display_background_notifications(
    completed_tasks: list[BackgroundTask],
) -> list[HumanMessage]:
    """Display Rich panels for completed background tasks.

    Returns HumanMessage(s) to inject into conversation history
    so the LLM is aware of the completion.

    """
    from dazi.theme import BORDER as _BORDER

    notification_msgs = []

    for task in completed_tasks:
        # Build notification display
        status_icon = {
            BackgroundTaskStatus.COMPLETED: "[green]✓[/green]",
            BackgroundTaskStatus.FAILED: "[red]✗[/red]",
            BackgroundTaskStatus.KILLED: "[bold red]⊘[/bold red]",
        }.get(task.status, "?")

        output = background_manager.get_output_tail(task.id, lines=10)

        lines = [
            f"{status_icon} Background task [bold]{task.id}[/bold] {task.status.value}",
            f"  Command: {task.command}",
        ]
        if task.exit_code is not None:
            lines.append(f"  Exit code: {task.exit_code}")
        if task.duration_seconds:
            lines.append(f"  Duration: {task.duration_seconds:.1f}s")
        if task.error:
            lines.append(f"  Error: {task.error}")
        if output:
            lines.append("\n  [dim]Output (last 10 lines):[/dim]")
            for out_line in output.splitlines()[-10:]:
                lines.append(f"  [dim]{out_line}[/dim]")

        border_style = {
            BackgroundTaskStatus.COMPLETED: _BORDER["success"],
            BackgroundTaskStatus.FAILED: _BORDER["error"],
            BackgroundTaskStatus.KILLED: "bold red",
        }.get(task.status, _BORDER["primary"])

        console.print(
            Panel(
                "\n".join(lines),
                title="Background Task Notification",
                border_style=border_style,
            )
        )

        # Create notification message for LLM context
        notification_content = (
            f"[Background task notification] Task {task.id} ({task.status.value}). "
            f"Command: {task.command}. "
        )
        if task.exit_code is not None:
            notification_content += f"Exit code: {task.exit_code}. "
        if output:
            notification_content += f"Output tail:\n{output}"
        if task.error:
            notification_content += f"Error: {task.error}"

        notification_msgs.append(HumanMessage(content=notification_content))

    return notification_msgs
