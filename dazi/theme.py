"""Centralized color theme for Dazi REPL display.

Single source of truth for all colors used across the codebase.
Three categories:

- RICH: Rich markup tag names (console.print("[cyan]...[/cyan]"))
- PROMPT: prompt_toolkit style strings (FormattedText tuples)
- BORDER: Panel border_style values (Panel(border_style=...))
"""

from __future__ import annotations

# Rich markup colors — used in console.print("[color]...[/color]")
RICH: dict[str, str] = {
    "primary": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "dim": "dim",
    "accent": "magenta",
    "user_title": "bold orange3",
}

# prompt_toolkit FormattedText styles — ("style", "text") tuples
PROMPT: dict[str, str] = {
    "mode_execute": "bold fg:ansigreen",
    "mode_plan": "bold fg:ansiblue",
    "primary": "fg:ansicyan",
    "dim": "noinherit fg:ansigray",
    "separator": "noinherit fg:ansigray",
    "token_ok": "fg:ansigreen",
    "token_warning": "fg:ansiyellow",
    "token_compact": "fg:ansired",
}

# Panel border_style values — Panel(border_style=...)
BORDER: dict[str, str] = {
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red",
    "dim": "dim",
    "brand": "bright_cyan",
    "primary": "blue",
    "user": "orange3",
}
