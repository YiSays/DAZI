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
}

# prompt_toolkit FormattedText styles — ("style", "text") tuples
PROMPT: dict[str, str] = {
    "mode_execute": "bold fg:green",
    "mode_plan": "bold fg:yellow",
    "primary": "fg:cyan",
    "dim": "fg:dim",
    "separator": "fg:dim",
    "token_ok": "fg:green",
    "token_warning": "fg:yellow",
    "token_compact": "fg:red",
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
}
