"""Tests for dazi.permissions — rule matching, parsing, check_permission."""

import pytest

from dazi.permissions import (
    MODE_DEFAULTS,
    SOURCE_PRIORITY,
    PermissionBehavior,
    PermissionMode,
    PermissionResult,
    PermissionRule,
    _shell_command_matches,
    check_permission,
    derive_permission_pattern,
    prompt_permission_decisions,
    parse_rule,
    parse_rules,
)


# ─────────────────────────────────────────────────────────
# ENUM VALUES
# ─────────────────────────────────────────────────────────


class TestPermissionEnums:
    def test_permission_mode_values(self):
        assert PermissionMode.DEFAULT.value == "default"
        assert PermissionMode.PLAN.value == "plan"
        assert PermissionMode.ACCEPT_EDITS.value == "acceptEdits"
        assert PermissionMode.BYPASS.value == "bypassPermissions"

    def test_permission_behavior_values(self):
        assert PermissionBehavior.ALLOW.value == "allow"
        assert PermissionBehavior.DENY.value == "deny"
        assert PermissionBehavior.ASK.value == "ask"


# ─────────────────────────────────────────────────────────
# PermissionRule.matches_tool
# ─────────────────────────────────────────────────────────


class TestRuleMatchesTool:
    def test_exact_match(self):
        rule = PermissionRule(behavior=PermissionBehavior.ALLOW, tool_name="file_reader")
        assert rule.matches_tool("file_reader") is True

    def test_exact_no_match(self):
        rule = PermissionRule(behavior=PermissionBehavior.ALLOW, tool_name="file_reader")
        assert rule.matches_tool("file_writer") is False

    def test_wildcard_star_only(self):
        rule = PermissionRule(behavior=PermissionBehavior.ALLOW, tool_name="*")
        assert rule.matches_tool("anything") is True

    def test_wildcard_prefix(self):
        rule = PermissionRule(behavior=PermissionBehavior.ALLOW, tool_name="git *")
        assert rule.matches_tool("git push") is True
        assert rule.matches_tool("git commit") is True
        assert rule.matches_tool("git") is True

    def test_wildcard_no_match(self):
        rule = PermissionRule(behavior=PermissionBehavior.ALLOW, tool_name="git *")
        assert rule.matches_tool("npm install") is False

    def test_prefix_match(self):
        rule = PermissionRule(behavior=PermissionBehavior.ALLOW, tool_name="npm:")
        assert rule.matches_tool("npm:install") is True
        assert rule.matches_tool("npm:run build") is True
        assert rule.matches_tool("npm") is True

    def test_prefix_no_match(self):
        rule = PermissionRule(behavior=PermissionBehavior.ALLOW, tool_name="npm:")
        assert rule.matches_tool("yarn install") is False

    def test_none_tool_name_matches_all(self):
        rule = PermissionRule(behavior=PermissionBehavior.ALLOW, tool_name=None)
        assert rule.matches_tool("anything") is True
        assert rule.matches_tool("file_reader") is True


# ─────────────────────────────────────────────────────────
# PermissionRule.matches_args
# ─────────────────────────────────────────────────────────


class TestRuleMatchesArgs:
    def test_no_pattern_matches_all(self):
        rule = PermissionRule(behavior=PermissionBehavior.ALLOW, tool_name="file_reader")
        assert rule.matches_args({"file_path": "/any/path"}) is True

    def test_file_path_glob_match(self):
        rule = PermissionRule(
            behavior=PermissionBehavior.ALLOW,
            tool_name="file_reader",
            pattern="/tmp/*",
        )
        assert rule.matches_args({"file_path": "/tmp/test.py"}) is True

    def test_file_path_glob_no_match(self):
        rule = PermissionRule(
            behavior=PermissionBehavior.DENY,
            tool_name="file_writer",
            pattern="/etc/*",
        )
        assert rule.matches_args({"file_path": "/tmp/test.py"}) is False

    def test_command_pattern_match(self):
        rule = PermissionRule(
            behavior=PermissionBehavior.ALLOW,
            tool_name="shell_exec",
            pattern="git *",
        )
        assert rule.matches_args({"command": "git push origin main"}) is True

    def test_command_pattern_no_match(self):
        rule = PermissionRule(
            behavior=PermissionBehavior.DENY,
            tool_name="shell_exec",
            pattern="rm *",
        )
        assert rule.matches_args({"command": "ls -la"}) is False

    def test_empty_args_matches(self):
        rule = PermissionRule(
            behavior=PermissionBehavior.ALLOW,
            tool_name="tool",
            pattern="/tmp/*",
        )
        assert rule.matches_args({}) is True


# ─────────────────────────────────────────────────────────
# _shell_command_matches
# ─────────────────────────────────────────────────────────


class TestShellCommandMatches:
    def test_exact_match(self):
        assert _shell_command_matches("git push", "git push") is True

    def test_wildcard_star(self):
        assert _shell_command_matches("anything here", "*") is True

    def test_wildcard_prefix(self):
        assert _shell_command_matches("git push origin", "git *") is True

    def test_prefix_colon(self):
        assert _shell_command_matches("npm install", "npm:") is True

    def test_no_match(self):
        assert _shell_command_matches("ls -la", "git push") is False

    def test_empty_command(self):
        assert _shell_command_matches("", "git *") is False

    # ── Pipeline awareness ──

    def test_pipeline_pipe_matches_subcmd(self):
        assert _shell_command_matches("cat file | rm /tmp/old", "rm /tmp/*") is True

    def test_pipeline_chain_matches_subcmd(self):
        assert _shell_command_matches("ls && rm /tmp/old", "rm *") is True

    def test_pipeline_or_chain(self):
        assert _shell_command_matches("true || rm /tmp/old", "rm *") is True

    def test_pipeline_semicolon(self):
        assert _shell_command_matches("echo hi; rm /tmp/old", "rm *") is True

    def test_pipeline_no_match(self):
        assert _shell_command_matches("cat file | grep foo", "rm *") is False

    def test_redirect_not_split(self):
        # Redirect operators are NOT pipeline splits — echo matches as the command
        assert _shell_command_matches("echo hi > /tmp/out", "echo *") is True

    def test_redirect_append_not_split(self):
        assert _shell_command_matches("echo hi >> /tmp/out", "echo *") is True

    def test_pipeline_first_subcmd_match(self):
        assert _shell_command_matches("rm /tmp/old | cat file", "rm *") is True

    def test_pipeline_multi_pipe(self):
        assert _shell_command_matches(
            "cat file | grep foo | rm /tmp/old", "rm /tmp/*"
        ) is True


# ─────────────────────────────────────────────────────────
# parse_rule / parse_rules
# ─────────────────────────────────────────────────────────


class TestParseRule:
    def test_basic_allow(self):
        rule = parse_rule("allow file_reader")
        assert rule.behavior == PermissionBehavior.ALLOW
        assert rule.tool_name == "file_reader"
        assert rule.pattern is None

    def test_with_pattern(self):
        rule = parse_rule("deny file_writer /etc/*")
        assert rule.behavior == PermissionBehavior.DENY
        assert rule.tool_name == "file_writer"
        assert rule.pattern == "/etc/*"

    def test_with_shell_pattern(self):
        rule = parse_rule("allow shell_exec git *")
        assert rule.behavior == PermissionBehavior.ALLOW
        assert rule.tool_name == "shell_exec"
        assert rule.pattern == "git *"

    def test_source(self):
        rule = parse_rule("allow file_reader", source="settings")
        assert rule.source == "settings"

    def test_invalid_format_too_few(self):
        with pytest.raises(ValueError, match="Invalid rule format"):
            parse_rule("allow")

    def test_invalid_behavior(self):
        with pytest.raises(ValueError, match="Invalid behavior"):
            parse_rule("maybe file_reader")

    def test_parse_rules_multiple(self):
        rules = parse_rules(["allow file_reader", "deny shell_exec rm *"], "cli")
        assert len(rules) == 2
        assert rules[0].behavior == PermissionBehavior.ALLOW
        assert rules[1].behavior == PermissionBehavior.DENY


# ─────────────────────────────────────────────────────────
# check_permission
# ─────────────────────────────────────────────────────────


class TestCheckPermission:
    def test_bypass_mode_allows_all(self):
        result = check_permission(
            "shell_exec",
            {"command": "rm -rf /"},
            [],
            mode=PermissionMode.BYPASS,
            tool_safety="destructive",
        )
        assert result.behavior == PermissionBehavior.ALLOW

    def test_plan_mode_blocks_write(self):
        result = check_permission(
            "file_writer",
            {"file_path": "/tmp/test.py"},
            [],
            mode=PermissionMode.PLAN,
            tool_safety="write",
        )
        assert result.behavior == PermissionBehavior.DENY

    def test_plan_mode_blocks_destructive(self):
        result = check_permission(
            "shell_exec",
            {"command": "rm -rf /"},
            [],
            mode=PermissionMode.PLAN,
            tool_safety="destructive",
        )
        assert result.behavior == PermissionBehavior.DENY

    def test_plan_mode_allows_safe(self):
        result = check_permission(
            "file_reader",
            {"file_path": "/tmp/test.py"},
            [],
            mode=PermissionMode.PLAN,
            tool_safety="safe",
        )
        assert result.behavior == PermissionBehavior.ALLOW

    def test_matched_allow_rule(self):
        rules = [
            PermissionRule(behavior=PermissionBehavior.ALLOW, tool_name="file_reader"),
        ]
        result = check_permission("file_reader", {}, rules)
        assert result.behavior == PermissionBehavior.ALLOW
        assert result.matched_rule is not None

    def test_matched_deny_rule(self):
        rules = [
            PermissionRule(behavior=PermissionBehavior.DENY, tool_name="shell_exec"),
        ]
        result = check_permission("shell_exec", {"command": "rm -rf /"}, rules)
        assert result.behavior == PermissionBehavior.DENY

    def test_matched_ask_rule(self):
        rules = [
            PermissionRule(behavior=PermissionBehavior.ASK, tool_name="file_writer"),
        ]
        result = check_permission("file_writer", {"file_path": "/tmp/test.py"}, rules)
        assert result.behavior == PermissionBehavior.ASK

    def test_cli_source_wins_over_settings(self):
        rules = [
            PermissionRule(
                behavior=PermissionBehavior.DENY,
                tool_name="file_reader",
                source="settings",
            ),
            PermissionRule(
                behavior=PermissionBehavior.ALLOW,
                tool_name="file_reader",
                source="cli",
            ),
        ]
        result = check_permission("file_reader", {}, rules)
        assert result.behavior == PermissionBehavior.ALLOW
        assert result.matched_rule.source == "cli"

    def test_deny_wins_same_source(self):
        rules = [
            PermissionRule(
                behavior=PermissionBehavior.ALLOW,
                tool_name="shell_exec",
                source="cli",
            ),
            PermissionRule(
                behavior=PermissionBehavior.DENY,
                tool_name="shell_exec",
                source="cli",
            ),
        ]
        result = check_permission("shell_exec", {}, rules)
        assert result.behavior == PermissionBehavior.DENY

    def test_no_rules_default_mode_safe(self):
        result = check_permission("file_reader", {}, [], tool_safety="safe")
        assert result.behavior == PermissionBehavior.ALLOW

    def test_no_rules_default_mode_write(self):
        result = check_permission("file_writer", {}, [], tool_safety="write")
        assert result.behavior == PermissionBehavior.ASK

    def test_no_rules_default_mode_destructive(self):
        result = check_permission("shell_exec", {}, [], tool_safety="destructive")
        assert result.behavior == PermissionBehavior.ASK


# ─────────────────────────────────────────────────────────
# derive_permission_pattern
# ─────────────────────────────────────────────────────────


class TestDerivePermissionPattern:
    # ── File path (unchanged) ──

    def test_file_path(self):
        pattern = derive_permission_pattern(
            "file_writer", {"file_path": "/tmp/test.py"}
        )
        assert pattern == "/tmp/*"

    def test_nested_file_path(self):
        pattern = derive_permission_pattern(
            "file_writer", {"file_path": "/home/user/src/main.py"}
        )
        assert pattern == "/home/user/src/*"

    # ── Subcommand awareness ──

    def test_git_subcommand(self):
        pattern = derive_permission_pattern(
            "shell_exec", {"command": "git push origin main"}
        )
        assert pattern == "git push *"

    def test_npm_subcommand(self):
        pattern = derive_permission_pattern(
            "shell_exec", {"command": "npm install lodash"}
        )
        assert pattern == "npm install *"

    def test_docker_subcommand(self):
        pattern = derive_permission_pattern(
            "shell_exec", {"command": "docker build -t app ."}
        )
        assert pattern == "docker build *"

    def test_non_subcommand_command(self):
        pattern = derive_permission_pattern(
            "shell_exec", {"command": "echo hello world"}
        )
        assert pattern == "echo *"

    # ── Path argument awareness ──

    def test_find_with_path(self):
        pattern = derive_permission_pattern(
            "shell_exec", {"command": "find /tmp -name '*.py'"}
        )
        assert pattern == "find /tmp/*"

    def test_rm_with_path(self):
        pattern = derive_permission_pattern(
            "shell_exec", {"command": "rm -rf /var/log/app.log"}
        )
        assert pattern == "rm /var/log/*"

    def test_ls_with_relative_path(self):
        pattern = derive_permission_pattern(
            "shell_exec", {"command": "ls ./src/components/"}
        )
        assert pattern == "ls ./src/components/*"

    # ── URL argument awareness ──

    def test_curl_with_url(self):
        pattern = derive_permission_pattern(
            "shell_exec", {"command": "curl -s https://api.example.com/v1/users"}
        )
        assert pattern == "curl https://api.example.com/*"

    def test_wget_with_url(self):
        pattern = derive_permission_pattern(
            "shell_exec", {"command": "wget https://releases.example.com/v2/app.tar.gz"}
        )
        assert pattern == "wget https://releases.example.com/*"

    # ── Pipeline awareness ──

    def test_pipeline_picks_most_specific(self):
        # rm /tmp/old/* is more specific than cat *
        pattern = derive_permission_pattern(
            "shell_exec", {"command": "cat file | rm /tmp/old"}
        )
        assert pattern == "rm /tmp/old/*"

    def test_pipeline_with_chain(self):
        pattern = derive_permission_pattern(
            "shell_exec", {"command": "echo ok && git push origin main"}
        )
        # "git push *" (10 chars) is more specific than "echo *" (6 chars)
        assert pattern == "git push *"

    # ── Edge cases ──

    def test_no_match(self):
        pattern = derive_permission_pattern("calculator", {"expression": "2+2"})
        assert pattern is None

    def test_empty_command(self):
        pattern = derive_permission_pattern("shell_exec", {"command": ""})
        assert pattern is None

    def test_command_flags_only(self):
        # No non-flag args — fallback to cmd + wildcard
        pattern = derive_permission_pattern(
            "shell_exec", {"command": "ls -la"}
        )
        assert pattern == "ls *"

    def test_find_with_current_dir(self):
        # find . should not become "find *" — it should be "find ./*"
        pattern = derive_permission_pattern(
            "shell_exec", {"command": "find . -name '*.py'"}
        )
        assert pattern == "find ./*"


# ─────────────────────────────────────────────────────────
# prompt_permission_decisions — "skip" with user message
# ─────────────────────────────────────────────────────────


class TestPromptPermissionDecisions:
    @pytest.mark.asyncio
    async def test_allow(self):
        from unittest.mock import AsyncMock, MagicMock

        session = AsyncMock()
        session.prompt_async = AsyncMock(return_value="y")
        console = MagicMock()
        decisions = await prompt_permission_decisions(
            [{"tool_call_id": "tc1", "tool_name": "shell_exec",
              "tool_args": {"command": "ls"}, "reason": "test"}],
            session, console,
        )
        assert decisions["tc1"]["action"] == "allow"

    @pytest.mark.asyncio
    async def test_deny_empty(self):
        from unittest.mock import AsyncMock, MagicMock

        session = AsyncMock()
        session.prompt_async = AsyncMock(return_value="")
        console = MagicMock()
        decisions = await prompt_permission_decisions(
            [{"tool_call_id": "tc1", "tool_name": "shell_exec",
              "tool_args": {"command": "ls"}, "reason": "test"}],
            session, console,
        )
        assert decisions["tc1"]["action"] == "deny"

    @pytest.mark.asyncio
    async def test_deny_no(self):
        from unittest.mock import AsyncMock, MagicMock

        session = AsyncMock()
        session.prompt_async = AsyncMock(return_value="no")
        console = MagicMock()
        decisions = await prompt_permission_decisions(
            [{"tool_call_id": "tc1", "tool_name": "shell_exec",
              "tool_args": {"command": "ls"}, "reason": "test"}],
            session, console,
        )
        assert decisions["tc1"]["action"] == "deny"

    @pytest.mark.asyncio
    async def test_skip_with_message(self):
        from unittest.mock import AsyncMock, MagicMock

        session = AsyncMock()
        session.prompt_async = AsyncMock(
            return_value="I already created the folder manually"
        )
        console = MagicMock()
        decisions = await prompt_permission_decisions(
            [{"tool_call_id": "tc1", "tool_name": "shell_exec",
              "tool_args": {"command": "mkdir tmp/data"}, "reason": "write"}],
            session, console,
        )
        assert decisions["tc1"]["action"] == "skip"
        assert decisions["tc1"]["message"] == "I already created the folder manually"
