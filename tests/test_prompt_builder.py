"""Tests for dazi/prompt_builder.py — prompt sections and builder."""

from __future__ import annotations

from dazi.prompt_builder import (
    DYNAMIC_BOUNDARY,
    STATIC_SECTIONS,
    PromptSection,
    SystemPromptBuilder,
    build_dazimd_section,
    build_environment_section,
    build_permissions_section,
    build_session_guidance,
    build_skills_section,
)

# ─────────────────────────────────────────────────────────
# PromptSection enum
# ─────────────────────────────────────────────────────────


class TestPromptSection:
    def test_static_sections_exist(self):
        static = [
            PromptSection.INTRO,
            PromptSection.SYSTEM,
            PromptSection.DOING_TASKS,
            PromptSection.ACTIONS,
            PromptSection.USING_TOOLS,
            PromptSection.TONE_AND_STYLE,
            PromptSection.OUTPUT_EFFICIENCY,
        ]
        for s in static:
            assert s in STATIC_SECTIONS

    def test_dynamic_sections_are_defined(self):
        dynamic = [
            PromptSection.SESSION_GUIDANCE,
            PromptSection.SKILLS,
            PromptSection.MEMORY,
            PromptSection.ENVIRONMENT,
            PromptSection.PERMISSIONS,
            PromptSection.DAZI_MD,
        ]
        for s in dynamic:
            assert isinstance(s.value, str)

    def test_total_sections_count(self):
        assert len(PromptSection) == 13


# ─────────────────────────────────────────────────────────
# build_session_guidance
# ─────────────────────────────────────────────────────────


class TestBuildSessionGuidance:
    def test_execute_mode(self):
        result = build_session_guidance(mode="execute")
        assert "EXECUTE" in result
        assert "All tools enabled" in result

    def test_plan_mode(self):
        result = build_session_guidance(mode="plan")
        assert "PLAN" in result
        assert "MUST NOT make any edits" in result
        assert "Phase 1: Understand" in result
        assert "Phase 2: Design" in result
        assert "Phase 3: Clarify" in result
        assert "Phase 4: Write Plan" in result
        assert "Phase 5: Finish" in result
        assert "plan_writer" in result
        assert "file_reader" in result
        assert "/go" in result

    def test_plan_mode_has_existing_plan(self):
        result = build_session_guidance(mode="plan", has_plan=True)
        assert "plan file already exists" in result
        assert "overwrite it" in result

    def test_plan_mode_no_existing_plan(self):
        result = build_session_guidance(mode="plan", has_plan=False)
        assert "No plan file exists yet" in result

    def test_has_plan_execute_mode(self):
        result = build_session_guidance(mode="execute", has_plan=True)
        assert "EXECUTE" in result
        assert "All tools enabled" in result
        assert "plan file exists" in result

    def test_no_plan_execute_mode(self):
        result = build_session_guidance(mode="execute", has_plan=False)
        assert "EXECUTE" in result
        assert "All tools enabled" in result
        assert "plan file" not in result


# ─────────────────────────────────────────────────────────
# build_environment_section
# ─────────────────────────────────────────────────────────


class TestBuildEnvironmentSection:
    def test_contains_cwd(self):
        result = build_environment_section()
        assert "Working directory:" in result

    def test_contains_os(self):
        result = build_environment_section()
        assert "OS:" in result

    def test_contains_python(self):
        result = build_environment_section()
        assert "Python:" in result


# ─────────────────────────────────────────────────────────
# build_permissions_section
# ─────────────────────────────────────────────────────────


class TestBuildPermissionsSection:
    def test_mode_displayed(self):
        result = build_permissions_section(mode="execute")
        assert "Mode: execute" in result

    def test_rule_count(self):
        result = build_permissions_section(mode="default", rule_count=5)
        assert "5" in result

    def test_plan_mode_desc(self):
        result = build_permissions_section(mode="plan")
        assert "Plan mode" in result


# ─────────────────────────────────────────────────────────
# build_dazimd_section
# ─────────────────────────────────────────────────────────


class TestBuildDazimdSection:
    def test_empty_content(self):
        assert build_dazimd_section("") == ""
        assert build_dazimd_section("   ") == ""

    def test_with_content(self):
        result = build_dazimd_section("Always use type hints")
        assert "Always use type hints" in result
        assert "Project Instructions" in result


# ─────────────────────────────────────────────────────────
# build_skills_section
# ─────────────────────────────────────────────────────────


class TestBuildSkillsSection:
    def test_empty_content(self):
        assert build_skills_section("") == ""
        assert build_skills_section("   ") == ""

    def test_with_content(self):
        result = build_skills_section("commit, review, explain")
        assert "commit, review, explain" in result
        assert "Available Skills" in result


# ─────────────────────────────────────────────────────────
# SystemPromptBuilder
# ─────────────────────────────────────────────────────────


class TestSystemPromptBuilder:
    def test_first_build(self):
        builder = SystemPromptBuilder()
        prompt = builder.build(mode="execute")
        assert len(prompt) > 0
        assert builder.build_count == 1
        assert builder.is_cached is True

    def test_caching(self):
        builder = SystemPromptBuilder()
        prompt1 = builder.build(mode="execute")
        prompt2 = builder.build(mode="execute")
        assert prompt1 == prompt2
        assert builder.build_count == 2

    def test_invalidate_on_dazimd_change(self):
        builder = SystemPromptBuilder()
        builder.build(mode="execute")
        assert builder.is_cached is True

        builder.set_dazimd_content("New instruction")
        assert builder.is_cached is False

        prompt2 = builder.build(mode="execute")
        assert "New instruction" in prompt2
        assert builder.is_cached is True

    def test_dynamic_boundary_present(self):
        builder = SystemPromptBuilder()
        prompt = builder.build(mode="execute")
        assert DYNAMIC_BOUNDARY in prompt

    def test_force_rebuild(self):
        builder = SystemPromptBuilder()
        builder.build()
        builder.build(force_rebuild=True)
        # After force_rebuild, cache is rebuilt
        assert builder.is_cached is True
