"""Tests for dazi/proactive.py — ProactiveState, ProactiveSource, ProactiveManager, format_tick."""

from __future__ import annotations

import re

from dazi.proactive import (
    ProactiveManager,
    ProactiveSource,
    ProactiveState,
    format_tick,
)

# ─────────────────────────────────────────────────────────
# ENUM TESTS
# ─────────────────────────────────────────────────────────


class TestProactiveEnums:
    def test_proactive_source_values(self):
        assert ProactiveSource.COMMAND == "command"
        assert ProactiveSource.ENV == "env"

    def test_proactive_state_values(self):
        assert ProactiveState.INACTIVE == "inactive"
        assert ProactiveState.ACTIVE == "active"
        assert ProactiveState.PAUSED == "paused"


# ─────────────────────────────────────────────────────────
# STATE MACHINE TESTS
# ─────────────────────────────────────────────────────────


class TestProactiveManagerStateMachine:
    def test_initial_state_is_inactive(self):
        pm = ProactiveManager()
        assert pm.state == ProactiveState.INACTIVE

    def test_activate_transitions_to_active(self):
        pm = ProactiveManager()
        pm.activate()
        assert pm.state == ProactiveState.ACTIVE
        assert pm._source == ProactiveSource.COMMAND

    def test_activate_with_env_source(self):
        pm = ProactiveManager()
        pm.activate(source=ProactiveSource.ENV)
        assert pm._source == ProactiveSource.ENV

    def test_pause_transitions_active_to_paused(self):
        pm = ProactiveManager()
        pm.activate()
        pm.pause()
        assert pm.state == ProactiveState.PAUSED

    def test_resume_transitions_paused_to_active(self):
        pm = ProactiveManager()
        pm.activate()
        pm.pause()
        pm.resume()
        assert pm.state == ProactiveState.ACTIVE

    def test_deactivate_transitions_to_inactive(self):
        pm = ProactiveManager()
        pm.activate()
        pm.deactivate()
        assert pm.state == ProactiveState.INACTIVE

    def test_pause_on_inactive_is_noop(self):
        pm = ProactiveManager()
        pm.pause()
        assert pm.state == ProactiveState.INACTIVE

    def test_resume_on_active_is_noop(self):
        pm = ProactiveManager()
        pm.activate()
        pm.resume()  # already active, not paused
        assert pm.state == ProactiveState.ACTIVE

    def test_full_cycle_inactive_active_paused_active_inactive(self):
        pm = ProactiveManager()
        assert pm.state == ProactiveState.INACTIVE
        pm.activate()
        assert pm.state == ProactiveState.ACTIVE
        pm.pause()
        assert pm.state == ProactiveState.PAUSED
        pm.resume()
        assert pm.state == ProactiveState.ACTIVE
        pm.deactivate()
        assert pm.state == ProactiveState.INACTIVE


# ─────────────────────────────────────────────────────────
# should_generate_tick / mark_tick_sent / properties
# ─────────────────────────────────────────────────────────


class TestTickGeneration:
    def test_should_not_generate_tick_when_inactive(self):
        pm = ProactiveManager()
        assert pm.should_generate_tick() is False

    def test_should_generate_tick_when_active(self):
        pm = ProactiveManager()
        pm.activate()
        assert pm.should_generate_tick() is True

    def test_should_not_generate_tick_when_paused(self):
        pm = ProactiveManager()
        pm.activate()
        pm.pause()
        assert pm.should_generate_tick() is False

    def test_mark_tick_sent_clears_first_tick(self):
        pm = ProactiveManager()
        pm.activate()
        assert pm.is_first_tick is True
        pm.mark_tick_sent()
        assert pm.is_first_tick is False

    def test_mark_tick_sent_records_last_tick_time(self):
        pm = ProactiveManager()
        pm.activate()
        assert pm.last_tick_time is None
        pm.mark_tick_sent()
        assert pm.last_tick_time is not None
        # Should match HH:MM:SS format
        assert re.match(r"\d{2}:\d{2}:\d{2}", pm.last_tick_time)

    def test_activation_count_increments(self):
        pm = ProactiveManager()
        assert pm.activation_count == 0
        pm.activate()
        assert pm.activation_count == 1
        pm.deactivate()
        pm.activate()
        assert pm.activation_count == 2

    def test_resume_resets_first_tick_pending(self):
        pm = ProactiveManager()
        pm.activate()
        pm.mark_tick_sent()
        assert pm.is_first_tick is False
        pm.pause()
        pm.resume()
        assert pm.is_first_tick is True


# ─────────────────────────────────────────────────────────
# SUBSCRIBER / RESET
# ─────────────────────────────────────────────────────────


class TestSubscriberAndReset:
    def test_subscriber_receives_state_changes(self):
        transitions = []
        pm = ProactiveManager()
        pm.subscribe(lambda old, new: transitions.append((old, new)))
        pm.activate()
        assert len(transitions) == 1
        assert transitions[0] == (ProactiveState.INACTIVE, ProactiveState.ACTIVE)

    def test_subscriber_exception_is_swallowed(self):
        pm = ProactiveManager()

        def bad_callback(old, new):
            raise RuntimeError("boom")

        pm.subscribe(bad_callback)
        # Should not raise
        pm.activate()
        assert pm.state == ProactiveState.ACTIVE

    def test_reset_clears_all_state(self):
        pm = ProactiveManager()
        pm.activate()
        pm.mark_tick_sent()
        pm.subscribe(lambda old, new: None)
        pm.reset()
        assert pm.state == ProactiveState.INACTIVE
        assert pm.activation_count == 0
        assert pm.is_first_tick is False
        assert pm.last_tick_time is None
        assert pm.source is None


# ─────────────────────────────────────────────────────────
# format_tick
# ─────────────────────────────────────────────────────────


class TestFormatTick:
    def test_format_tick_returns_correct_format(self):
        result = format_tick()
        assert result.startswith("<tick>")
        assert result.endswith("</tick>")
        # Extract the time portion
        time_part = result[len("<tick>") : -len("</tick>")]
        assert re.match(r"\d{2}:\d{2}:\d{2}", time_part)
