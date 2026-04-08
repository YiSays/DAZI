"""Tests for dazi.protocols — protocol message factories."""

import uuid
from datetime import datetime, timezone

from dazi.protocols import (
    _new_id,
    _now,
    create_idle_notification,
    create_permission_request,
    create_permission_response,
    create_plan_approval_request,
    create_plan_approval_response,
    create_shutdown_request,
    create_shutdown_response,
    create_text_message,
)
from dazi.team import TEAM_LEAD_NAME


class TestHelpers:
    def test_now_iso_format(self):
        result = _now()
        # Should be parseable as ISO 8601
        datetime.fromisoformat(result)

    def test_now_utc(self):
        result = _now()
        # UTC ISO strings contain +00:00 or Z
        assert "+00:00" in result or "Z" in result or "T" in result

    def test_new_id_uuid_format(self):
        result = _new_id()
        # Should be a valid UUID4
        uuid.UUID(result)

    def test_new_id_unique(self):
        assert _new_id() != _new_id()


class TestTextMessage:
    def test_basic_fields(self):
        msg = create_text_message("alice", "bob", "Hello!")
        assert msg.from_agent == "alice"
        assert msg.to_agent == "bob"
        assert msg.text == "Hello!"
        assert msg.msg_type == "text"
        assert msg.id is not None
        assert msg.timestamp is not None

    def test_summary_provided(self):
        msg = create_text_message("a", "b", "Text", summary="Short")
        assert msg.summary == "Short"

    def test_summary_default_empty(self):
        msg = create_text_message("a", "b", "Text")
        assert msg.summary == ""

    def test_broadcast(self):
        msg = create_text_message("a", "*", "Broadcast!")
        assert msg.to_agent == "*"


class TestShutdownProtocol:
    def test_request_fields(self):
        msg = create_shutdown_request("leader", "worker", "Maintenance")
        assert msg.msg_type == "shutdown_request"
        assert msg.from_agent == "leader"
        assert msg.to_agent == "worker"
        assert msg.metadata["reason"] == "Maintenance"
        assert "request_id" in msg.metadata

    def test_request_no_reason(self):
        msg = create_shutdown_request("leader", "worker")
        assert msg.metadata["reason"] == ""
        assert "Shutdown request" in msg.text

    def test_response_approve(self):
        msg = create_shutdown_response("worker", "leader", "req-123", True)
        assert msg.msg_type == "shutdown_response"
        assert msg.metadata["approve"] is True
        assert msg.metadata["request_id"] == "req-123"
        assert "approved" in msg.text

    def test_response_reject(self):
        msg = create_shutdown_response("worker", "leader", "req-123", False, "Busy")
        assert msg.metadata["approve"] is False
        assert "rejected" in msg.text
        assert msg.metadata["reason"] == "Busy"


class TestPermissionProtocol:
    def test_request_fields(self):
        msg = create_permission_request(
            "worker", "file_writer", {"path": "/tmp/test.py"}
        )
        assert msg.msg_type == "permission_request"
        assert msg.to_agent == TEAM_LEAD_NAME
        assert msg.metadata["tool_name"] == "file_writer"
        assert msg.metadata["tool_args"] == {"path": "/tmp/test.py"}
        assert "request_id" in msg.metadata

    def test_request_with_reason(self):
        msg = create_permission_request(
            "worker", "shell_exec", {"command": "npm test"}, reason="Testing"
        )
        assert msg.metadata["reason"] == "Testing"

    def test_response_approved(self):
        msg = create_permission_response("leader", "worker", "req-1", True)
        assert msg.metadata["approved"] is True
        assert "approved" in msg.text

    def test_response_denied(self):
        msg = create_permission_response("leader", "worker", "req-1", False, "Unsafe")
        assert msg.metadata["approved"] is False
        assert "denied" in msg.text


class TestPlanApprovalProtocol:
    def test_request(self):
        msg = create_plan_approval_request("worker", "leader", "Step 1: Do stuff")
        assert msg.msg_type == "plan_approval_request"
        assert msg.text == "Step 1: Do stuff"
        assert "request_id" in msg.metadata

    def test_response_approve(self):
        msg = create_plan_approval_response("leader", "worker", "req-1", True)
        assert msg.metadata["approve"] is True

    def test_response_with_feedback(self):
        msg = create_plan_approval_response(
            "leader", "worker", "req-1", False, feedback="Add tests"
        )
        assert msg.metadata["approve"] is False
        assert msg.metadata["feedback"] == "Add tests"


class TestIdleNotification:
    def test_defaults(self):
        msg = create_idle_notification("worker")
        assert msg.msg_type == "idle_notification"
        assert msg.to_agent == "*"
        assert msg.metadata["idleReason"] == "no_pending_work"

    def test_with_task_id(self):
        msg = create_idle_notification("worker", completed_task_id="42")
        assert msg.metadata["completedTaskId"] == "42"

    def test_custom_reason(self):
        msg = create_idle_notification("worker", idle_reason="interrupted")
        assert msg.metadata["idleReason"] == "interrupted"

    def test_custom_summary(self):
        msg = create_idle_notification("worker", summary="All done")
        assert msg.summary == "All done"
