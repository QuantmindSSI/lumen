"""Audit logging helpers for the Lumen memory palace."""

import json
import sqlite3


def log_audit_event(
    conn: sqlite3.Connection,
    event_type: str,
    actor: str | None,
    resource_type: str | None,
    resource_id: int | None,
    action: str | None,
    metadata_json: str | None,
    client_ip: str | None,
    request_id: str | None,
) -> None:
    """Insert a single row into the audit_log table."""
    conn.execute(
        """
        INSERT INTO audit_log (
            event_type, actor, resource_type, resource_id,
            action, metadata_json, client_ip, request_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (event_type, actor, resource_type, resource_id, action, metadata_json, client_ip, request_id),
    )
    conn.commit()


def audit_from_request(
    conn: sqlite3.Connection,
    event_type: str,
    action: str,
    resource_id: int | None,
    request,
    metadata_dict: dict | None = None,
) -> None:
    """Convenience wrapper that extracts client_ip / request_id from a request
    object (e.g. FastAPI Request) and forwards the audit event.
    """
    client_ip = None
    request_id = None
    if hasattr(request, "client") and request.client:
        client_ip = request.client.host
    if hasattr(request, "headers"):
        request_id = request.headers.get("x-request-id")
    metadata_json = json.dumps(metadata_dict) if metadata_dict else None
    log_audit_event(
        conn=conn,
        event_type=event_type,
        actor=None,
        resource_type=None,
        resource_id=resource_id,
        action=action,
        metadata_json=metadata_json,
        client_ip=client_ip,
        request_id=request_id,
    )
