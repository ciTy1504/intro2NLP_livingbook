"""Email delivery.

Adapted from `EngChi/BE/src/services/mailer.js`: same credential model (Gmail +
App Password via ``GMAIL_USER`` / ``GMAIL_PASS``), same inline-CSS table layout so the
message renders in clients that strip stylesheets, and the same operational discipline
from `reengagement.job.js` — per-recipient error isolation so one bad address cannot
abort the batch, plus a persisted send record so a retry never double-sends.

Reimplemented in Python `smtplib` rather than ported, since this system is Python.
No credentials were copied from that project.
"""

from __future__ import annotations

import asyncio
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr, make_msgid
from typing import Any

from ..config import get_config, get_secrets
from ..obs import get_logger, new_id
from ..state.store import get_store, utcnow
from .registry import Capability, ToolError, ToolUnavailable, tool


@tool("send_email", [Capability.EMAIL],
      description="Send an HTML notification email and record the send.")
async def send_email(
    *,
    subject: str,
    html: str,
    text: str = "",
    recipients: list[str] | None = None,
    pipeline_id: str | None = None,
    dedupe_key: str | None = None,
) -> dict[str, Any]:
    cfg = get_config()
    secrets = get_secrets()
    log = get_logger()
    store = get_store()

    if not cfg.get("email.enabled", True):
        return {"sent": False, "reason": "email.enabled is false"}

    to = recipients or cfg.get("email.recipients") or secrets.email_recipients()
    if not to:
        raise ToolUnavailable(
            "No email recipients configured. Set EMAIL_RECIPIENTS in secrets/.env "
            "or email.recipients in config/config.yaml."
        )

    user = secrets.get("GMAIL_USER")
    password = secrets.get("GMAIL_PASS")
    if not user or not password:
        raise ToolUnavailable(
            "GMAIL_USER / GMAIL_PASS are not set; cannot send email. "
            "Use a Google App Password, not the account password."
        )

    # A pipeline that is retried must not re-notify. The persisted record is the
    # guard, mirroring the reengagementStage counter in the EngChi job.
    if dedupe_key:
        already = store.query_one(
            "SELECT id, sent_at FROM emails_sent WHERE message_id = ?", (dedupe_key,))
        if already:
            return {"sent": False, "reason": "already sent",
                    "sent_at": already["sent_at"], "deduped": True}

    host = cfg.get("email.smtp_host", "smtp.gmail.com")
    port = int(cfg.get("email.smtp_port", 587))
    from_name = cfg.get("email.from_name", "Living Book Agent")

    results: list[dict[str, Any]] = []

    def _send_all() -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        context = ssl.create_default_context()
        try:
            with smtplib.SMTP(host, port, timeout=60) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(user, password)
                for address in to:
                    # Per-recipient isolation: one rejected address must not abort
                    # the rest of the batch.
                    try:
                        msg = EmailMessage()
                        msg["Subject"] = subject
                        msg["From"] = formataddr((from_name, user))
                        msg["To"] = address
                        mid = make_msgid(domain="livingbook.local")
                        msg["Message-ID"] = mid
                        msg.set_content(text or _html_to_text(html))
                        msg.add_alternative(html, subtype="html")
                        server.send_message(msg)
                        out.append({"recipient": address, "ok": True, "message_id": mid})
                    except Exception as exc:
                        out.append({"recipient": address, "ok": False,
                                    "error": f"{type(exc).__name__}: {exc}"})
        except smtplib.SMTPAuthenticationError as exc:
            raise ToolUnavailable(
                f"SMTP authentication failed ({exc.smtp_code}). Gmail requires an "
                "App Password when 2FA is enabled."
            ) from exc
        except (smtplib.SMTPException, OSError) as exc:
            raise ToolUnavailable(f"SMTP connection failed: {type(exc).__name__}: {exc}") from exc
        return out

    results = await asyncio.to_thread(_send_all)

    sent_ok = [r for r in results if r["ok"]]
    for r in sent_ok:
        store.execute(
            "INSERT INTO emails_sent (id, pipeline_id, recipient, subject, sent_at, "
            "message_id, status) VALUES (?,?,?,?,?,?,?)",
            (new_id("mail"), pipeline_id, r["recipient"], subject, utcnow(),
             dedupe_key or r["message_id"], "sent"),
        )

    failed = [r for r in results if not r["ok"]]
    log.info(
        f"email '{subject[:60]}' -> {len(sent_ok)}/{len(to)} delivered"
        + (f", {len(failed)} failed" if failed else ""),
        tool="send_email", status="ok" if sent_ok else "error",
    )
    if not sent_ok:
        raise ToolError(f"email delivery failed for every recipient: {failed[:2]}")

    return {"sent": True, "delivered": len(sent_ok), "recipients": [r["recipient"] for r in sent_ok],
            "failed": failed}


@tool("email_status", [Capability.EMAIL, Capability.FS_READ],
      description="Report whether email is configured and what has been sent.")
async def email_status() -> dict[str, Any]:
    cfg = get_config()
    secrets = get_secrets()
    store = get_store()
    recent = store.query(
        "SELECT recipient, subject, sent_at FROM emails_sent ORDER BY sent_at DESC LIMIT 10")
    return {
        "enabled": bool(cfg.get("email.enabled", True)),
        "configured": bool(secrets.get("GMAIL_USER") and secrets.get("GMAIL_PASS")),
        "recipients": cfg.get("email.recipients") or secrets.email_recipients(),
        "smtp": f"{cfg.get('email.smtp_host')}:{cfg.get('email.smtp_port')}",
        "sent_total": int(store.scalar("SELECT COUNT(*) FROM emails_sent") or 0),
        "recent": [dict(r) for r in recent],
    }


def _html_to_text(html: str) -> str:
    """Plain-text alternative, so the message is readable without HTML."""
    import re
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</(p|div|tr|h[1-6]|li)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    import html as html_mod
    text = html_mod.unescape(text)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.splitlines()]
    return "\n".join(ln for ln in lines if ln)
