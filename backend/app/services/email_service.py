import logging
import re

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

RESEND_API_URL = "https://api.resend.com/emails"

# Lightweight User-Agent parsing mirroring the Node reference's
# ua-parser-js "request details" block. No external dependency needed.
_UA_BROWSER = re.compile(
    r"(Edg|OPR|Chrome|Firefox|Safari|Version)"
    r"[/ ]([\d.]+)",
    re.IGNORECASE,
)
_UA_OS = re.compile(r"\((.*?)\)")
_BROWSER_MAP = {
    "edg": ("Edge", "edg"),
    "opr": ("Opera", "opr"),
    "chrome": ("Chrome", "chrome"),
    "firefox": ("Firefox", "firefox"),
}


def _parse_device(user_agent: str | None) -> dict:
    if not user_agent:
        return {"browser": "Unknown", "os": "Unknown", "device": "Unknown"}

    browser = "Unknown"
    browser_version = ""
    m = _UA_BROWSER.search(user_agent)
    if m:
        name = m.group(1).lower()
        browser = _BROWSER_MAP.get(name, (m.group(1), m.group(1)))[0]
        browser_version = m.group(2)
    elif "Edg/" in user_agent:
        browser = "Edge"
        ver = re.search(r"Edg/([\d.]+)", user_agent)
        browser_version = ver.group(1) if ver else ""
    elif "OPR/" in user_agent:
        browser = "Opera"
        ver = re.search(r"OPR/([\d.]+)", user_agent)
        browser_version = ver.group(1) if ver else ""
    elif "Safari" in user_agent:
        browser = "Safari"
        ver = re.search(r"Version/([\d.]+)", user_agent)
        browser_version = ver.group(1) if ver else ""

    os_name = "Unknown"
    os_match = _UA_OS.search(user_agent)
    if os_match:
        fragment = os_match.group(1)
        os_name = fragment.split(";")[0].strip()

    mobile = bool(re.search(r"Mobile|Android", user_agent, re.IGNORECASE))
    tablet = bool(re.search(r"iPad|Tablet", user_agent, re.IGNORECASE))
    if mobile:
        device = "Mobile"
    elif tablet:
        device = "Tablet"
    else:
        device = "Desktop/Laptop"

    browser_label = (
        f"{browser} {browser_version}".strip()
        if browser_version
        else browser
    )
    return {"browser": browser_label, "os": os_name, "device": device}


async def _get_location_from_ip(ip: str | None) -> str:
    if not ip or ip in ("127.0.0.1", "::1", "localhost"):
        return "Local Network"
    clean = ip.replace("::ffff:", "")
    try:
        import json

        async with httpx.AsyncClient(timeout=5) as client:
            res = await client.get(
                f"http://ip-api.com/json/{clean}",
                params={"fields": "city,regionName,country"},
            )
            if res.status_code == 200 and res.text:
                data = json.loads(res.text)
                if data.get("city"):
                    return f"{data['city']}, {data['regionName']}, {data['country']}"
    except Exception:  # pragma: no cover - best effort
        pass
    return "Unknown Location"


def _format_date(date) -> str:
    return date.strftime("%b %d, %Y, %I:%M %p")


def _request_details_html(context: dict) -> str:
    rows = [
        ("Requested", context.get("requestedAt", "")),
        ("Browser", context.get("browser", "")),
        ("OS", context.get("os", "")),
        ("Device", context.get("device", "")),
        ("Location", context.get("location", "")),
        ("IP", context.get("ipAddress", "")),
    ]
    cells = ""
    for label, value in rows:
        cells += (
            "<tr>"
            f'<td style="padding:5px 0; color:#60564c; width:110px;">{label}</td>'
            f'<td style="padding:5px 0;">{value}</td>'
            "</tr>"
        )
    return cells


def _envelope(subject: str, body: str) -> str:
    return (
        "<!DOCTYPE html>"
        '<html><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
        "</head>"
        '<body style="margin:0; padding:0; background-color:#f9f9f8; '
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,"
        "'Helvetica Neue',sans-serif;\">"
        '<table width="100%" cellpadding="0" cellspacing="0" '
        'style="background-color:#f9f9f8; padding:48px 20px;">'
        "<tr><td align=\"center\">"
        '<table width="520" cellpadding="0" cellspacing="0" '
        'style="background-color:#ffffff; border-radius:12px; overflow:hidden; '
        'border:1px solid #e8e5e0;">'
        # Header
        '<tr><td style="background-color:#0b0805; padding:24px 32px;">'
        '<span style="color:#f9f9f8; font-size:18px; font-weight:700; '
        'letter-spacing:-0.5px;">Mistral RAG Studio</span>'
        "</td></tr>"
        f"{body}"
        # Footer
        '<tr><td style="background-color:#f4f4f3; padding:18px 32px; '
        'text-align:center; border-top:1px solid #e8e5e0;">'
        '<p style="margin:0; color:#60564c; font-size:11px;">'
        "&copy; RAG Document Workspace. All rights reserved.</p>"
        "</td></tr>"
        "</table></td></tr></table></body></html>"
    )


def _build_context(user_agent: str | None, ip: str | None, is_resend: bool = False) -> dict:
    device = _parse_device(user_agent)
    from datetime import datetime

    return {
        "requestedAt": _format_date(datetime.now()),
        "browser": device["browser"],
        "os": device["os"],
        "device": device["device"],
        "location": "Unknown",  # filled lazily in senders via _get_location_from_ip
        "ipAddress": (ip or "Unknown").replace("::ffff:", ""),
        "isResend": is_resend,
    }


class EmailService:
    """Thin Resend client (plain httpx, no SDK).

    When RESEND_API_KEY is empty the service logs the email instead of
    sending, so local development does not fail on registration."""

    def _send(self, to_email: str, subject: str, html: str) -> bool:
        if not settings.resend_api_key:
            logger.warning(
                "RESEND_API_KEY not configured; skipping email to %s "
                "(was going to send subject: %s)",
                to_email,
                subject,
            )
            return False

        if not settings.resend_from_email:
            logger.error(
                "RESEND_FROM_EMAIL not configured; cannot send to %s",
                to_email,
            )
            return False

        try:
            response = httpx.post(
                RESEND_API_URL,
                headers={
                    "Authorization": f"Bearer {settings.resend_api_key}",
                    "Content-Type": "application/json",
                    "X-HTTP-Timeout": "40",
                },
                json={
                    "from": settings.resend_from_email,
                    "to": [to_email],
                    "subject": subject,
                    "html": html,
                },
                timeout=40,
            )
        except httpx.HTTPError as exc:  # pragma: no cover - network
            logger.error("Resend network error to %s: %s", to_email, exc)
            return False

        if response.status_code >= 300:
            logger.error(
                "Resend rejected email to %s: %s %s",
                to_email,
                response.status_code,
                response.text[:300],
            )
            return False

        logger.info("Email sent to %s (subject=%s)", to_email, subject)
        return True

    def send_verification_email(
        self,
        to_email: str,
        token: str,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> bool:
        link = (
            f"{settings.frontend_url.rstrip('/')}"
            f"/verify?token={token}"
        )
        subject = "Verify your email address"

        body = (
            '<tr><td style="padding:32px 32px 24px;">'
            '<h1 style="margin:0 0 6px; color:#0b0805; font-size:20px; '
            'font-weight:700;">Verify Your Email</h1>'
            '<p style="margin:0 0 20px; color:#60564c; font-size:14px; '
            'line-height:1.6;">Click the link below to activate your account '
            "for the RAG document workspace:</p>"
            '<table width="100%" cellpadding="0" cellspacing="0">'
            "<tr><td align=\"center\">"
            f'<a href="{link}" style="display:inline-block; background-color:#0b0805; '
            'color:#f9f9f8; font-size:14px; font-weight:600; text-decoration:none; '
            'padding:12px 32px; border-radius:8px;">Verify Email</a>'
            "</td></tr></table>"
            '<p style="margin:14px 0 0; color:#60564c; font-size:12px; '
            'text-align:center;">This link expires in '
            f"{settings.verification_token_expire_hours} hours.</p>"
            "</td></tr>"
            '<tr><td style="padding:0 32px;"><hr style="border:none; '
            'border-top:1px solid #e8e5e0; margin:0;"></td></tr>'
            '<tr><td style="padding:20px 32px;">'
            '<p style="margin:0 0 10px; color:#0b0805; font-size:12px; '
            'font-weight:600; text-transform:uppercase;">Request Details</p>'
            '<table width="100%" cellpadding="0" cellspacing="0" '
            f'style="font-size:12px; color:#0b0805;">'
            f"{_request_details_html(_build_context(user_agent, ip_address))}"
            "</table></td></tr>"
        )

        return self._send(to_email, subject, _envelope(subject, body))

    def send_password_reset_email(
        self,
        to_email: str,
        reset_link: str,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> bool:
        subject = "Reset Your Password"
        body = (
            '<tr><td style="padding:32px 32px 24px;">'
            '<h1 style="margin:0 0 6px; color:#0b0805; font-size:20px; '
            'font-weight:700;">Reset Your Password</h1>'
            '<p style="margin:0 0 20px; color:#60564c; font-size:14px; '
            'line-height:1.6;">We received a request to reset your password. '
            "Click below to choose a new one:</p>"
            '<table width="100%" cellpadding="0" cellspacing="0">'
            "<tr><td align=\"center\">"
            f'<a href="{reset_link}" style="display:inline-block; background-color:#0b0805; '
            'color:#f9f9f8; font-size:14px; font-weight:600; text-decoration:none; '
            'padding:12px 32px; border-radius:8px;">Reset Password</a>'
            "</td></tr></table>"
            '<p style="margin:14px 0 0; color:#60564c; font-size:12px; '
            'text-align:center;">This link expires in '
            f"{settings.password_reset_token_expire_minutes} minutes.</p>"
            "</td></tr>"
            '<tr><td style="padding:0 32px;"><hr style="border:none; '
            'border-top:1px solid #e8e5e0; margin:0;"></td></tr>'
            '<tr><td style="padding:20px 32px;">'
            '<p style="margin:0 0 10px; color:#0b0805; font-size:12px; '
            'font-weight:600; text-transform:uppercase;">Request Details</p>'
            '<table width="100%" cellpadding="0" cellspacing="0" '
            f'style="font-size:12px; color:#0b0805;">'
            f"{_request_details_html(_build_context(user_agent, ip_address))}"
            "</table></td></tr>"
        )

        return self._send(to_email, subject, _envelope(subject, body))
