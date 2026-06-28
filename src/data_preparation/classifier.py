import re


def classify_content_type(row: dict) -> str:
    subject = (row.get("subject") or "").strip()
    body = (row.get("body") or "").strip()
    filename = (row.get("filename") or "").strip()
    sender = (row.get("from") or "").strip()

    # Conversation: Re:/Fwd: in subject or known reply pattern
    if re.search(r"^(Re|Fwd|FW|AW|WG):", subject, re.IGNORECASE):
        return "conversation"

    # Attachment notice: body patterns
    if _is_attachment_notice(body, subject):
        return "attachment_notice"

    # Log: system-generated, no real sender
    if _is_log_file(filename, body, sender):
        return "log"

    # Newsletter: high link ratio, unsubscribe
    if _is_newsletter(body):
        return "newsletter"

    return "conversation"


def _is_attachment_notice(body: str, subject: str) -> bool:
    body_lower = body.lower()
    subject_lower = subject.lower()
    if re.search(r"\bATTENTION\b", subject):
        return True
    body_attach = bool(re.search(r"\b(attachment|attached)\b", body_lower))
    subj_attach = bool(re.search(r"\b(attachment|attached)\b", subject_lower))
    if (body_attach or subj_attach) and "archive" not in subject_lower:
        return True
    return False


def _is_log_file(filename: str, body: str, sender: str) -> bool:
    is_index_file = bool(re.match(r"^\d+\.\s*\d+$", filename))
    is_archiver = bool(re.search(r"archiving|enron\.com.*archive", sender.lower()))
    body_lower = (body or "").lower()
    has_created_by = bool(re.search(r"\(created by", body_lower))
    has_re_entries = bool(re.search(r"(^|\n)\s*re:", body_lower))

    if is_index_file and (is_archiver or has_created_by):
        return True
    if is_index_file and has_re_entries and len(body or "") > 1000:
        return True
    return False


def _is_newsletter(body: str) -> bool:
    if len(body) < 500:
        return False
    body_lower = body.lower()
    if re.search(r"unsubscribe|to\s+unsubscribe", body_lower):
        return True
    links = len(re.findall(r"https?://", body_lower))
    if links > 5 and links / max(len(body.split()), 1) > 0.05:
        return True
    return False


def classify_all(df):
    df["content_type"] = df.apply(classify_content_type, axis=1)
    return df
