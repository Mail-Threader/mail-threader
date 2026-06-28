import re

from . import parser
from . import cleaner
from . import utils


def split_all_messages(email_text):
    forward_pattern = re.compile(
        r"-{2,}\s*Forwarded by\s+.*?-{2,}", re.DOTALL | re.MULTILINE | re.IGNORECASE
    )
    original_pattern = re.compile(
        r"-{2,}\s*Original Message\s*-{2,}", re.MULTILINE | re.IGNORECASE
    )

    forward_matches = list(forward_pattern.finditer(email_text))
    original_matches = list(original_pattern.finditer(email_text))
    all_markers = sorted(
        [(m.start(), "forwarded", m) for m in forward_matches]
        + [(m.start(), "original", m) for m in original_matches]
    )

    if not all_markers:
        return email_text.strip(), [], []

    main_text = email_text[: all_markers[0][0]]
    forward_parts = []
    original_parts = []

    for i, (start, kind, m) in enumerate(all_markers):
        end = all_markers[i + 1][0] if i + 1 < len(all_markers) else len(email_text)
        segment = email_text[start:end]
        if kind == "forwarded":
            forward_parts.append(segment)
        else:
            original_parts.append(segment)

    return main_text.strip(), forward_parts, original_parts


def extract_all_emails(email_text):
    extracted_emails = []
    main_text, forwards_raw, original_parts = split_all_messages(email_text)

    # --- Main email ---
    header_part, body = parser.split_headers_body(main_text)
    main_data = parser.parse_headers_main(header_part)
    is_html = bool(re.search(r"Content-Type:\s*text/html", header_part, re.IGNORECASE))
    encoding = parser.detect_encoding(header_part)
    main_data["body"] = cleaner.clean_body(body, is_html=is_html, encoding=encoding)
    main_data["type"] = "main"
    main_data["parent_message_id"] = parser.parse_references(header_part)
    main_data["is_html"] = is_html
    main_data["has_body"] = bool(main_data.get("body"))
    extracted_emails.append(main_data)

    # --- Original emails ---
    for org_text in original_parts:
        org_data = {
            "message_id": f"{main_data['message_id']}_original",
            "parent_message_id": main_data["message_id"],
            "main_id": main_data["message_id"],
            "filename": None, "type": "original",
            "date": None, "from": None, "to": None, "cc": None,
            "X-From": None, "X-To": None, "X-cc": None,
            "subject": None, "body": None, "has_body": False, "is_html": False,
        }
        org_text = re.sub(r"\.{2,}", ".", org_text)
        parsed_org = parser.parse_original_block(org_text)
        org_data["date"] = utils.normalize_dates(parsed_org.get("date") or None)
        org_data["from"] = parsed_org.get("original_sender")
        org_data["to"] = parsed_org.get("to")
        org_data["cc"] = parsed_org.get("cc")
        org_data["subject"] = parsed_org.get("subject")
        body_raw = parsed_org.get("body", "")
        encoding = parsed_org.get("encoding", "")
        org_data["body"] = cleaner.clean_body(body_raw, is_html=False, encoding=encoding)
        org_data["has_body"] = bool(org_data.get("body"))
        extracted_emails.append(org_data)

    # --- Forwarded emails ---
    for fwd_text in forwards_raw:
        fwd_data = {
            "message_id": f"{main_data['message_id']}_forwarded",
            "parent_message_id": main_data["message_id"],
            "main_id": main_data["message_id"],
            "filename": None, "type": "forwarded",
            "date": None, "from": None, "to": None, "cc": None,
            "X-From": None, "X-To": None, "X-cc": None,
            "subject": None, "body": None, "has_body": False, "is_html": False,
        }
        parsed_fwd = parser.parse_forwarded_block(fwd_text)
        fwd_data["date"] = utils.normalize_dates(parsed_fwd.get("date") or None)
        fwd_data["from"] = parsed_fwd.get("from")
        fwd_data["to"] = parsed_fwd.get("to")
        fwd_data["cc"] = parsed_fwd.get("cc")
        fwd_data["subject"] = parsed_fwd.get("subject")
        body_raw = parsed_fwd.get("body", "")
        encoding = parsed_fwd.get("encoding", "")
        fwd_data["body"] = cleaner.clean_body(body_raw, is_html=False, encoding=encoding)
        fwd_data["has_body"] = bool(fwd_data.get("body"))
        extracted_emails.append(fwd_data)

    return extracted_emails
