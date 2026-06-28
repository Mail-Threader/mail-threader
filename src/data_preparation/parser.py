import re
import email
import email.utils
import email.header

from . import utils


FIELD_NAMES = ("From", "To", "Cc", "Subject", "Date", "Sent")


def split_headers_body(email_text):
    lines = email_text.splitlines()
    headers = []
    body_lines = []
    found_split = False
    for line in lines:
        if not found_split:
            if re.match(r"^\s*$", line):
                found_split = True
                continue
            headers.append(line)
        else:
            body_lines.append(line)
    return "\n".join(headers), "\n".join(body_lines)


def parse_addresses(raw):
    if not raw:
        return ""
    parsed = email.utils.getaddresses([raw])
    addrs = [addr[1] for addr in parsed if addr[1] and "@" in addr[1]]
    return "; ".join(addrs) if addrs else raw.strip()


def decode_mime_header(value):
    if not value:
        return ""
    try:
        decoded_parts = email.header.decode_header(value)
        result = []
        for part, charset in decoded_parts:
            if isinstance(part, bytes):
                try:
                    result.append(part.decode(charset or "utf-8", errors="replace"))
                except (LookupError, UnicodeDecodeError):
                    result.append(part.decode("utf-8", errors="replace"))
            else:
                result.append(part)
        return " ".join(result)
    except Exception:
        return value


def detect_encoding(header_text):
    m = re.search(
        r"^Content-Transfer-Encoding:\s*(\S+)",
        header_text, re.MULTILINE | re.IGNORECASE,
    )
    return m.group(1).strip().lower() if m else ""


def parse_references(header_text):
    for pattern in [r"^In-Reply-To:\s*<([^>]+)>", r"^References:\s*<([^>]+)>"]:
        m = re.search(pattern, header_text, re.MULTILINE)
        if m:
            return m.group(1).strip()
    return ""


def parse_headers_main(header_text):
    fields = {
        "message_id": "", "filename": "", "date": "",
        "from": "", "to": "", "cc": "", "subject": "",
        "X-From": "", "X-To": "", "X-cc": "",
    }

    unfolded = re.sub(r"\n[ \t]+", " ", header_text)
    for line in unfolded.split("\n"):
        if line.startswith("Message-ID:"):
            fields["message_id"] = line.split(":", 1)[1].strip()
        elif line.startswith("Date:"):
            fields["date"] = utils.normalize_dates(line.split(":", 1)[1].strip())
        elif line.startswith("From:"):
            fields["from"] = parse_addresses(line.split(":", 1)[1].strip())
        elif line.startswith("To:"):
            fields["to"] = parse_addresses(line.split(":", 1)[1].strip())
        elif line.startswith("Cc:"):
            fields["cc"] = parse_addresses(line.split(":", 1)[1].strip())
        elif line.startswith("Subject:"):
            fields["subject"] = decode_mime_header(line.split(":", 1)[1].strip())
        elif line.startswith("X-From:"):
            fields["X-From"] = utils.strip_x500_dn(line.split(":", 1)[1].strip())
        elif line.startswith("X-To:"):
            fields["X-To"] = utils.strip_x500_dn(line.split(":", 1)[1].strip())
        elif line.startswith("X-cc:"):
            fields["X-cc"] = utils.strip_x500_dn(line.split(":", 1)[1].strip())

    if not fields["from"] and fields["X-From"]:
        fields["from"] = fields["X-From"]
    if not fields["to"] and fields["X-To"]:
        fields["to"] = fields["X-To"]

    return fields


def extract_fields_line_by_line(text):
    lines = text.split("\n")
    fields = {}
    current_field = None
    current_value = []
    body_lines = []
    in_headers = True
    found_any_field = False

    for line in lines:
        stripped = line.strip()
        field_match = re.match(
            r"^[ \t]*(" + "|".join(FIELD_NAMES) + r"):[ \t]*(.*)",
            line,
            re.IGNORECASE,
        )
        if field_match:
            fname = field_match.group(1).capitalize()
            fvalue = field_match.group(2)
            if current_field and current_value:
                fields[current_field] = " ".join(current_value)
            current_field = fname
            current_value = [fvalue] if fvalue else []
            found_any_field = True
            in_headers = True
        elif in_headers and found_any_field:
            if stripped == "":
                in_headers = False
            elif current_field:
                if line[0] in (" ", "\t"):
                    current_value.append(stripped)
                else:
                    if current_field and current_value:
                        fields[current_field] = " ".join(current_value)
                    current_field = None
                    current_value = []
                    body_lines.append(line)
                    in_headers = False
        elif in_headers and not found_any_field:
            pass
        else:
            body_lines.append(line)

    if current_field and current_value:
        fields[current_field] = " ".join(current_value)

    return fields, "\n".join(body_lines)


def parse_original_block(original_block):
    org_data = {"date": "", "original_sender": "", "to": "", "cc": "", "subject": "", "body": ""}

    content = re.sub(
        r"^-{2,}\s*Original Message\s*-{2,}",
        "", original_block, count=1, flags=re.MULTILINE | re.IGNORECASE,
    ).strip()

    fields, body_text = extract_fields_line_by_line(content)

    sender_raw = fields.get("From", "")
    org_data["original_sender"] = parse_addresses(sender_raw)
    org_data["to"] = parse_addresses(fields.get("To", ""))
    org_data["cc"] = parse_addresses(fields.get("Cc", ""))
    org_data["subject"] = decode_mime_header(fields.get("Subject", ""))
    date_val = fields.get("Date", fields.get("Sent", "")).strip()
    org_data["date"] = date_val
    org_data["body"] = body_text.strip()

    encoding = ""
    m = re.search(
        r"^Content-Transfer-Encoding:\s*(\S+)", content, re.MULTILINE | re.IGNORECASE,
    )
    if m:
        encoding = m.group(1).strip().lower()
    org_data["encoding"] = encoding

    return org_data


def parse_forwarded_block(forward_block):
    fwd_data = {"date": "", "from": "", "to": "", "cc": "", "subject": "", "body": ""}

    forward_info = re.search(
        r"-+ Forwarded by (.+?) on\s+([A-Za-z0-9 /:]+(?:AM|PM)?)",
        forward_block, re.IGNORECASE,
    )
    if forward_info:
        fwd_data["date"] = forward_info.group(2).strip()
        fwd_data["from"] = forward_info.group(1).strip()

    content = re.sub(
        r"^-{2,}.*?(?:Forwarded|Original).*(?:\n|$)",
        "", forward_block, count=1, flags=re.MULTILINE | re.IGNORECASE,
    ).strip()

    fields, body_text = extract_fields_line_by_line(content)

    if not fwd_data.get("from") and fields.get("From"):
        fwd_data["from"] = parse_addresses(fields["From"])
    elif not fwd_data.get("from") and fields.get("Sent"):
        fwd_data["from"] = parse_addresses(fields["Sent"])

    if fields.get("To"):
        fwd_data["to"] = parse_addresses(fields["To"])
    if fields.get("Cc"):
        fwd_data["cc"] = parse_addresses(fields["Cc"])

    if fields.get("Subject"):
        fwd_data["subject"] = decode_mime_header(fields["Subject"])

    if fields.get("Date") and not fwd_data.get("date"):
        fwd_data["date"] = fields["Date"]
    if fields.get("Sent") and not fwd_data.get("date"):
        fwd_data["date"] = fields["Sent"]

    fwd_data["body"] = body_text

    encoding = ""
    m = re.search(
        r"^Content-Transfer-Encoding:\s*(\S+)", content, re.MULTILINE | re.IGNORECASE,
    )
    if m:
        encoding = m.group(1).strip().lower()
    fwd_data["encoding"] = encoding

    return fwd_data
