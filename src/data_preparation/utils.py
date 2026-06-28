import re
import dateparser


def normalize_dates(text):
    if not text:
        return ""
    cleaned_text = re.sub(r"\s*\([A-Z]{2,}\)$", "", text.strip())
    parsed_date = dateparser.parse(
        cleaned_text,
        settings={"TIMEZONE": "UTC", "TO_TIMEZONE": "UTC", "RETURN_AS_TIMEZONE_AWARE": False},
    )
    if parsed_date:
        return parsed_date.strftime("%d/%m/%Y %H:%M:%S")
    return ""


def strip_x500_dn(value):
    cleaned = re.sub(r"\s*/O=[^ ]+(?: [^ ]+)*/?", "", value)
    cleaned = re.sub(r"\s*</?O=ENRON[^>]*>", "", cleaned)
    cleaned = re.sub(r"\s*<[^>]*>", "", cleaned)
    return cleaned.strip()
