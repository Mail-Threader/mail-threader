import re
import quopri
import html
import warnings
from cleantext import clean
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def decode_qp_body(body, encoding):
    if encoding == "quoted-printable":
        try:
            decoded = quopri.decodestring(body.encode("utf-8", errors="replace"))
            return decoded.decode("utf-8", errors="replace")
        except Exception:
            pass
    return body


def clean_qp_artifacts(body):
    if not re.search(r"=", body):
        return body
    body = re.sub(r"=20", " ", body)
    body = re.sub(r"=09", "\t", body)
    body = re.sub(r"=0D=0A", "\n", body)
    body = re.sub(r"=0D", "\r", body)
    body = re.sub(r"=0A", "\n", body)
    return body


def clean_body(body, is_html=False, encoding=""):
    if not body:
        return ""

    body = decode_qp_body(body, encoding)
    body = clean_qp_artifacts(body)

    if is_html or re.search(r"<style|<script|<html", body, re.IGNORECASE):
        body = re.sub(
            r"<(style|script)[^>]*>.*?</\1>", "", body, flags=re.DOTALL | re.IGNORECASE
        )

    body = html.unescape(body)
    body = BeautifulSoup(body, "html.parser").get_text(separator=" ")
    body = re.sub(r"\*{5,}.*?\*{5,}", "", body, flags=re.DOTALL)

    body = re.sub(
        r"^(to:|cc:|subject:|from:|copy:|re:|sent:|bcc:).*\n?",
        "", body, flags=re.MULTILINE | re.IGNORECASE,
    )

    body = re.sub(r"^>+\s?", "", body, flags=re.MULTILINE)

    body = clean(body, fix_unicode=True, to_ascii=True, no_line_breaks=True, lang="en")
    body = re.sub(r"([!?.,;:\-])\1{1,}", r"\1", body)
    body = re.sub(r"[^\x00-\x7F]+", " ", body)
    body = re.sub(r"\s+", " ", body).strip()

    if not re.search(r"[a-zA-Z]", body):
        body = ""

    return body
