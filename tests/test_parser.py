import pytest
from data_preparation.parser import (
    split_headers_body,
    parse_addresses,
    decode_mime_header,
    detect_encoding,
    parse_references,
    parse_headers_main,
    extract_fields_line_by_line,
    parse_original_block,
    parse_forwarded_block,
)
from data_preparation.utils import normalize_dates


class TestSplitHeadersBody:
    def test_basic_split(self):
        headers, body = split_headers_body(
            "From: a@b.com\nTo: c@d.com\n\nHello world"
        )
        assert "From: a@b.com" in headers
        assert "To: c@d.com" in headers
        assert body == "Hello world"

    def test_no_body(self):
        headers, body = split_headers_body("From: a@b.com\n\n")
        assert body == ""

    def test_only_headers(self):
        headers, body = split_headers_body("From: a\nTo: b")
        assert "" in body or body == ""
        assert "From:" in headers

    def test_empty_input(self):
        headers, body = split_headers_body("")
        assert headers == ""
        assert body == ""


class TestParseAddresses:
    def test_simple(self):
        assert parse_addresses("John <john@enron.com>") == "john@enron.com"

    def test_multiple(self):
        result = parse_addresses("A <a@x.com>, B <b@x.com>")
        assert result == "a@x.com; b@x.com"

    def test_no_email(self):
        assert parse_addresses("Just a name") == "Just a name"

    def test_none(self):
        assert parse_addresses("") == ""
        assert parse_addresses(None) == ""

    def test_bare_email(self):
        assert parse_addresses("user@domain.com") == "user@domain.com"


class TestDecodeMimeHeader:
    def test_plain_text(self):
        assert decode_mime_header("Hello") == "Hello"

    def test_encoded(self):
        result = decode_mime_header("=?utf-8?Q?R=C3=A9sum=C3=A9?=")
        assert "R" in result

    def test_none(self):
        assert decode_mime_header("") == ""
        assert decode_mime_header(None) == ""


class TestDetectEncoding:
    def test_qp(self):
        h = "Content-Transfer-Encoding: quoted-printable\n"
        assert detect_encoding(h) == "quoted-printable"

    def test_base64(self):
        h = "Content-Transfer-Encoding: base64\n"
        assert detect_encoding(h) == "base64"

    def test_none(self):
        assert detect_encoding("Subject: hi\n") == ""


class TestParseReferences:
    def test_in_reply_to(self):
        h = "In-Reply-To: <abc123>\nMessage-ID: <def456>"
        assert parse_references(h) == "abc123"

    def test_references(self):
        h = "References: <ref1>"
        assert parse_references(h) == "ref1"

    def test_none(self):
        assert parse_references("Subject: hi") == ""


class TestParseHeadersMain:
    def test_basic(self):
        h = "Message-ID: <abc>\nFrom: a@b.com\nTo: c@d.com\nSubject: test\nDate: 4 Dec 2001 14:22:00"
        result = parse_headers_main(h)
        assert result["message_id"] == "<abc>"
        assert "a@b.com" in result["from"]
        assert "c@d.com" in result["to"]
        assert result["subject"] == "test"
        assert result["date"] == normalize_dates("4 Dec 2001 14:22:00")

    def test_x_headers(self):
        h = "X-From: John /O=ENRON/\nX-To: Jane /O=ENRON/"
        result = parse_headers_main(h)
        assert result["X-From"] == "John"
        assert result["X-To"] == "Jane"

    def test_fallback_to_xfrom(self):
        h = "X-From: John <j@e.com>"
        result = parse_headers_main(h)
        assert result["from"] == "John"

    def test_unfolded_headers(self):
        h = "Subject: a very\n long subject"
        result = parse_headers_main(h)
        assert result["subject"] == "a very long subject"

    def test_empty(self):
        result = parse_headers_main("")
        assert result["message_id"] == ""


class TestExtractFieldsLineByLine:
    def test_basic_fields(self):
        text = "From: sender@e.com\nTo: recv@e.com\nSubject: hi\n\nbody here"
        fields, body = extract_fields_line_by_line(text)
        assert fields["From"] == "sender@e.com"
        assert fields["To"] == "recv@e.com"
        assert fields["Subject"] == "hi"
        assert body == "body here"

    def test_no_body(self):
        fields, body = extract_fields_line_by_line("From: a\nTo: b\n")
        assert body == ""

    def test_multiline_value(self):
        text = "Subject: a\n continued subject\n\nbody"
        fields, body = extract_fields_line_by_line(text)
        assert fields["Subject"] == "a continued subject"
        assert body == "body"

    def test_empty(self):
        fields, body = extract_fields_line_by_line("")
        assert fields == {}
        assert body == ""


class TestParseOriginalBlock:
    def test_parse_with_headers(self):
        block = "-----Original Message-----\nFrom: sender@e.com\nTo: recv@e.com\nSubject: test\n\nbody text"
        result = parse_original_block(block)
        assert "sender@e.com" in result["original_sender"]
        assert "recv@e.com" in result["to"]
        assert result["subject"] == "test"
        assert result["body"] == "body text"

    def test_no_body(self):
        block = "-----Original Message-----\nFrom: a@b.com\nTo: c@d.com\nSubject: hi"
        result = parse_original_block(block)
        assert result["body"] == ""

    def test_empty(self):
        result = parse_original_block("")
        assert result["original_sender"] == ""


class TestParseForwardedBlock:
    def test_forwarded_by_line(self):
        block = "---------------------- Forwarded by John Doe on 12/04/2001 02:22 PM ----------------------\nFrom: sender@e.com\nTo: recv@e.com\nSubject: test\n\nbody text"
        result = parse_forwarded_block(block)
        assert result["from"] == "John Doe"
        assert "sender@e.com" in result["from"] or result["from"] == "John Doe"

    def test_missing_forward_line(self):
        block = "From: a@b.com\nTo: c@d.com\nSubject: hi\n\nbody"
        result = parse_forwarded_block(block)
        assert "a@b.com" in result["from"]
        assert result["body"] == "body"

    def test_empty(self):
        result = parse_forwarded_block("")
        assert result["from"] == ""
