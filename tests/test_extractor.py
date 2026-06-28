import pytest
from data_preparation.extractor import split_all_messages, extract_all_emails


class TestSplitAllMessages:
    def test_no_markers(self):
        main, forwards, originals = split_all_messages("From: a@b.com\n\nHello")
        assert main == "From: a@b.com\n\nHello"
        assert forwards == []
        assert originals == []

    def test_with_forwarded(self):
        text = (
            "From: a@b.com\n\nmain body\n"
            "---------------------- Forwarded by X on date ----------------------\n"
            "From: c@d.com\n\nfwd body"
        )
        main, forwards, originals = split_all_messages(text)
        assert "main body" in main
        assert len(forwards) == 1
        assert "fwd body" in forwards[0]

    def test_with_original(self):
        text = (
            "From: a@b.com\n\nmain body\n"
            "-----Original Message-----\n"
            "From: c@d.com\n\noriginal body"
        )
        main, forwards, originals = split_all_messages(text)
        assert "main body" in main
        assert len(originals) == 1
        assert "original body" in originals[0]

    def test_mixed_forwarded_and_original(self):
        text = (
            "From: a@b.com\n\nmain\n"
            "---------------------- Forwarded by X ----------------------\nfwd1\n"
            "-----Original Message-----\norig\n"
            "---------------------- Forwarded by Y ----------------------\nfwd2"
        )
        main, forwards, originals = split_all_messages(text)
        assert len(forwards) == 2
        assert len(originals) == 1


class TestExtractAllEmails:
    def test_returns_list_of_dicts(self):
        result = extract_all_emails("From: a@b.com\nTo: c@d.com\nSubject: test\n\nbody")
        assert isinstance(result, list)
        assert len(result) >= 1
        assert all(isinstance(e, dict) for e in result)

    def test_main_email_has_type(self):
        result = extract_all_emails("From: a@b.com\nSubject: test\n\nbody")
        mains = [e for e in result if e.get("type") == "main"]
        assert len(mains) == 1

    def test_original_email_appended(self):
        text = (
            "From: a@b.com\nSubject: test\n\nmain\n"
            "-----Original Message-----\nFrom: c@d.com\nSubject: orig\n\norig body"
        )
        result = extract_all_emails(text)
        originals = [e for e in result if e.get("type") == "original"]
        assert len(originals) == 1
        assert originals[0]["parent_message_id"] == result[0]["message_id"]

    def test_forwarded_email_appended(self):
        text = (
            "From: a@b.com\nSubject: test\n\nmain\n"
            "---------------------- Forwarded by X ----------------------\n"
            "From: c@d.com\nSubject: fwd\n\nfwd body"
        )
        result = extract_all_emails(text)
        forwards = [e for e in result if e.get("type") == "forwarded"]
        assert len(forwards) >= 1

    def test_empty_input(self):
        result = extract_all_emails("")
        assert len(result) >= 1

    def test_minimal_email(self):
        result = extract_all_emails("From: a@b.com\n\nhello")
        assert result[0]["from"] != ""
