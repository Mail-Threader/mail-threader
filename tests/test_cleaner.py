from data_preparation.cleaner import clean_body, clean_qp_artifacts


def test_clean_body_empty():
    assert clean_body("") == ""
    assert clean_body(None) == ""


def test_clean_body_plain_text():
    result = clean_body("Hello world", is_html=False)
    assert "hello world" in result


def test_clean_body_html_stripped():
    result = clean_body("<html><body><p>Hello</p></body></html>", is_html=True)
    assert "hello" in result
    assert "<html>" not in result


def test_clean_body_html_auto_detected():
    result = clean_body("<div>Hello</div>")
    assert "hello" in result


def test_clean_body_strips_quoting():
    result = clean_body("> quoted text\n> more\nactual reply")
    assert "actual reply" in result
    assert "quoted text" in result


def test_clean_body_removes_header_lines():
    result = clean_body("From: someone\nTo: someone\nSubject: re: stuff\nactual content")
    assert "actual content" in result


def test_clean_qp_artifacts_no_qp():
    assert clean_qp_artifacts("hello world") == "hello world"


def test_clean_qp_artifacts_with_qp():
    result = clean_qp_artifacts("=20hello=20world=20")
    assert "hello world" in result


def test_clean_body_empty_after_non_alpha():
    result = clean_body("1234 5678 90!@#$%^&*()")
    assert result == ""
