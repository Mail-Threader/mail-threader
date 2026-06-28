from data_preparation.utils import normalize_dates, strip_x500_dn


def test_normalize_dates_none():
    assert normalize_dates(None) == ""
    assert normalize_dates("") == ""


def test_normalize_dates_common_format():
    result = normalize_dates("Mon, 4 Dec 2001 14:22:00 -0800 (PST)")
    assert result == "04/12/2001 22:22:00"


def test_normalize_dates_no_timezone():
    result = normalize_dates("4 Dec 2001 14:22:00")
    assert result == "04/12/2001 14:22:00"


def test_normalize_dates_unparseable():
    assert normalize_dates("garbage") == ""


def test_strip_x500_dn_typical():
    result = strip_x500_dn("John Doe /O=ENRON/")
    assert result == "John Doe"


def test_strip_x500_dn_html_tag():
    result = strip_x500_dn("John Doe <O=ENRON>")
    assert result == "John Doe"


def test_strip_x500_dn_full_email_tag():
    result = strip_x500_dn("John Doe <john@enron.com>")
    assert result == "John Doe"


def test_strip_x500_dn_none():
    assert strip_x500_dn(None) == ""


def test_strip_x500_dn_clean_already():
    result = strip_x500_dn("john.doe@enron.com")
    assert result == "john.doe@enron.com"


def test_strip_x500_dn_mixed():
    result = strip_x500_dn("/O=ENRON/ John Doe /O=ENRON/")
    assert result == "John Doe"

def test_strip_x500_dn_no_whitespace_before():
    result = strip_x500_dn("/O=ENRON/jdoe@enron.com")
    assert result == "jdoe@enron.com"
