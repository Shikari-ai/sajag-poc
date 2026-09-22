from sajag.links import check_url, extract_urls


def test_official_sites_are_clean():
    for u in ["https://www.onlinesbi.sbi", "https://www.hdfcbank.com/login",
              "https://echallan.parivahan.gov.in", "https://uidai.gov.in/en"]:
        assert check_url(u).risk == 0.0, u


def test_brand_impersonation_is_flagged():
    v = check_url("http://sbi-kyc-update.xyz/verify")
    assert v.risk >= 0.8
    assert any("SBI" in r for r in v.reasons)


def test_lookalikes_are_flagged():
    for u in ["https://icicibnak.com/login", "https://hdfcbank.co/login", "https://axisbnk.com"]:
        v = check_url(u)
        assert v.risk >= 0.8, u
        assert any("look-alike" in r for r in v.reasons), u


def test_apk_download_ip_and_shortener():
    assert check_url("https://sbi-yono.top/SBI_Rewards.apk").risk >= 0.9
    assert check_url("http://45.12.99.7/app/Update.apk").risk >= 0.9
    v = check_url("https://bit.ly/3xYz9Ab")
    assert 0.4 <= v.risk < 0.8
    assert any("hides" in r for r in v.reasons)


def test_punycode_is_flagged():
    assert check_url("https://xn--sb-lka.com/login").risk >= 0.7


def test_unknown_but_plain_site_is_not_flagged():
    v = check_url("https://www.example.com/menu")
    assert v.risk == 0.0
    assert "not a guarantee" in v.reasons[0]


def test_extract_urls_strips_trailing_punctuation():
    assert extract_urls("Pay here: https://bit.ly/abc. Thanks") == ["https://bit.ly/abc"]
