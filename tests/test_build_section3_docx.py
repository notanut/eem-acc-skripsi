"""Section III .docx build (tools/build_section3_docx.py): schema fixes and content. Needs pandoc (skipped if absent)."""
import importlib.util
import re
import shutil
import zipfile
from pathlib import Path

import pytest

etree = pytest.importorskip("lxml.etree")  # tool-only dependency (not in requirements.txt)

REPO = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("pandoc") is None, reason="pandoc not installed")
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


def _load():
    spec = importlib.util.spec_from_file_location("build_s3", REPO / "tools" / "build_section3_docx.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    mod = _load()
    out = tmp_path_factory.mktemp("docx") / "s3.docx"
    mod.build(out)
    return mod, out


def test_docx_has_native_equations_and_valid_math_order(built):
    _, out = built
    with zipfile.ZipFile(out) as z:
        assert not any(n.endswith("/") for n in z.namelist())
        doc = etree.fromstring(z.read("word/document.xml"))
    assert len(list(doc.iter(f"{{{M}}}oMath"))) > 50
    for rpr in doc.iter(f"{{{M}}}rPr"):
        tags = [etree.QName(k).localname for k in rpr]
        if "nor" in tags:
            assert "sty" not in tags and "scr" not in tags
        if "scr" in tags and "sty" in tags:
            assert tags.index("scr") < tags.index("sty")
    for dpr in doc.iter(f"{{{M}}}dPr"):
        tags = [etree.QName(k).localname for k in dpr]
        order = [t for t in ["begChr", "sepChr", "endChr", "grow", "shp"] if t in tags]
        assert [t for t in tags if t in order] == order
    for ppr in doc.iter(f"{{{W}}}pPr"):
        ps = ppr.find(f"{{{W}}}pStyle")
        assert ps is None or ppr.index(ps) == 0


def test_settings_in_schema_order_and_nsid_length(built):
    mod, out = built
    with zipfile.ZipFile(out) as z:
        st = etree.fromstring(z.read("word/settings.xml"))
        num = etree.fromstring(z.read("word/numbering.xml"))
    key = {mod._qname(n): i for i, n in enumerate(mod.SETTINGS_ORDER)}
    idx = [key[k.tag] for k in st]
    assert idx == sorted(idx)
    for n in num.iter(f"{{{W}}}nsid"):
        assert re.fullmatch(r"[0-9A-Fa-f]{8}", n.get(f"{{{W}}}val"))


def test_listing_style_size_and_rpr_order(built):
    mod, out = built
    with zipfile.ZipFile(out) as z:
        styles = etree.fromstring(z.read("word/styles.xml"))
    st = [x for x in styles.iter(f"{{{W}}}style") if x.get(f"{{{W}}}styleId") == "VerbatimChar"]
    assert len(st) == 1
    rpr = st[0].find(f"{{{W}}}rPr")
    tags = [etree.QName(k).localname for k in rpr]
    assert rpr.find(f"{{{W}}}sz").get(f"{{{W}}}val") == "15" and rpr.find(f"{{{W}}}szCs").get(f"{{{W}}}val") == "15"
    assert tags.count("sz") == 1 and tags.index("szCs") == tags.index("sz") + 1
    assert all(tags.index("sz") < tags.index(t) for t in mod.RPR_AFTER_SZCS if t in tags)


def test_styles_fixer_inserts_size_before_later_elements():
    mod = _load()
    xml = (f'<w:styles xmlns:w="{W}"><w:style w:styleId="VerbatimChar"><w:rPr><w:rFonts/><w:sz w:val="22"/>'
           f'<w:lang/></w:rPr></w:style></w:styles>').encode()
    rpr = etree.fromstring(mod.fix_styles(xml)).find(f".//{{{W}}}rPr")
    assert [etree.QName(k).localname for k in rpr] == ["rFonts", "sz", "szCs", "lang"]


def test_settings_fixer_refuses_unknown_elements():
    mod = _load()
    xml = f'<w:settings xmlns:w="{W}"><w:zoom/><w:somethingNew/></w:settings>'.encode()
    with pytest.raises(ValueError):
        mod.fix_settings(xml)


def test_docx_text_carries_patch_r7(built):
    import subprocess
    _, out = built
    txt = subprocess.run(["pandoc", str(out), "-t", "plain", "--wrap=none"], capture_output=True, text=True,
                         check=True).stdout
    for needle in ["504 (Gateway Timeout)", "Algorithm 2: DispatchWithRetry", "Hyndman", "overload rate",
                   "no latency sample", "a 4xx other than 429, or a 500", "whereas a 504", "408, 500 and 502 are retried and count as attempts of the control interval", "set in Stage 3"]:
        assert needle in txt, needle
    assert "throttle rate" not in txt and "(throttled)" not in txt
    assert "429, 503, or time out" not in txt
