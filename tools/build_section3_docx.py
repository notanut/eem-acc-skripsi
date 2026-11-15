"""Build docs/Section_III_Methodology_revised.docx from docs/section3_revised.md.

pandoc writes native Word equations (OMML) but a few elements in an order the OOXML schema does not allow; this
script reorders them (no content change):
  - m:rPr: m:nor excludes m:scr/m:sty (drop both next to m:nor); m:scr must precede m:sty
  - m:dPr: begChr, sepChr, endChr, grow, shp
  - w:settings: children in CT_Settings schema order
  - w:pPr: w:pStyle first; w:nsid padded to 8 hex digits
  - VerbatimChar (algorithm listings) at 7.5 pt
Known and left alone: 4 schema-order notes in pandoc's default styles.xml (Word opens the file).
Built and checked with pandoc 3.1.3; another pandoc version may produce a slightly different file.
Usage: python tools/build_section3_docx.py [out.docx]
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

from lxml import etree

REPO = Path(__file__).resolve().parents[1]
M = "http://schemas.openxmlformats.org/officeDocument/2006/math"
W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
DPR_ORDER = ["begChr", "sepChr", "endChr", "grow", "shp", "ctrlPr"]
RPR_ORDER = ["lit", "nor", "scr", "sty", "brk", "aln"]


def _reorder(el, order, ns):
    kids = list(el)
    key = {f"{{{ns}}}{n}": i for i, n in enumerate(order)}
    kids.sort(key=lambda k: key.get(k.tag, len(order)))
    for k in kids:
        el.remove(k)
    for k in kids:
        el.append(k)


def fix_document(xml: bytes) -> bytes:
    root = etree.fromstring(xml)
    for rpr in root.iter(f"{{{M}}}rPr"):
        if rpr.find(f"{{{M}}}nor") is not None:
            for bad in rpr.findall(f"{{{M}}}sty") + rpr.findall(f"{{{M}}}scr"):
                rpr.remove(bad)
        _reorder(rpr, RPR_ORDER, M)
    for dpr in root.iter(f"{{{M}}}dPr"):
        _reorder(dpr, DPR_ORDER, M)
    for ppr in root.iter(f"{{{W}}}pPr"):
        ps = ppr.find(f"{{{W}}}pStyle")
        if ps is not None and ppr.index(ps) != 0:
            ppr.remove(ps)
            ppr.insert(0, ps)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


SETTINGS_ORDER = (  # CT_Settings sequence, ISO/IEC 29500-4 wml.xsd
    "writeProtection view zoom removePersonalInformation removeDateAndTime doNotDisplayPageBoundaries "
    "displayBackgroundShape printPostScriptOverText printFractionalCharacterWidth printFormsData embedTrueTypeFonts "
    "embedSystemFonts saveSubsetFonts saveFormsData mirrorMargins alignBordersAndEdges bordersDoNotSurroundHeader "
    "bordersDoNotSurroundFooter gutterAtTop hideSpellingErrors hideGrammaticalErrors activeWritingStyle proofState "
    "formsDesign attachedTemplate linkStyles stylePaneFormatFilter stylePaneSortMethod documentType mailMerge "
    "revisionView trackRevisions doNotTrackMoves doNotTrackFormatting documentProtection autoFormatOverride "
    "styleLockTheme styleLockQFSet defaultTabStop autoHyphenation consecutiveHyphenLimit hyphenationZone "
    "doNotHyphenateCaps showEnvelope summaryLength clickAndTypeStyle defaultTableStyle evenAndOddHeaders "
    "bookFoldRevPrinting bookFoldPrinting bookFoldPrintingSheets drawingGridHorizontalSpacing "
    "drawingGridVerticalSpacing displayHorizontalDrawingGridEvery displayVerticalDrawingGridEvery "
    "doNotUseMarginsForDrawingGridOrigin drawingGridHorizontalOrigin drawingGridVerticalOrigin doNotShadeFormData "
    "noPunctuationKerning characterSpacingControl printTwoOnOne strictFirstAndLastChars noLineBreaksAfter "
    "noLineBreaksBefore savePreviewPicture doNotValidateAgainstSchema saveInvalidXml ignoreMixedContent "
    "alwaysShowPlaceholderText doNotDemarcateInvalidXml saveXmlDataOnly useXSLTWhenSaving saveThroughXslt "
    "showXMLTags alwaysMergeEmptyNamespace updateFields hdrShapeDefaults footnotePr endnotePr compat docVars rsids "
    "m:mathPr attachedSchema themeFontLang clrSchemeMapping doNotIncludeSubdocsInStats doNotAutoCompressPictures "
    "forceUpgrade captions readModeInkLockDown smartTagType sl:schemaLibrary shapeDefaults doNotEmbedSmartTags "
    "decimalSymbol listSeparator"
).split()
SL = "http://schemas.openxmlformats.org/schemaLibrary/2006/main"


def _qname(n: str) -> str:
    if n.startswith("m:"):
        return f"{{{M}}}{n[2:]}"
    if n.startswith("sl:"):
        return f"{{{SL}}}{n[3:]}"
    return f"{{{W}}}{n}"


def fix_settings(xml: bytes) -> bytes:
    root = etree.fromstring(xml)
    key = {_qname(n): i for i, n in enumerate(SETTINGS_ORDER)}
    unknown = [k.tag for k in root if not isinstance(k.tag, str) or k.tag not in key]
    if unknown:
        raise ValueError(f"settings.xml: unknown elements {unknown}; extend SETTINGS_ORDER")
    kids = sorted(root, key=lambda k: key[k.tag])
    for k in kids:
        root.remove(k)
    for k in kids:
        root.append(k)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def fix_numbering(xml: bytes) -> bytes:
    root = etree.fromstring(xml)
    for nsid in root.iter(f"{{{W}}}nsid"):
        v = nsid.get(f"{{{W}}}val")
        nsid.set(f"{{{W}}}val", v.zfill(8)[-8:])  # ST_LongHexNumber: exactly 8 hex digits
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


RPR_AFTER_SZCS = ("highlight", "u", "effect", "bdr", "shd", "fitText", "vertAlign", "rtl", "cs", "em", "lang",
                  "eastAsianLayout", "specVanish", "oMath")


def fix_styles(xml: bytes) -> bytes:
    """Algorithm listings: 7.5 pt monospace so lines up to ~100 characters fit the text width without wrapping."""
    root = etree.fromstring(xml)
    for st in root.iter(f"{{{W}}}style"):
        if st.get(f"{{{W}}}styleId") == "VerbatimChar":
            rpr = st.find(f"{{{W}}}rPr")
            if rpr is None:
                rpr = etree.SubElement(st, f"{{{W}}}rPr")
            for tag in ("sz", "szCs"):
                for old in rpr.findall(f"{{{W}}}{tag}"):
                    rpr.remove(old)
            # CT_RPr sequence: sz, szCs come before these elements
            after = {f"{{{W}}}{t}" for t in RPR_AFTER_SZCS}
            pos = next((i for i, k in enumerate(rpr) if k.tag in after), len(rpr))
            for i, tag in enumerate(("sz", "szCs")):
                el = etree.Element(f"{{{W}}}{tag}")
                el.set(f"{{{W}}}val", "15")
                rpr.insert(pos + i, el)
    return etree.tostring(root, xml_declaration=True, encoding="UTF-8", standalone=True)


def build(out: Path) -> None:
    src = REPO / "docs" / "section3_revised.md"
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw.docx"
        subprocess.run(["pandoc", str(src), "-f", "markdown", "-o", str(raw)], check=True)
        with zipfile.ZipFile(raw) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                if info.filename.endswith("/"):
                    continue  # no folder entries
                data = zin.read(info.filename)
                if info.filename == "word/document.xml":
                    data = fix_document(data)
                elif info.filename == "word/settings.xml":
                    data = fix_settings(data)
                elif info.filename == "word/styles.xml":
                    data = fix_styles(data)
                elif info.filename == "word/numbering.xml":
                    data = fix_numbering(data)
                zout.writestr(info.filename, data)


if __name__ == "__main__":
    build(Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "docs" / "Section_III_Methodology_revised.docx")
