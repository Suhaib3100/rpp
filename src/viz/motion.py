"""Slide transitions and entrance animations.

python-pptx has no API for either, so both are injected as raw DrawingML. The
elements used are the plain PresentationML ones that PowerPoint, Keynote and
LibreOffice all understand:

* ``<p:transition>`` — a slide-level transition. We use ``<p:fade>``, the one
  transition that reads as intentional rather than decorative and that every
  renderer implements identically.
* ``<p:timing>`` — a build. We emit the minimal main-sequence tree that fades
  selected shapes in on click, which is what PowerPoint itself writes for a
  "Fade" entrance effect.

Both are deliberately conservative. Anything more elaborate (morph, fly-in,
motion paths) is renderer-dependent and the first thing to break on import, and
a research talk gains nothing from it.
"""
from __future__ import annotations

from pptx.oxml.ns import qn
from lxml import etree

P_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"


def _el(tag: str, **attrs) -> etree._Element:
    e = etree.SubElement(etree.Element("root"), qn(tag))
    for k, v in attrs.items():
        e.set(k, str(v))
    return e


def add_fade_transition(slide, duration_ms: int = 400) -> None:
    """Fade this slide in, advancing on click.

    ``p:transition`` must be the last child of ``p:sld`` after ``p:clrMapOvr``,
    so it is appended rather than inserted.
    """
    sld = slide._element
    for existing in sld.findall(qn("p:transition")):
        sld.remove(existing)
    tr = etree.SubElement(sld, qn("p:transition"))
    tr.set("spd", "med")
    tr.set(
        "{http://schemas.microsoft.com/office/powerpoint/2010/main}dur",
        str(duration_ms),
    )
    etree.SubElement(tr, qn("p:fade"))


_TIMING_TEMPLATE = """
<p:timing xmlns:p="{p}" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
  <p:tnLst>
    <p:par><p:cTn id="1" dur="indefinite" restart="never" nodeType="tmRoot">
      <p:childTnLst>
        <p:seq concurrent="1" nextAc="seek">
          <p:cTn id="2" dur="indefinite" nodeType="mainSeq">
            <p:childTnLst>{groups}</p:childTnLst>
          </p:cTn>
          <p:prevCondLst><p:cond evt="onPrev" delay="0">
            <p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:prevCondLst>
          <p:nextCondLst><p:cond evt="onNext" delay="0">
            <p:tgtEl><p:sldTgt/></p:tgtEl></p:cond></p:nextCondLst>
        </p:seq>
      </p:childTnLst>
    </p:cTn></p:par>
  </p:tnLst>
</p:timing>
"""

_GROUP_TEMPLATE = """
<p:par><p:cTn id="{a}" fill="hold"><p:stCondLst><p:cond delay="{delay}"/></p:stCondLst>
  <p:childTnLst>
    <p:par><p:cTn id="{b}" fill="hold"><p:stCondLst><p:cond delay="0"/></p:stCondLst>
      <p:childTnLst>
        <p:par><p:cTn id="{c}" presetID="10" presetClass="entr" presetSubtype="0"
                      fill="hold" grpId="0" nodeType="{node}">
          <p:stCondLst><p:cond delay="0"/></p:stCondLst>
          <p:childTnLst>
            <p:set><p:cBhvr><p:cTn id="{d}" dur="1" fill="hold">
              <p:stCondLst><p:cond delay="0"/></p:stCondLst></p:cTn>
              <p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl>
              <p:attrNameLst><p:attrName>style.visibility</p:attrName></p:attrNameLst>
            </p:cBhvr><p:to><p:strVal val="visible"/></p:to></p:set>
            <p:animEffect transition="in" filter="fade">
              <p:cBhvr><p:cTn id="{e}" dur="{dur}"/>
                <p:tgtEl><p:spTgt spid="{spid}"/></p:tgtEl></p:cBhvr>
            </p:animEffect>
          </p:childTnLst>
        </p:cTn></p:par>
      </p:childTnLst>
    </p:cTn></p:par>
  </p:childTnLst>
</p:cTn></p:par>
"""


def add_fade_in_build(slide, shapes, dur_ms: int = 350, stagger_ms: int = 120) -> None:
    """Fade ``shapes`` in one after another, automatically after the slide loads.

    The first element starts on click (``clickEffect``); the rest follow
    automatically (``afterEffect``) so the whole build runs from a single
    advance rather than needing one click per element.
    """
    shapes = [s for s in shapes if s is not None]
    if not shapes:
        return
    sld = slide._element
    for existing in sld.findall(qn("p:timing")):
        sld.remove(existing)

    groups, nid = [], 3
    for i, sh in enumerate(shapes):
        groups.append(_GROUP_TEMPLATE.format(
            a=nid, b=nid + 1, c=nid + 2, d=nid + 3, e=nid + 4,
            spid=sh.shape_id,
            delay=0 if i == 0 else stagger_ms,
            node="clickEffect" if i == 0 else "afterEffect",
            dur=dur_ms,
        ))
        nid += 5

    xml = _TIMING_TEMPLATE.format(p=P_NS, groups="".join(groups))
    sld.append(etree.fromstring(xml))
