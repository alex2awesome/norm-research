"""a41 hybrid: Press assets and visual communication.

Criterion: complete, accessible media kit; clean downloadable assets with proper
branding; scannable structure; high-quality, accurately labeled visuals (captions,
credits) and clear data visualizations.

Judge behavior (train pack):
  * non-releases (news articles, nav chrome, stock-photo pages) -> ~0
  * genuine releases with NO visual/asset evidence -> also ~0 (Carrier, CAIR)
  * releases with partial asset signals (photo credits, 'press photos', footer
    media-kit links, PDF/infographic downloads) -> 0.25-0.35
  * releases offering a real media kit (hi-res downloads with file sizes, b-roll,
    labeled 'Photo - http://...' / 'Logo - http://...' asset links, press kits)
    -> 0.7-0.75

Design: score = release_gate * (0.87*asset_evidence + 0.13*scannable_structure).
PRESENCE IS NOT QUALITY: detectors require structural context (labeled asset URLs,
download verb + asset noun, credit/caption attribution, file sizes) rather than
bare keywords, so injected words like "photo video logo" do not fire. Stock-photo
agency / commerce chrome (Getty-style) is explicitly damped.
"""
import re

LLM_FIELDS = {
    "media_assets": ("List the downloadable press/media assets this document offers "
                     "journalists (hi-res photos, logo files, b-roll, press kit, "
                     "infographic, video, PDF), with formats/sizes if given; answer "
                     "NONE if none are offered."),
    "is_press_release": ("Answer YES if this document is a press release or official "
                         "newsroom announcement issued by an organization; answer NO "
                         "if it is a news article, blog, product page, or site "
                         "navigation chrome."),
}

_NONE_RE = re.compile(r"(?i)^\s*(none|n/?a|no\b|nothing|unknown|not\s)")

# ---------------- asset-evidence detectors (context-anchored) ----------------
# 'Photo -  http://...  Logo -  http://...LOGO' style labeled asset links
_LABELED_ASSET = re.compile(
    r"(?i)\b(photos?|logos?|images?|videos?|b-?roll|infographics?)\s*[-:]{1,2}\s*"
    r"(?:https?://|www\.)")
# inline '(Press Kit)' link label vs. generic 'media kit' mention
_KIT_LABELED = re.compile(r"(?i)\(\s*(?:press|media)\s+kit\s*\)")
_KIT = re.compile(r"(?i)\b(?:press|media)\s+kits?\b|\bbrand\s+(?:assets|center)\b")
_DL_HIRES = re.compile(r"(?i)\bdownload\s+(?:\w+\s+){0,2}?hi(?:gh)?[- ]?res(?:olution)?\b")
_HIRES = re.compile(r"(?i)\bhi(?:gh)?[- ]?res(?:olution)?\b")
_BROLL = re.compile(r"(?i)\bb-?roll\b")
# 'Download photos' / 'Download Earnings infographic' (visual asset nouns only)
_DL_VISUAL = re.compile(
    r"(?i)\bdownload\s+(?:\w+\s+){0,2}?(?:photos?|images?|logos?|videos?|"
    r"infographics?|b-?roll)\b")
_DL_PDF = re.compile(r"(?i)\bdownload\b[^.\n]{0,50}\bpdf\b")
# 'Press photos' nav / 'Photos(1)' asset counters
_PRESS_PHOTOS = re.compile(r"(?i)\bpress\s+photos\b|\bphotos?\s*\(\s*\d+\s*\)")
# captions & credits: 'Photo credit:', 'Caption:', 'Photo by Jack Duman', '(Photo: X)'
_CAPTION = re.compile(
    r"(?i)\b(?:photo|image|picture)\s+(?:credit|caption|courtesy)s?\b"
    r"|\bcaption\s*:"
    r"|\bphotos?\s+by\s+[A-Z]"
    r"|\(\s*photo\s*:\s*[^)]{3,80}\)"
    r"|\balt\s+text\b")
_FILESIZE = re.compile(r"(?i)\b\d+(?:\.\d+)?\s*[KM]B\b")
_IMG_FMT = re.compile(r"(?i)\bimage/(?:jpe?g|png|gif|tiff?)\b|\.(?:jpg|jpeg|png|tiff?|eps|zip)\b")
_ASSET_URL = re.compile(r"(?i)\bphotos?\.[a-z][\w-]*\.(?:com|net|org)\b")
# concrete downloadable-asset URLs (photo-host paths or image-file links)
_ASSET_URL_FULL = re.compile(
    r"(?i)(?:https?://)?photos?\.[a-z][\w-]*\.(?:com|net|org)/[\w/.\-]+"
    r"|https?://\S+\.(?:jpg|jpeg|png|zip|eps|tiff?)\b")
_WEBCAST = re.compile(r"(?i)\bwebcast\b|\byoutube\s+video\b|\bwatch\s+(?:the\s+)?video\b")
_INFOGRAPHIC = re.compile(r"(?i)\binfographic\b|\bdata\s+visuali[sz]ation\b")
_IMAGES_NAV = re.compile(r"(?i)\bimages\b")
_VIDEOS_NAV = re.compile(r"(?i)\bvideos\b")

# ---------------- release-gate detectors ----------------
_FIR = re.compile(r"(?i)\bfor\s+immediate\s+release\b")
_WIRE = re.compile(
    r"(?i)(/\s*PRNewswire\s*/?|\(BUSINESS\s+WIRE\)|GLOBE\s+NEWSWIRE|/\s*CNW\s*/|"
    r"ACCESSWIRE|Marketwired|\bnews\s+provided\s+by\b)")
_PR_PHRASE = re.compile(r"(?i)\b(?:press|news)\s+releases?\b")
_SOURCE_LINE = re.compile(r"\bSOURCE[: ]\s{0,3}[A-Z][\w&.,'\- ]{2,60}")
_DATELINE = re.compile(
    r"\b[A-Z]{3,}[A-Z .]{0,25},\s*(?:[A-Za-z.]{2,15},?\s+)?"
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}")
_TODAY_VERB = re.compile(r"(?i)\btoday\s+(?:announc|launch|report|introduc|unveil|releas)")
_NEWSROOM = re.compile(r"(?i)\bnews\s*room\b|\bpress\s+room\b")
_CONTACT_ANCHOR = re.compile(
    r"(?i)\b(?:media|press|news)\s+contacts?\b|\b(?:press|media)\s+(?:inquiries|enquiries)\b"
    r"|\bmedia\s+relations\b|\bpress\s+office\b|\bcontacts?\s*:")
_CONTACT_PR = re.compile(r"(?i)\bcontact\s+\w{0,20}\s*(?:PR|public\s+relations)\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]{2,}")
_PHONE = re.compile(r"(?<![\d\w])(\+?\d{1,4}(?:[ ().\-]{1,3}\d{2,4}){2,5})(?!\d)")
_ABOUT = re.compile(r"\bAbout\s+[A-Z][\w'&.-]+")
# stock-photo agency / commerce chrome -> hard damp (Getty hazard)
_STOCK_COMMERCE = re.compile(
    r"(?i)\broyalty[- ]free\b|\bgetty\s+images\b|\bpremium\s+access\b|\badd\s+to\s+cart\b"
    r"|\bshopping\s+cart\b|\bpurchase\s+history\b|\bembed(?:ded)?\s+im(?:age|ages)\b"
    r"|\ball\s+royalty\b")


def _contact_block(tail):
    """A real press-contact block (anchor + phone/email nearby) near the end."""
    for m in _CONTACT_ANCHOR.finditer(tail):
        win = tail[m.start():m.start() + 400]
        if _EMAIL.search(win) or _PHONE.search(win):
            return True
    return False


def _asset_score(t):
    a = 0.0
    labeled = {m.group(1).lower().rstrip("s") for m in _LABELED_ASSET.finditer(t)}
    a += 0.28 * min(len(labeled), 2)
    if _KIT_LABELED.search(t):
        a += 0.35
    elif _KIT.search(t):
        a += 0.28
    if _DL_HIRES.search(t):
        a += 0.30
    elif _HIRES.search(t):
        a += 0.10
    if _BROLL.search(t):
        a += 0.18
    if _DL_VISUAL.search(t):
        a += 0.18
    if _PRESS_PHOTOS.search(t):
        a += 0.15
    ncap = len(_CAPTION.findall(t))
    if ncap:
        a += 0.15 + (0.08 if ncap >= 2 else 0.0)
    if _FILESIZE.search(t):
        a += 0.10
    if _IMG_FMT.search(t) or _ASSET_URL.search(t):
        a += 0.08
    n_urls = len({m.group(0).lower() for m in _ASSET_URL_FULL.finditer(t)})
    if n_urls >= 2:
        a += 0.12  # multiple concrete asset links = a real downloadable kit
    if _DL_PDF.search(t):
        a += 0.15
    if _WEBCAST.search(t):
        a += 0.10
    if _INFOGRAPHIC.search(t):
        a += 0.06
    # newsroom asset hub: press kit(s) + Images + Videos nav co-present
    if _KIT.search(t) and _IMAGES_NAV.search(t) and _VIDEOS_NAV.search(t):
        a += 0.15
    return min(1.0, a)


def _gate_score(t, head, tail):
    g = 0.0
    if _FIR.search(t):
        g += 0.50
    if _WIRE.search(t):
        g += 0.45
    if _PR_PHRASE.search(t):
        g += 0.30
    if _SOURCE_LINE.search(tail):
        g += 0.30
    if _DATELINE.search(head):
        g += 0.30
    if _TODAY_VERB.search(head):
        g += 0.15
    if _contact_block(tail):
        g += 0.25
    if _ABOUT.search(tail):
        g += 0.12
    if _NEWSROOM.search(t):
        g += 0.08
    if _CONTACT_PR.search(t):
        g += 0.15
    return min(1.0, g)


def _structure_score(t, head, tail):
    s = 0.0
    if _FIR.search(t):
        s += 0.35
    if _DATELINE.search(head):
        s += 0.20
    if _contact_block(tail):
        s += 0.25
    if _ABOUT.search(tail):
        s += 0.15
    if re.search(r"(?i)\bprint\b", t) and re.search(r"(?i)\bshare\b", t):
        s += 0.05
    return min(1.0, s)


def score(text: str, extracted: dict, ops) -> float:
    try:
        t = ops.normalize(text or "")
        if len(t) < 50:
            return 0.02
        head, tail = t[:3500], t[-4500:]

        a = _asset_score(t)
        g = _gate_score(t, head, tail)
        s = _structure_score(t, head, tail)

        # ---- thick-input grounding (predicate stays in code) ----
        ex = extracted or {}
        assets_ans = (ex.get("media_assets") or "").strip()
        if len(assets_ans) >= 3 and not _NONE_RE.match(assets_ans):
            al = assets_ans.lower()
            ntypes = sum(1 for pat in (r"photo|image", r"logo", r"video|b-?roll",
                                       r"kit", r"infographic|chart", r"pdf")
                         if re.search(pat, al))
            bonus = 0.05 if re.search(r"hi(?:gh)?[- ]?res|b-?roll|press kit", al) else 0.0
            a = max(a, min(0.85, 0.30 + 0.10 * min(ntypes, 4) + bonus))
        rel_ans = (ex.get("is_press_release") or "").strip().lower()
        if rel_ans.startswith("yes"):
            g = max(g, 0.70)
        elif rel_ans.startswith("no"):
            g *= 0.35

        # stock-photo agency / commerce page: keyword soup is not a media kit
        if _STOCK_COMMERCE.search(t):
            g *= 0.25
        else:
            # strong concrete asset evidence itself implies a media-facing page
            # (e.g. university news item with credited, downloadable photos)
            if a >= 0.45:
                g = max(g, 0.75)
            elif a >= 0.30:
                g = max(g, 0.55)

        g = max(g, 0.04)
        val = g * (0.87 * a + 0.13 * s)
        return max(0.0, min(1.0, val))
    except Exception:
        return 0.5
