"""
Captcha-aware book downloader for pdfcoffee.com / dokumen.pub / similar.

Uses Playwright (headless Chromium) to drive the page + Capsolver to solve
the embedded reCAPTCHA v2 / Cloudflare Turnstile challenges. Once solved,
waits through the queue timer and captures the PDF download.

Usage:
  python capsolver_book_downloader.py <url> [<url> ...]    # download given URLs
  python capsolver_book_downloader.py --batch              # process the embedded TARGETS list

Outputs PDFs into ~/Downloads/capsolver_books/.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from playwright.sync_api import sync_playwright, TimeoutError as PWTimeout


CAPSOLVER_KEY = (Path.home() / ".cap-solver-key.txt").read_text().strip()
DOWNLOADS = Path.home() / "Downloads" / "capsolver_books"
DOWNLOADS.mkdir(parents=True, exist_ok=True)
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
      "(KHTML, like Gecko) Version/17.0 Safari/605.1.15")


# Pre-discovered candidate URLs to try in order of preference per book.
# Most pdfcoffee URLs follow a slug-based pattern; we discovered some via WebSearch.
TARGETS = {
    # pdfcoffee
    "garner_winning_brief":              "https://pdfcoffee.com/winning-brief-tips-1-100-pdf-pdf-free.html",
    "mellinkoff_language_of_the_law":    "https://pdfcoffee.com/language-of-the-law-pdf-free.html",
    "mauet_trial_techniques":            "https://pdfcoffee.com/trial-techniques-and-trials-mauet-trial-ad-book-4-pdf-free.html",
    "faber_patent_claim_drafting":       "https://pdfcoffee.com/landis-on-mechanics-of-patent-claim-drafting-2000-pdf-free.html",
    # dokumen.pub (patent canonicals)
    "khan_american_patent_law_history":  "https://dokumen.pub/american-patent-law-a-business-and-economic-history-1009123416-9781009123419.html",
    "concepts_of_property_in_ip":        "https://dokumen.pub/concepts-of-property-in-intellectual-property-law-9781107421547-9781107041820.html",
    "european_patent_law_upc_epc":       "https://dokumen.pub/european-patent-law-the-unified-patent-court-and-the-european-patent-convention-9783110781687-9783110774016.html",
    "nolo_ip_desk_reference":            "https://dokumen.pub/patent-copyright-et-trademark-an-intellectual-property-desk-reference-16nbsped-9781413327762-1413327761.html",
    "fundamentals_of_patent_law":        "https://dokumen.pub/fundamentals-of-patent-law-interpretation-and-scope-of-protection-9781472564061-9781841136929.html",
    "users_guide_to_patents_uk":         "https://dokumen.pub/a-users-guide-to-patents-9781526508706-9781526508683-9781526508713.html",
}

# Map each book slug to which task dir its PDF should land in
TASK_OF = {
    "garner_winning_brief":              "legal-outcome-prediction",
    "mellinkoff_language_of_the_law":    "legal-outcome-prediction",
    "mauet_trial_techniques":            "legal-outcome-prediction",
    "inside_jokes_hurley_dennett":       "humor",
    "boorstin_image_pseudo_events":      "press-releases",
    "sigal_reporters_and_officials":     "press-releases",
    "faber_patent_claim_drafting":       "patents",
    "mancosu_philosophy_math_practice":  "math-stackexchange",
    # dokumen.pub patents
    "khan_american_patent_law_history":  "patents",
    "concepts_of_property_in_ip":        "patents",
    "european_patent_law_upc_epc":       "patents",
    "nolo_ip_desk_reference":            "patents",
    "fundamentals_of_patent_law":        "patents",
    "users_guide_to_patents_uk":         "patents",
}

# Site reCAPTCHA v2 site keys (same across all docs on each host)
PDFCOFFEE_SITEKEY = "6Le_trkUAAAAAEg6edIIuGuFzUY3ruFn6NY9LK-S"
DOKUMEN_SITEKEY   = "6Ld9bsMUAAAAAFUJ3kb3qtJCvEbX7XEDp18HE5iQ"


def solve_recaptcha_v2(website_url: str, sitekey: str, timeout: int = 180) -> Optional[str]:
    """Submit a reCAPTCHA v2 task to Capsolver, poll until solved. Returns the gRecaptchaResponse token."""
    create = requests.post("https://api.capsolver.com/createTask", json={
        "clientKey": CAPSOLVER_KEY,
        "task": {
            "type": "ReCaptchaV2TaskProxyLess",
            "websiteURL": website_url,
            "websiteKey": sitekey,
        }
    }, timeout=30).json()
    if create.get("errorId") != 0:
        print(f"    capsolver createTask error: {create}")
        return None
    task_id = create["taskId"]
    print(f"    capsolver task created: {task_id}")
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(5)
        result = requests.post("https://api.capsolver.com/getTaskResult", json={
            "clientKey": CAPSOLVER_KEY, "taskId": task_id,
        }, timeout=30).json()
        status = result.get("status")
        if status == "ready":
            return result["solution"]["gRecaptchaResponse"]
        if result.get("errorId", 0) != 0:
            print(f"    capsolver error: {result}")
            return None
        print(f"    ... still solving (status={status})")
    print("    capsolver timeout")
    return None


def download_one(url: str, slug: str, headless: bool = True) -> Optional[Path]:
    """Drive landing → qdownload → captcha → wait → download.

    Works for pdfcoffee.com and dokumen.pub. Both share the same flow but
    dokumen.pub redirects /qdownload/ to landing unless reached by clicking
    DOWNLOAD FILE on the landing page (server checks Referer / sets a cookie).
    So we always click the button rather than navigate directly.
    """
    print(f"\n=== {slug}\n   url: {url}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        ctx = browser.new_context(user_agent=UA, accept_downloads=True)
        page = ctx.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
        except PWTimeout:
            print(f"   page load timeout"); browser.close(); return None
        page.wait_for_timeout(2000)

        # Skip if the page is a 404 / not found
        title = (page.title() or "").lower()
        if "404" in title or "not found" in page.content().lower()[:1500]:
            print(f"   404 / not found"); browser.close(); return None

        # Find DOWNLOAD FILE link on landing page
        btn = page.query_selector("a:has-text('DOWNLOAD FILE')") or page.query_selector("a[href*='/download/']")
        if not btn:
            print(f"   no DOWNLOAD FILE link found"); browser.close(); return None
        print(f"   landing button href: {btn.get_attribute('href')}")
        # Click it (server-side redirect/cookie sets up the queue page)
        try:
            btn.click()
        except Exception as e:
            print(f"   click error: {e}"); browser.close(); return None
        page.wait_for_timeout(3000)
        try:
            page.wait_for_load_state("domcontentloaded", timeout=15000)
        except PWTimeout:
            pass
        qurl = page.url
        print(f"   queue url: {qurl}")

        # Find reCAPTCHA sitekey on the queue page
        sitekey_el = page.query_selector("[data-sitekey]")
        sitekey = sitekey_el.get_attribute("data-sitekey") if sitekey_el else (
            DOKUMEN_SITEKEY if "dokumen.pub" in qurl else PDFCOFFEE_SITEKEY
        )
        print(f"   sitekey: {sitekey}")

        # Solve via capsolver
        token = solve_recaptcha_v2(qurl, sitekey)
        if not token:
            print(f"   captcha solve failed"); browser.close(); return None
        print(f"   token len: {len(token)} (first 40: {token[:40]}...)")

        # Inject the token into the page and trigger any onchange/callback
        page.evaluate(f"""
            (() => {{
                const tas = document.querySelectorAll('textarea[id*="g-recaptcha-response"], textarea[name="g-recaptcha-response"]');
                tas.forEach(t => {{ t.value = {json.dumps(token)}; t.style.display = 'block'; }});
                if (typeof captchaSolved === 'function') captchaSolved({json.dumps(token)});
                if (typeof onCaptchaSuccess === 'function') onCaptchaSuccess({json.dumps(token)});
                if (typeof grecaptcha !== 'undefined' && grecaptcha.getResponse) {{
                    tas.forEach(t => t.dispatchEvent(new Event('change', {{bubbles:true}})));
                }}
                return tas.length;
            }})()
        """)
        print(f"   token injected; waiting for queue timer + download...")

        # Wait for download to start. dokumen says "116 seconds download finish",
        # so allow a generous window.
        try:
            with page.expect_download(timeout=180000) as dl_info:
                # Prefer DOWNLOAD PDF if present (dokumen has PDF/DOCX/PPTX trio)
                pdf_btn = (page.query_selector("a:has-text('DOWNLOAD PDF')")
                           or page.query_selector("button:has-text('DOWNLOAD PDF')"))
                if pdf_btn:
                    try: pdf_btn.click()
                    except: pass
                # Fallback to a generic submit
                submit = page.query_selector("button[type='submit'], input[type='submit']")
                if submit:
                    try: submit.click()
                    except: pass
            d = dl_info.value
            target = DOWNLOADS / f"{slug}_{d.suggested_filename}"
            d.save_as(str(target))
            sz = target.stat().st_size
            print(f"   ✅ downloaded: {target}  ({sz/1e6:.2f} MB)")
            browser.close()
            return target
        except PWTimeout:
            print(f"   ❌ no download after 180s; current URL: {page.url}")
            browser.close()
            return None


def main():
    args = sys.argv[1:]
    if not args or args == ["--batch"]:
        targets = list(TARGETS.items())
    else:
        targets = [(re.sub(r'\W+', '_', a)[:60], a) for a in args]
    results = []
    for slug, url in targets:
        try:
            r = download_one(url, slug, headless=True)
            results.append((slug, url, r))
        except Exception as e:
            print(f"\n=== {slug} CRASHED: {e}")
            results.append((slug, url, None))
    print("\n========== SUMMARY ==========")
    for slug, url, path in results:
        marker = "✅" if path else "❌"
        size = f"{path.stat().st_size/1e6:.1f}MB" if path else "—"
        print(f"  {marker}  {slug:45s}  {size:>8s}")


if __name__ == "__main__":
    main()
