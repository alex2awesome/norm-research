"""
Webshare rotating-proxy helper.

Pulls the account's proxy list from the Webshare API and returns it in the
shape Playwright/Camoufox expects:

    [{"server": "http://IP:PORT", "username": U, "password": P}, ...]

API key lives in a one-line text file (default ~/.proxies-webshare-key.txt).

Usage (module):
    from webshare_proxies import fetch_proxy_list
    proxies = fetch_proxy_list()

Usage (CLI smoke test):
    python scripts/webshare_proxies.py            # prints count + first proxy
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

DEFAULT_KEY_PATH = Path.home() / ".proxies-webshare-key.txt"
API_URL = "https://proxy.webshare.io/api/v2/proxy/list/"


def _read_key(key_path: Optional[Path] = None) -> str:
    path = Path(key_path) if key_path else DEFAULT_KEY_PATH
    key = path.read_text().strip()
    if not key:
        raise RuntimeError(f"Empty Webshare API key at {path}")
    return key


def fetch_proxy_list(
    key_path: Optional[Path] = None,
    page_size: int = 100,
    max_pages: int = 20,
    timeout_s: float = 30.0,
) -> List[Dict[str, str]]:
    """Fetch all direct-mode proxies for the account (paginated)."""
    key = _read_key(key_path)
    proxies: List[Dict[str, str]] = []
    for page in range(1, max_pages + 1):
        qs = urllib.parse.urlencode(
            {"mode": "direct", "page": page, "page_size": page_size}
        )
        req = urllib.request.Request(
            f"{API_URL}?{qs}", headers={"Authorization": f"Token {key}"}
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for r in data.get("results", []):
            if not r.get("valid", True):
                continue
            proxies.append(
                {
                    "server": f"http://{r['proxy_address']}:{r['port']}",
                    "username": r["username"],
                    "password": r["password"],
                    "country": r.get("country_code", "?"),
                }
            )
        if not data.get("next"):
            break
    if not proxies:
        raise RuntimeError("Webshare returned zero valid proxies")
    return proxies


def main() -> int:
    key_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    proxies = fetch_proxy_list(key_path)
    print(f"{len(proxies)} valid proxies")
    first = dict(proxies[0])
    first["password"] = "***"
    print("first:", json.dumps(first))
    return 0


if __name__ == "__main__":
    sys.exit(main())
