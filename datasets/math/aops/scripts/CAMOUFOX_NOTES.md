# AoPS Camoufox Crawler — Install, Rotation Behavior, Pilot Results

Replacement for `aops_bulk_crawl.py` (Chromium + playwright-stealth), which has
been 100% Cloudflare-blocked from sk3 since 2026-05-30. New stack:
**Camoufox** (anti-detect Firefox fork, Playwright-compatible API) +
**Webshare rotating datacenter proxies**.

## Files

| File | Purpose |
|---|---|
| `aops_camoufox_crawl.py` | Crawler. Same CLI + output contract as `aops_bulk_crawl.py`. |
| `webshare_proxies.py` | Pulls the proxy list from the Webshare API. |

## Install (sk3)

```bash
# Pin HOME to /lfs first if running under nohup (AFS token expiry!)
export HOME=/lfs/skampere3/0/spangher   # adjust to the usual /lfs home

pip install -U "camoufox[geoip]"
python -m camoufox fetch        # downloads browser (~150MB) + GeoIP db (~44MB) into ~/.cache/camoufox

# Webshare API key (one line, no newline issues):
#   ~/.proxies-webshare-key.txt   (i.e. under the pinned HOME)
# Copy it from the laptop: scp ~/.proxies-webshare-key.txt sk3:<pinned-home>/

# Smoke test the key:
python scripts/webshare_proxies.py     # prints "100 valid proxies" + first entry
```

Notes for sk3:
- Camoufox is Firefox-based; headless works without X. If it ever complains
  about a display, use `headless='virtual'` (needs `Xvfb` installed) — not
  needed on the laptop pilot.
- The browser+GeoIP download goes to `$HOME/.cache/camoufox` (Linux), so the
  pinned /lfs HOME keeps it off AFS.

## Usage (identical contract to aops_bulk_crawl.py)

```bash
cd datasets/math/aops

# Pilot
python scripts/aops_camoufox_crawl.py --start 495500 --end 495530 --shard camoufox_pilot

# Full run, 4 workers, modulo-sharded (each worker = 1 browser process, no GPU)
for w in 0 1 2 3; do
  nohup python scripts/aops_camoufox_crawl.py \
    --start 1 --end 1000000 --shard full \
    --worker $w --num-workers 4 --delay 2.0 \
    > logs/full_w${w}.nohup 2>&1 &
done
```

Outputs (same as v1, resumable — re-running skips ids in the `.done` ledger):

```
raw/shards/<shard>__w<worker>.jsonl.gz   # {"topic_id": N, "response": {...}} per line
raw/shards/<shard>__w<worker>.done       # completed topic ids
logs/<shard>__w<worker>.log              # progress + rotation log
```

New flags vs v1:

| Flag | Default | Meaning |
|---|---|---|
| `--proxy-index N` | `worker % n_proxies` | Starting proxy in the Webshare list |
| `--no-proxy` | off | Direct connection (no Webshare) |
| `--key-path` | `~/.proxies-webshare-key.txt` | Webshare API key file |
| `--cf-threshold` | 3 | Consecutive CF failures before rotating |
| `--max-rotations` | 200 | Abort guard if every proxy is burned |

## Proxy rotation behavior

- Proxy list is fetched once at startup from
  `GET https://proxy.webshare.io/api/v2/proxy/list/?mode=direct` (paginated;
  invalid entries dropped). 100 proxies on the account as of 2026-06-10.
- Each worker starts at proxy `worker % n_proxies` (so parallel workers spread
  across IPs) unless `--proxy-index` is given.
- A **Cloudflare failure** is: ajax HTTP 403/429/503, a 200 whose body is a
  "Just a moment…" challenge page, or a session-warm stuck on a challenge.
  Each failure re-warms `/community` and retries the same topic id.
- After `--cf-threshold` consecutive CF failures, or any browser/page crash
  (`Page crashed`, proxy connection death): **close browser → advance to next
  proxy (wraps around) → relaunch → re-warm → retry the same topic id.**
  Every rotation is logged (`ROTATE #k [CF|BROWSER]: old → new`).
- `warm_session` waits for both the `AoPS.session` id **and** the
  `cf_clearance` cookie (8s grace) before declaring the session warm, because
  the HTML page can render before CF has cleared the XHR path.
- Politeness: 2.0s between requests (per worker), unchanged from v1.

## Pilot results (laptop, 2026-06-10)

Probe of the ajax endpoint per connection type (topic 495500):

| Connection | sid | cf_clearance | ajax | Verdict |
|---|---|---|---|---|
| direct (laptop IP) | Y | Y | 200 JSON | works |
| proxy[0] AR 43.229.11.167 | Y | — | page crash | bad proxy (browser dies) |
| proxy[7] US 198.23.147.198 | Y | N | 403 challenge | CF-burned IP |
| proxy[23] GB 107.181.132.129 | Y | Y | 403 challenge | CF-burned IP (clearance not sufficient!) |
| proxy[51] CL 2.57.31.160 | Y | Y | 200 JSON | works |

Key findings:
- Camoufox itself passes CF (the old Chromium+stealth stack did not) — the
  remaining variable is purely **IP reputation** of each datacenter proxy.
  Roughly half the Webshare IPs are pre-burned at AoPS; rotation finds clean
  ones automatically.
- `cf_clearance` alone does not guarantee XHR success (proxy[23]); per-IP CF
  scoring 403s the POST anyway → hence rotate-on-403 rather than
  wait-for-cookie as the primary strategy.
- 30-topic pilot (`--start 495500 --end 495530 --shard camoufox_pilot`,
  through Webshare, rotation starting at proxy[0]): **ok=30 err=0
  rotations=1** in ~80s. proxy[0] (AR) crashed the page → BROWSER-rotated to
  proxy[1] (ES), which served all 30 topics. Shard validation: 30/30 lines
  parse as JSON, 19 topics with posts (251 posts total, 1–151 per topic),
  11 empty/deleted topics (expected for arbitrary id ranges). Resume
  verified: immediate exit with `remaining=0` on re-run.
- The embedded `AoPS.session` id can be identical across browsers/proxies
  (CDN-cached /community HTML). Harmless — ajax.php accepts it.

## Gotchas

- **Bad proxies crash the page**, not just 403 (`Page.goto: Page crashed`).
  The crawler treats any Playwright error escaping the per-topic handlers as a
  rotation event — don't "fix" this by retrying in place.
- `geoip=True` requires the `camoufox[geoip]` extra **and** the GeoIP database
  downloaded by `python -m camoufox fetch`. Without it, proxy + default
  fingerprint mismatch (timezone/locale vs exit IP) raises CF suspicion.
- `python -m camoufox fetch` is idempotent; re-run it after any
  `pip install -U camoufox` (browser build is version-locked).
- Webshare API: `Authorization: Token <key>` (not `Bearer`).
- One Camoufox instance ≈ 400-600MB RAM; 4 workers is fine on sk3 CPU.
