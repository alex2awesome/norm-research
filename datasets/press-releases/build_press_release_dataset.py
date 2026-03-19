import hashlib
import math
import os
import sqlite3
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

os.environ.setdefault("NPY_DISABLE_MAC_OS_CHECK", "1")

import pandas as pd
from datasets import load_from_disk
import tldextract


RAW_DATA_DIR = Path("raw_data")
FORWARD_SOURCE = "Press Releases -> News Articles"
BACKWARD_SOURCE = "Press Releases <- News Articles"


def normalize_url(raw: Optional[str]) -> Optional[str]:
    """Normalize URLs so we can match them across data sources."""
    if not isinstance(raw, str):
        return None
    url = raw.strip()
    if not url:
        return None
    if "://" in url:
        url = url.split("://", 1)[1]
    url = url.split("#", 1)[0]
    if url.endswith("/"):
        url = url[:-1]
    return url.lower()


def surt_to_standard_url(raw: str) -> str:
    if ")/" not in raw:
        url = raw.strip()
        if not url:
            return url
        if "://" in url:
            return url
        return f"https://{url}"
    first_split = raw.find(")/")
    domain_part = raw[:first_split]
    last_split = raw.rfind(")/")
    if last_split > first_split:
        path = raw[last_split + 2 :]
    else:
        path = raw[first_split + 2 :]
    domain_tokens = [token for token in domain_part.replace("(", "").replace(")", "").split(",") if token]
    if not domain_tokens:
        domain = domain_part.replace(")", "").replace("(", "")
    else:
        domain = ".".join(reversed(domain_tokens))
    path = path.lstrip("/")
    if path:
        return f"https://{domain}/{path}"
    return f"https://{domain}"


def ensure_scheme(raw: Optional[str]) -> Optional[str]:
    if not isinstance(raw, str) or not raw:
        return None
    url = raw.strip()
    if "://" not in url:
        if ")/" in url:
            return surt_to_standard_url(url)
        url = f"https://{url}"
    return url


def empty_to_none(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    return value


def extract_domain_and_subdomain(raw: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    cleaned = empty_to_none(raw)
    if not cleaned:
        return (None, None)
    candidate = cleaned.strip()
    ensured = ensure_scheme(candidate)
    target = ensured or candidate
    try:
        extracted = tldextract.extract(target)
    except Exception:
        extracted = None
    if extracted:
        registered = ".".join(part for part in (extracted.domain, extracted.suffix) if part)
        subdomain = extracted.subdomain or None
        if registered or subdomain:
            return (registered or None, subdomain)
    try:
        parsed = urlparse(target)
    except ValueError:
        return (None, None)
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return (netloc or None, None)


def extract_domain(raw: Optional[str]) -> Optional[str]:
    domain, _ = extract_domain_and_subdomain(raw)
    return domain


def extract_subdomain(raw: Optional[str]) -> Optional[str]:
    _, subdomain = extract_domain_and_subdomain(raw)
    return subdomain


def ms_to_iso_str(value: Optional[float]) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        return pd.to_datetime(value, unit="ms", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, OverflowError, TypeError):
        return None


def sha1_text(value: Optional[str]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def first_non_null(series: pd.Series) -> Optional[str]:
    for item in series:
        cleaned = empty_to_none(item)
        if cleaned is not None:
            return cleaned
    return None


def build_forward_df() -> pd.DataFrame:
    csv_path = RAW_DATA_DIR / "article-to-pr-mapper.csv.gz"
    mapper = pd.read_csv(
        csv_path,
        usecols=[
            "URL",
            "Target URL",
            "Date First Seen",
            "company_name",
            "news_url_domain",
        ],
        dtype=str,
    ).rename(
        columns={
            "URL": "news_article_url",
            "Target URL": "press_release_url",
            "Date First Seen": "date_first_seen",
            "company_name": "press_release_company",
            "news_url_domain": "news_article_source",
        }
    )
    mapper["news_norm"] = mapper["news_article_url"].map(normalize_url)
    mapper["press_norm"] = mapper["press_release_url"].map(normalize_url)
    mapper["news_article_text"] = None
    mapper["press_release_text"] = None
    mapper["news_article_date"] = None
    mapper["press_release_date"] = None
    mapper["press_release_company"] = mapper["press_release_company"].map(empty_to_none)
    source_parts = mapper["news_article_source"].map(extract_domain_and_subdomain)
    mapper["news_article_source"] = source_parts.map(lambda part: part[0])
    mapper["news_article_subdomain"] = source_parts.map(lambda part: part[1])

    news_index_map: Dict[str, list] = dict(mapper.groupby("news_norm").indices)
    press_index_map: Dict[str, list] = dict(mapper.groupby("press_norm").indices)
    required_urls = set(news_index_map.keys()) | set(press_index_map.keys())

    dataset_path = RAW_DATA_DIR / "all-coref-resolved"
    dataset = load_from_disk(str(dataset_path)).select_columns(["article_url", "article_text", "target_timestamp"])
    for row in dataset:
        norm = normalize_url(row["article_url"])
        if not norm or norm not in required_urls:
            continue
        iso_timestamp = ms_to_iso_str(row["target_timestamp"])
        if norm in news_index_map:
            idx = news_index_map.pop(norm)
            mapper.loc[list(idx), "news_article_text"] = row["article_text"]
            mapper.loc[list(idx), "news_article_date"] = iso_timestamp
        if norm in press_index_map:
            idx = press_index_map.pop(norm)
            mapper.loc[list(idx), "press_release_text"] = row["article_text"]
            mapper.loc[list(idx), "press_release_date"] = iso_timestamp
        if norm not in news_index_map and norm not in press_index_map:
            required_urls.discard(norm)
        if not required_urls:
            break

    fallback_dt = pd.to_datetime(mapper["date_first_seen"], errors="coerce", utc=True)
    fallback_str = fallback_dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    fallback_str = fallback_str.where(~fallback_dt.isna(), None)
    mapper["news_article_date"] = mapper["news_article_date"].fillna(fallback_str)

    forward_df = mapper[
        [
            "press_release_text",
            "news_article_text",
            "press_release_url",
            "news_article_url",
            "press_release_date",
            "news_article_date",
            "press_release_company",
            "news_article_source",
            "news_article_subdomain",
        ]
    ].copy()
    forward_df["press_release_text"] = forward_df["press_release_text"].map(empty_to_none)
    forward_df = forward_df.dropna(subset=["press_release_text"])
    forward_df["press_release_url"] = forward_df["press_release_url"].map(ensure_scheme)
    forward_df["news_article_url"] = forward_df["news_article_url"].map(ensure_scheme)
    forward_df["was_covered_by_news_article"] = True
    forward_df["source"] = FORWARD_SOURCE
    return forward_df[
        [
            "press_release_text",
            "was_covered_by_news_article",
            "news_article_text",
            "news_article_url",
            "news_article_source",
            "news_article_subdomain",
            "press_release_url",
            "press_release_company",
            "press_release_date",
            "news_article_date",
            "source",
        ]
    ]


def build_backward_df() -> pd.DataFrame:
    con = sqlite3.connect(RAW_DATA_DIR / "article_to_press_release_data.db")
    query = """
        SELECT
            p.data as press_release_text,
            a.article_text as news_article_text,
            p.article_url as press_release_url,
            COALESCE(m.target_article_url, h.article_url) as news_article_url,
            p.target_timestamp as press_release_date,
            a.canonical_timestamp as news_article_date,
            a.canonical_domain as news_article_source,
            NULL as press_release_company
        FROM article_to_href h
        JOIN press_release_data p ON p.article_url = h.href
        JOIN article_data a ON a.common_crawl_url = h.article_url
        LEFT JOIN article_map m ON m.archival_url = h.article_url
        WHERE h.is_press_release = 1
    """
    backward_df = pd.read_sql_query(query, con)
    con.close()
    backward_df["press_release_text"] = backward_df["press_release_text"].map(empty_to_none)
    backward_df = backward_df.dropna(subset=["press_release_text"])
    backward_df["press_release_url"] = backward_df["press_release_url"].map(ensure_scheme)
    backward_df["news_article_url"] = backward_df["news_article_url"].map(ensure_scheme)
    press_dt = pd.to_datetime(backward_df["press_release_date"], errors="coerce", utc=True)
    news_dt = pd.to_datetime(backward_df["news_article_date"], errors="coerce", utc=True)
    backward_df["press_release_date"] = press_dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ").where(~press_dt.isna(), None)
    backward_df["news_article_date"] = news_dt.dt.strftime("%Y-%m-%dT%H:%M:%SZ").where(~news_dt.isna(), None)
    backward_df["press_release_company"] = backward_df["press_release_company"].map(empty_to_none)
    source_parts = backward_df["news_article_source"].map(extract_domain_and_subdomain)
    backward_df["news_article_source"] = source_parts.map(lambda part: part[0])
    backward_df["news_article_subdomain"] = source_parts.map(lambda part: part[1])
    backward_df["was_covered_by_news_article"] = True
    backward_df["source"] = BACKWARD_SOURCE
    return backward_df[
        [
            "press_release_text",
            "was_covered_by_news_article",
            "news_article_text",
            "news_article_url",
            "news_article_source",
            "news_article_subdomain",
            "press_release_url",
            "press_release_company",
            "press_release_date",
            "news_article_date",
            "source",
        ]
    ]


def build_entity_outputs(combined: pd.DataFrame) -> None:
    print("Cleaning column values...")
    combined["press_release_url"] = combined["press_release_url"].map(empty_to_none)
    combined["news_article_url"] = combined["news_article_url"].map(empty_to_none)
    combined["news_article_text"] = combined["news_article_text"].map(empty_to_none)
    combined["news_article_date"] = combined["news_article_date"].map(empty_to_none)
    combined["press_release_company"] = combined["press_release_company"].map(empty_to_none)
    if "news_article_subdomain" not in combined.columns:
        combined["news_article_subdomain"] = None
    combined["news_article_source"] = combined["news_article_source"].map(empty_to_none)
    combined["news_article_subdomain"] = combined["news_article_subdomain"].map(empty_to_none)
    missing_company = combined["press_release_company"].isna() & combined["press_release_url"].notna()
    combined.loc[missing_company, "press_release_company"] = combined.loc[missing_company, "press_release_url"].map(
        extract_domain
    )
    missing_source = combined["news_article_source"].isna() & combined["news_article_url"].notna()
    combined.loc[missing_source, "news_article_source"] = combined.loc[missing_source, "news_article_url"].map(
        extract_domain
    )
    missing_subdomain = combined["news_article_subdomain"].isna() & combined["news_article_url"].notna()
    combined.loc[missing_subdomain, "news_article_subdomain"] = combined.loc[
        missing_subdomain, "news_article_url"
    ].map(extract_subdomain)

    print("Deduplicating press releases...")
    press_key = combined["press_release_url"].copy()
    missing_press_url = press_key.isna()
    if missing_press_url.any():
        fallback_press_hash = combined.loc[missing_press_url, "press_release_text"].map(sha1_text)
        press_key.loc[missing_press_url] = fallback_press_hash.map(lambda h: f"text::{h}")
    combined["press_key"] = press_key
    press_release_entities = (
        combined.groupby("press_key", dropna=False)
        .agg(
            press_release_text=("press_release_text", first_non_null),
            press_release_url=("press_release_url", first_non_null),
            press_release_date=("press_release_date", first_non_null),
            press_release_company=("press_release_company", first_non_null),
        )
        .reset_index()
    )
    press_release_entities.insert(0, "press_release_id", range(len(press_release_entities)))
    combined = combined.merge(
        press_release_entities[["press_key", "press_release_id"]],
        on="press_key",
        how="left",
    )
    press_release_entities.drop(columns=["press_key"], inplace=True)
    press_release_entities = press_release_entities[
        [
            "press_release_id",
            "press_release_text",
            "press_release_url",
            "press_release_date",
            "press_release_company",
        ]
    ]

    print("Deduplicating news articles...")
    news_key = combined["news_article_url"].copy()
    missing_url_mask = news_key.isna()
    if missing_url_mask.any():
        fallback_hash = combined.loc[missing_url_mask, "news_article_text"].map(sha1_text)
        fallback_hash = fallback_hash.map(lambda h: f"text::{h}" if h else None)
        still_missing = fallback_hash.isna()
        if still_missing.any():
            fallback_hash.loc[still_missing] = [
                f"row::{idx}" for idx in fallback_hash.index[still_missing]
            ]
        news_key.loc[missing_url_mask] = fallback_hash
    combined["news_key"] = news_key
    news_entities = (
        combined.groupby("news_key", dropna=False)
        .agg(
            news_article_text=("news_article_text", first_non_null),
            news_article_url=("news_article_url", first_non_null),
            news_article_date=("news_article_date", first_non_null),
            news_article_source=("news_article_source", first_non_null),
            news_article_subdomain=("news_article_subdomain", first_non_null),
        )
        .reset_index()
    )
    news_entities.insert(0, "news_article_id", range(len(news_entities)))
    combined = combined.merge(
        news_entities[["news_key", "news_article_id"]],
        on="news_key",
        how="left",
    )
    news_entities.drop(columns=["news_key"], inplace=True)
    news_entities = news_entities[
        [
            "news_article_id",
            "news_article_text",
            "news_article_url",
            "news_article_date",
            "news_article_source",
            "news_article_subdomain",
        ]
    ]

    print("Writing split outputs...")
    mapper_df = combined[
        [
            "press_release_id",
            "news_article_id",
            "was_covered_by_news_article",
            "source",
        ]
    ].drop_duplicates(ignore_index=True)

    press_release_entities.to_csv("press_releases.csv", index=False)
    news_entities.to_csv("news_articles.csv", index=False)
    mapper_df.to_csv("press_release_news_mappings.csv", index=False)


def main():
    print("Loading forward-direction pairs...")
    forward_df = build_forward_df()
    print("Loading backward-direction pairs...")
    backward_df = build_backward_df()
    print("Combining datasets...")
    combined = pd.concat([forward_df, backward_df], ignore_index=True)
    print("Building normalized entity tables...")
    build_entity_outputs(combined)

if __name__ == "__main__":
    main()
