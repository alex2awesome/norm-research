#!/usr/bin/env python3
"""Discover OARD rejection encodings: distinct values of action_type / action_subtype
and how many distinct applications fall into each.
"""
from google.cloud import bigquery

client = bigquery.Client(project="usc-research")

print("=== action_type distribution (rejections table) ===")
q = """
SELECT action_type, COUNT(*) AS n_rows, COUNT(DISTINCT app_id) AS n_apps
FROM `patents-public-data.uspto_oce_office_actions.rejections`
GROUP BY action_type
ORDER BY n_apps DESC
"""
for r in client.query(q).result():
    print(f"  {str(r.action_type)[:50]:50s} | rows={r.n_rows:>10,d} | apps={r.n_apps:>10,d}")

print()
print("=== action_subtype distribution (rejections table) ===")
q = """
SELECT action_subtype, COUNT(*) AS n_rows, COUNT(DISTINCT app_id) AS n_apps
FROM `patents-public-data.uspto_oce_office_actions.rejections`
GROUP BY action_subtype
ORDER BY n_apps DESC
LIMIT 50
"""
for r in client.query(q).result():
    print(f"  {str(r.action_subtype)[:60]:60s} | rows={r.n_rows:>10,d} | apps={r.n_apps:>10,d}")

print()
print("=== sample joint (action_type, action_subtype) ===")
q = """
SELECT action_type, action_subtype, COUNT(*) AS n_rows, COUNT(DISTINCT app_id) AS n_apps
FROM `patents-public-data.uspto_oce_office_actions.rejections`
GROUP BY action_type, action_subtype
ORDER BY n_apps DESC
LIMIT 30
"""
for r in client.query(q).result():
    a = str(r.action_type)[:30]
    s = str(r.action_subtype)[:50]
    print(f"  type={a:30s}  sub={s:50s} | apps={r.n_apps:>10,d}")
