#!/usr/bin/env python3
"""Check if r/Jokes has reply comments that explain jokes."""
import requests
import json

# Get a popular joke
r = requests.get("https://arctic-shift.photon-reddit.com/api/posts/search",
    params={"subreddit": "Jokes", "limit": 3, "sort": "desc", "sort_type": "num_comments"},
    timeout=15)
posts = r.json()["data"]

for p in posts[:2]:
    post_id = p["id"]
    print("=== Joke (score=%s, %d comments) ===" % (p["score"], p.get("num_comments", 0)))
    print("Title: %s" % p["title"][:100])
    print("Body: %s" % p.get("selftext", "")[:200])

    # Get top-level comments
    r2 = requests.get("https://arctic-shift.photon-reddit.com/api/comments/search",
        params={"link_id": "t3_" + post_id, "limit": 10}, timeout=15)
    comments = r2.json()["data"]

    # Sort by score
    comments.sort(key=lambda c: c.get("score", 0), reverse=True)

    print("\nTop comments:")
    for c in comments[:5]:
        is_toplevel = c.get("parent_id", "").startswith("t3_")
        level = "TOP" if is_toplevel else "reply"
        body = c.get("body", "").replace("\n", " ")[:150]
        author = c.get("author", "")
        if author == "AutoModerator":
            continue
        print("  [%s] score=%s: %s" % (level, c.get("score", 0), body))
    print()
