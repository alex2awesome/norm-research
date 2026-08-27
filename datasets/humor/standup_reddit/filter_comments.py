"""Filter sampled threads from r/StandUpWorkshop for substantive (critique-bearing) comments.

Filtering rationale (per dataset-build task spec):
- length >= 100 chars (drops one-liner riffs like "lol same" / "i agree")
- in StandUpWorkshop subreddit (workshop nature → critique-heavy)
- reply depth >= 1 (replies to OP rather than top-level independent jokes;
  on Workshop the top-level commenter is usually the critic and OP is the joke author —
  so depth >= 1 captures *replies from the critic to OP* OR *clarification chains*; we keep
  TOP-LEVEL comments too since they ARE the primary critique on Workshop)
- exclude bot patterns / [deleted] / [removed] authors
- exclude AutoModerator

Final unit (decision: option **a** — post + concatenated comments): one row per thread,
text = post title + selftext + all surviving comments in threaded order.

We also emit one row per comment for inspection in filtered_comments.jsonl
(per spec wording: 'each row is {thread_id, comment_id, parent_id, body, score, author, ts}').
"""
import json, re, sys, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "sampled_threads.jsonl")
OUT_COMMENTS = os.path.join(HERE, "filtered_comments.jsonl")
OUT_THREADS = os.path.join(HERE, "filtered_threads.jsonl")

MIN_LEN = 100
BAD_AUTHORS = {"AutoModerator", "[deleted]", "[removed]", None, ""}
BAD_BODIES = {"[removed]", "[deleted]", ""}
# bot-like patterns
BOT_RX = re.compile(r"\bI am a bot\b|\baction was performed automatically\b|^Welcome to /?r/", re.I)


def is_substantive(c):
    body = (c.get("body") or "").strip()
    if body in BAD_BODIES:
        return False
    if c.get("author") in BAD_AUTHORS:
        return False
    if BOT_RX.search(body):
        return False
    if len(body) < MIN_LEN:
        return False
    return True


def thread_to_text(post, comments_kept, all_comments_by_id):
    """Render a post + its surviving comments in threaded order.
    Children grouped under their parent; surviving children of dropped comments
    are still included (we just collapse the dropped link)."""
    lines = []
    lines.append("[POST by u/{}]".format(post.get("author")))
    lines.append("Title: {}".format(post.get("title", "").strip()))
    selftext = (post.get("selftext") or "").strip()
    if selftext:
        lines.append("Body: {}".format(selftext))
    lines.append("")
    lines.append("[COMMENTS]")

    # Build child map
    children = defaultdict(list)
    for c in all_comments_by_id.values():
        pid = c.get("parent_id", "") or ""
        if pid.startswith("t1_"):
            children[pid[3:]].append(c)
        elif pid.startswith("t3_"):
            children["__ROOT__"].append(c)
    # Sort each by score desc
    for k in children:
        children[k].sort(key=lambda c: c.get("score", 0) or 0, reverse=True)

    kept_ids = {c["id"] for c in comments_kept}

    def render(c, depth):
        if c["id"] in kept_ids:
            body = (c.get("body") or "").strip().replace("\n", " ")
            prefix = "  " * depth + "- "
            lines.append("{}u/{} (score={}): {}".format(prefix, c.get("author"), c.get("score"), body))
        for ch in children.get(c["id"], []):
            render(ch, depth + 1 if c["id"] in kept_ids else depth)

    for c in children.get("__ROOT__", []):
        render(c, 0)
    return "\n".join(lines)


def main():
    rows_comments = []
    rows_threads = []
    n_threads_in = 0
    n_threads_kept = 0
    for line in open(SRC):
        rec = json.loads(line)
        n_threads_in += 1
        post = rec["post"]
        comments = rec["comments"]
        all_by_id = {c["id"]: c for c in comments if c.get("id")}
        kept = [c for c in comments if is_substantive(c)]
        if len(kept) < 2:  # need at least 2 substantive comments per thread
            continue
        n_threads_kept += 1
        thread_id = post["id"]
        for c in kept:
            rows_comments.append({
                "thread_id": thread_id,
                "comment_id": c["id"],
                "parent_id": c.get("parent_id"),
                "body": c["body"],
                "score": c.get("score", 0),
                "author": c.get("author"),
                "ts": c.get("created_utc"),
            })
        text = thread_to_text(post, kept, all_by_id)
        rows_threads.append({
            "thread_id": thread_id,
            "post_title": post.get("title", ""),
            "post_author": post.get("author"),
            "n_comments_kept": len(kept),
            "n_comments_total": len(comments),
            "score": post.get("score", 0),
            "num_comments_metadata": post.get("num_comments", 0),
            "permalink": post.get("permalink"),
            "month": post.get("month"),
            "text": text,
        })

    with open(OUT_COMMENTS, "w") as f:
        for r in rows_comments:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_THREADS, "w") as f:
        for r in rows_threads:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("threads in: {}  kept: {}".format(n_threads_in, n_threads_kept))
    print("substantive comments: {}".format(len(rows_comments)))
    print("wrote: {}".format(OUT_COMMENTS))
    print("wrote: {}".format(OUT_THREADS))


if __name__ == "__main__":
    main()
