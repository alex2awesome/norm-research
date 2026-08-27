#!/usr/bin/env bash
# Organize each datasets/<task>/online-rubrics/ into:
#   - online-rubrics/claude-parsed/   <- curated .md rubric notes (YAML frontmatter, NOT *_raw.md)
#   - online-rubrics/raw/             <- raw HTML/PDF/text fetched by curl (incl. *_raw.md)
#   - online-rubrics/urls-visited.csv (kept at top level)
#
# Never deletes; uses git-mv-style mv. Idempotent: safe to re-run.

set -euo pipefail
ROOT=/Users/spangher/Projects/stanford-research/norm-research/datasets

is_curated_md() {
  # Returns 0 if this looks like a Claude-parsed rubric .md (has YAML frontmatter with source_url)
  local f="$1"
  [ "${f##*.}" = "md" ] || return 1
  case "$(basename "$f")" in *_raw.md) return 1 ;; esac
  head -10 "$f" 2>/dev/null | grep -qE '^(source_url|source|url):' && return 0
  return 1
}

for task in creative-writing peer-review math-stackexchange news-homepages press-releases code-review grant-funding humor legal-outcome-prediction notice-and-comment patents; do
  base="$ROOT/$task/online-rubrics"
  [ -d "$base" ] || { echo "skip missing $base"; continue; }
  mkdir -p "$base/claude-parsed" "$base/raw"

  moved_curated=0
  moved_raw=0
  for f in "$base"/*; do
    [ -f "$f" ] || continue
    name="$(basename "$f")"
    case "$name" in
      urls-visited.csv) continue ;;
      .DS_Store) continue ;;
    esac
    if is_curated_md "$f"; then
      mv "$f" "$base/claude-parsed/$name" && moved_curated=$((moved_curated+1))
    else
      # Anything else at top level (raw HTML/PDF/text, *_raw.md) -> raw/
      # Avoid clobber: if a same-name file already exists in raw/, suffix with -dup<N>
      target="$base/raw/$name"
      if [ -e "$target" ]; then
        i=1
        while [ -e "$base/raw/${name%.*}-dup${i}.${name##*.}" ]; do i=$((i+1)); done
        target="$base/raw/${name%.*}-dup${i}.${name##*.}"
      fi
      mv "$f" "$target" && moved_raw=$((moved_raw+1))
    fi
  done
  cp=$(find "$base/claude-parsed" -maxdepth 1 -type f | wc -l | tr -d ' ')
  rw=$(find "$base/raw" -maxdepth 1 -type f | wc -l | tr -d ' ')
  printf '%-30s moved_curated=%-4d moved_raw=%-4d  totals: claude-parsed=%-4d raw=%-5d\n' \
    "$task" "$moved_curated" "$moved_raw" "$cp" "$rw"
done
