#!/bin/bash
# SNL ASR-lane audio fetch: all snl_asr_manifest.jsonl entries (YouTube URLs only,
# verified at manifest build). Resume-safe (skips existing m4a); <=2 attempts per
# video; polite rate limit. Cut-for-time 36 already on disk from wave1.
cd "$(dirname "$0")"
mkdir -p audio_wave1
while IFS= read -r line; do
  url=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['url'])" "$line")
  key=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['id'])" "$line")
  case "$url" in
    *youtube.com*|*youtu.be*) ;;
    *) echo "SKIP_NONYT $key $url"; continue ;;
  esac
  [ -f "audio_wave1/$key.m4a" ] && continue
  ok=""
  for attempt in 1 2; do
    if yt-dlp -q -x --audio-format m4a --audio-quality 5 --no-playlist \
         -o "audio_wave1/$key.%(ext)s" "$url"; then
      ok=1; break
    fi
    sleep 30
  done
  if [ -n "$ok" ]; then echo "OK $key"; else echo "FAIL $key $url"; fi
  sleep 15
done < snl_asr_manifest.jsonl
echo SNL_ASR_AUDIO_DONE
