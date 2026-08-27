#!/bin/bash
# SNL wave-1 audio fetch (56 cut-for-time w/ URLs + 300 stable-hash aired sample).
# Slow politeness rate; resume-safe (skips existing); Whisper transcription on sk3 after.
cd "$(dirname "$0")"
mkdir -p audio_wave1
n=0
while IFS= read -r line; do
  url=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['url'])" "$line" 2>/dev/null)
  key=$(python3 -c "import json,sys,hashlib; r=json.loads(sys.argv[1]); print(hashlib.sha256(r['url'].encode()).hexdigest()[:16])" "$line")
  [ -f "audio_wave1/$key.m4a" ] && continue
  yt-dlp -q -x --audio-format m4a --audio-quality 5 -o "audio_wave1/$key.%(ext)s" "$url" \
    && echo "OK $key" || echo "FAIL $key $url"
  n=$((n+1))
  sleep 20
done < wave1_urls.jsonl
echo SNL_WAVE1_AUDIO_DONE
