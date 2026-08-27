#!/usr/bin/env python3
"""SNL ASR lane: transcribe ALL manifest audio (both classes) through ONE
Whisper pipeline (faster-whisper large-v3), making the text pipeline identical
by construction. Output: asr_json/<id>.asr.json {id, class, text, segments,
language, duration, model}.

Run (sk2): HOME=/lfs/skampere2/0/alexspan CUDA_VISIBLE_DEVICES=<free gpu> \
    $HOME/envs/fwhisper/bin/python transcribe_asr.py --audio_dir audio_wave1 \
    --manifest snl_asr_manifest.jsonl --out asr_json
Resume-safe: skips ids whose .asr.json already exists.
"""
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="snl_asr_manifest.jsonl")
    ap.add_argument("--audio_dir", default="audio_wave1")
    ap.add_argument("--out", default="asr_json")
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compute_type", default="float16")
    args = ap.parse_args()

    from faster_whisper import WhisperModel

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    rows = [json.loads(l) for l in open(args.manifest)]
    todo = []
    for r in rows:
        ap_ = Path(args.audio_dir) / f"{r['id']}.m4a"
        op = out / f"{r['id']}.asr.json"
        if op.exists():
            continue
        if not ap_.exists():
            print(f"MISSING_AUDIO {r['id']} {r['class']}", flush=True)
            continue
        todo.append((r, ap_, op))
    print(f"todo={len(todo)} of {len(rows)}", flush=True)

    model = WhisperModel(args.model, device=args.device,
                         compute_type=args.compute_type)
    print("MODEL_LOADED", flush=True)

    for i, (r, ap_, op) in enumerate(todo):
        t0 = time.time()
        try:
            segs, info = model.transcribe(str(ap_), language="en",
                                          vad_filter=True, beam_size=5)
            seg_list = [{"start": round(s.start, 2), "end": round(s.end, 2),
                         "text": s.text} for s in segs]
        except Exception as e:  # noqa: BLE001
            print(f"ASR_FAIL {r['id']} {e}", flush=True)
            continue
        text = " ".join(s["text"].strip() for s in seg_list).strip()
        op.write_text(json.dumps({
            "id": r["id"], "class": r["class"], "season": r["season"],
            "title": r["title"], "url": r["url"], "model": args.model,
            "language": info.language, "audio_duration": round(info.duration, 1),
            "text": text, "segments": seg_list,
        }))
        print(f"DONE {i+1}/{len(todo)} {r['id']} {r['class']} "
              f"chars={len(text)} took={time.time()-t0:.0f}s", flush=True)

    print("SNL_ASR_TRANSCRIBE_DONE", flush=True)


if __name__ == "__main__":
    main()
