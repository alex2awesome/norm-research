#!/usr/bin/env python
"""Execute frozen name and holistic target-view orbits over sealed fresh packets."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from methods.codability.experiments.build_fresh_item_partitions import sha256_file, text_sha256
from methods.codability.experiments.validate_fresh_item_partitions import validate_packet
from methods.metric_implementer import config as cfgmod
from methods.metric_implementer.vllm_backend import make_judge_backend


MANIFEST_PATH = Path(__file__).with_name("fresh_target_view_manifest_v1.json")


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict:
    return json.loads(Path(path).read_text())


def _by_id(rows: list[dict]) -> dict[str, dict]:
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("duplicate manifest id")
    return result


def load_domain_items(packet_root: str | Path, domain: str,
                      partitions: list[str] | tuple[str, ...] | None = None) -> dict:
    item_root = Path(packet_root) / domain / "items"
    paths = (
        [item_root / f"{partition}.jsonl" for partition in partitions]
        if partitions is not None else sorted(item_root.glob("*.jsonl"))
    )
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"requested item partitions are missing for {domain}: {missing}")
    if not paths:
        raise ValueError(f"no item partitions for {domain}")
    rows = []
    for path in paths:
        partition = path.stem
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                row["partition"] = partition
                rows.append(row)
    hashes = [row["text_sha256"] for row in rows]
    if len(hashes) != len(set(hashes)):
        raise ValueError(f"duplicate probe hash in {domain}")
    if any(text_sha256(row["text"]) != row["text_sha256"] for row in rows):
        raise ValueError(f"content hash mismatch in {domain}")
    set_hash = hashlib.sha256("\n".join(hashes).encode()).hexdigest()
    return {"rows": rows, "texts": [row["text"] for row in rows], "hashes": hashes,
            "partitions": [row["partition"] for row in rows], "item_set_sha256": set_hash,
            "partition_files": [{"path": str(path), "sha256": sha256_file(path)}
                                for path in paths]}


def _model_tag(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9.-]+", "_", Path(model.rstrip("/")).name)


def score_domain(*, backend, model: str, model_job_id: str, domain_job: dict,
                 cells: dict[str, dict], items: dict, readout_template: str,
                 prompt_manifest_sha256: str, packet_manifest_sha256: str,
                 out_dir: str | Path, repetition: int, overwrite: bool = False) -> dict:
    # This legacy target-view scorer uses the historical top-logprob signature.  Import it only
    # when this scorer runs; breadth imports this module solely for the label-free item loader.
    from methods.metric_implementer.experiments.alpha_probe import signature

    domain = domain_job["domain"]
    out = Path(out_dir) / model_job_id / f"grid_{domain}_{_model_tag(model)}_rep{repetition}.npz"
    sidecar = out.with_suffix(".json")
    if out.exists() and not overwrite:
        with np.load(out, allow_pickle=True) as saved:
            if (str(saved["prompt_manifest_sha256"]) != prompt_manifest_sha256
                    or str(saved["packet_manifest_sha256"]) != packet_manifest_sha256
                    or str(saved["probe_set_sha256"]) != items["item_set_sha256"]):
                raise ValueError(f"stale output exists at {out}; use --overwrite only after audit")
        return {"domain": domain, "status": "already_complete", "path": str(out)}

    task_cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), domain_job["task"])
    scores, meta = [], []
    for cell_id in domain_job["cells"]:
        cell = cells[cell_id]
        if cell["domain"] != domain:
            raise ValueError(f"cell {cell_id} does not belong to {domain}")
        for form in cell["forms"]:
            prompt = form["prompt"]
            scores.append(signature(backend, prompt, items["texts"],
                                    task_cfg.max_text_chars, template=readout_template))
            meta.append({"cell_id": cell_id, "view": cell["view"], "domain": domain,
                         "gi": cell.get("gi"), "construct": cell.get("construct"),
                         "form": form["id"], "prompt_sha256": text_sha256(prompt)})
    score_matrix = np.vstack(scores)
    report = {
        "schema": "fresh_target_view_scores/v1", "model_job_id": model_job_id,
        "model": model, "domain": domain, "task": domain_job["task"],
        "repetition": repetition, "n_items": len(items["rows"]),
        "n_score_rows": len(meta), "cell_ids": domain_job["cells"],
        "probe_set_sha256": items["item_set_sha256"],
        "prompt_manifest_sha256": prompt_manifest_sha256,
        "packet_manifest_sha256": packet_manifest_sha256,
        "readout_template_sha256": text_sha256(readout_template),
        "partition_files": items["partition_files"],
        "nan_rate": float(np.isnan(score_matrix).mean()),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, scores=score_matrix,
        meta=np.asarray([json.dumps(row, sort_keys=True) for row in meta], dtype=object),
        probe_sha256=np.asarray(items["hashes"]),
        probe_partition=np.asarray(items["partitions"]),
        probe_set_sha256=items["item_set_sha256"], reader=model,
        model_job_id=model_job_id, repetition=repetition,
        prompt_manifest_sha256=prompt_manifest_sha256,
        packet_manifest_sha256=packet_manifest_sha256,
        readout_template_sha256=text_sha256(readout_template),
    )
    sidecar.write_text(json.dumps(report, indent=1))
    return {"domain": domain, "status": "complete", "path": str(out),
            "report": str(sidecar), "n_items": len(items["rows"]),
            "n_score_rows": len(meta), "nan_rate": report["nan_rate"]}


def run_model_job(*, model_job_id: str, packet_root: str, packet_manifest: str,
                  out_dir: str, manifest_path: str | Path = MANIFEST_PATH,
                  repetition: int = 0, fake: bool = False,
                  overwrite: bool = False) -> dict:
    requested_domains = {row["domain"] for row in _by_id(
        load_manifest(manifest_path)["model_jobs"])[model_job_id]["domains"]}
    integrity = validate_packet(packet_manifest, domains=requested_domains)
    if not integrity["valid"]:
        raise ValueError(f"fresh packet failed integrity validation: {integrity['errors']}")
    manifest = load_manifest(manifest_path)
    jobs = _by_id(manifest["model_jobs"])
    cells = _by_id(manifest["cells"])
    job = jobs[model_job_id]
    cfg = cfgmod.apply_task_preset(cfgmod.ImplementerConfig(), job["domains"][0]["task"])
    if fake:
        cfg.vllm_fake = True
    backend = make_judge_backend(job["model"], cfg, temperature=None)
    prompt_hash = sha256_file(manifest_path)
    packet_hash = sha256_file(packet_manifest)
    results = []
    for domain_job in job["domains"]:
        results.append(score_domain(
            backend=backend, model=job["model"], model_job_id=model_job_id,
            domain_job=domain_job, cells=cells,
            items=load_domain_items(packet_root, domain_job["domain"]),
            readout_template=manifest["readout_template"],
            prompt_manifest_sha256=prompt_hash, packet_manifest_sha256=packet_hash,
            out_dir=out_dir, repetition=repetition, overwrite=overwrite))
    return {"schema": "fresh_target_view_execution/v1", "model_job_id": model_job_id,
            "model": job["model"], "repetition": repetition, "results": results,
            "prompt_manifest_sha256": prompt_hash, "packet_manifest_sha256": packet_hash}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-job", required=True)
    parser.add_argument("--packet-root", required=True)
    parser.add_argument("--packet-manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--repetition", type=int, default=0)
    parser.add_argument("--fake", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = run_model_job(
        model_job_id=args.model_job, packet_root=args.packet_root,
        packet_manifest=args.packet_manifest, out_dir=args.out_dir,
        manifest_path=args.manifest, repetition=args.repetition,
        fake=args.fake, overwrite=args.overwrite)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
