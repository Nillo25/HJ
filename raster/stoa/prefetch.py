"""Fill the Claude cache through the Batch API (50% price) before running the bench.

python -m stoa.prefetch --list reports/eval80.txt --kind opus_batch --model claude-opus-5-5 --effort medium
kinds: opus_batch  - the per-plan counting call used by router route "opus"
       verify_hj   - windows+doors verification over colleague YOLO boxes (see verify.py)
"""
import argparse

from . import claude_vlm, config


def jobs_for(kind, paths, model, effort):
    jobs = []
    for p in paths:
        if kind == "opus_batch":
            jobs.append(dict(image_path=p, prompt=claude_vlm.prompt_batch(boxes=False),
                             schema=claude_vlm.schema_batch(boxes=False), model=model, effort=effort))
        elif kind == "verify_hj":
            from . import verify
            jobs.append(verify.job_hj(p, model=model, effort=effort))
        else:
            raise ValueError(kind)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--kind", default="opus_batch")
    ap.add_argument("--model", default=None)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--est", type=float, default=0.03, help="estimated list price per request (USD) for the cap check")
    a = ap.parse_args()
    paths = [config.DATASET_DIR / l.strip() for l in open(a.list) if l.strip()]
    n = claude_vlm.call_batch(jobs_for(a.kind, paths, a.model, a.effort), est_cost_each=a.est)
    print(f"sent {n} requests")


if __name__ == "__main__":
    main()
